from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.causal_graph_learner import (
    DAGMALogDetConstraint,
    LocalLinear,
    adjacency_directionality,
    normalized_dag_loss,
    normalized_l1_loss,
)


def _inverse_softplus(value):
    value = torch.as_tensor(value).clamp_min(1e-8)
    return torch.log(torch.expm1(value))


@dataclass
class TemporalSEMOutput:
    x_hat: torch.Tensor
    target: torch.Tensor
    a0: torch.Tensor
    a_lag: torch.Tensor
    a_lag_raw: torch.Tensor | None
    a_shared: torch.Tensor
    a_effective: torch.Tensor
    dag_penalty: torch.Tensor
    dag_metadata: dict
    a_delta: torch.Tensor | None = None
    normalized_input: torch.Tensor | None = None
    signed_sample_graph: torch.Tensor | None = None
    candidate_lag_mask: torch.Tensor | None = None
    prediction_baseline: torch.Tensor | None = None
    innovation_hat: torch.Tensor | None = None
    innovation_target: torch.Tensor | None = None

    @property
    def c_hat(self):
        return self.x_hat

    @property
    def adjacency(self):
        return self.a_effective


class TemporalSEMCausalLearner(nn.Module):
    """Temporal NTS-NOTEARS learner for module 2.

    The main graph is A_lag: historical ROI signals predict future ROI signals.
    A0 models weak contemporaneous residual dependency and receives the DAG
    penalty. A_lag is not forced to be DAG because temporal order already fixes
    the direction from past to future.
    """

    def __init__(
        self,
        n_nodes,
        lag_order=3,
        init_logit=-3.0,
        dagma_logdet_s=1.0,
        dagma_logdet_margin=0.1,
        dagma_power_iters=5,
        input_norm="time_zscore",
        use_sample_graph_residual=False,
        sample_graph_delta_scale=0.02,
        sample_lag_graph_mode="abs",
        sample_graph_rank=4,
        lambda_pred=1.0,
        lambda_dag=0.001,
        lambda_sparse=0.0005,
        lambda_smooth=0.0001,
        lambda_group_sparse=0.0,
        lambda_lag_hierarchy=0.0,
        prediction_loss_mode="bold_alff",
        pred_huber_delta=1.0,
        lambda_pred_delta=0.2,
        lambda_pred_lowfreq=0.2,
        lambda_pred_corr=0.05,
        lowfreq_kernel_size=9,
        a0_sparse_ratio=0.2,
        candidate_parent_topk=0,
        lambda_sample_l1=0.0,
        lambda_sample_deviation=0.0,
        dagma_warmup_epochs=5,
        dagma_barrier_epochs=20,
        reg_warmup_epochs=0,
        graph_hidden_dim=None,
        decoder_activation="identity",
        prediction_target_mode="innovation",
        a0_scale=0.03,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.lag_order = max(int(lag_order), 1)
        self.input_norm = str(input_norm or "none").lower()
        self.use_sample_graph_residual = bool(use_sample_graph_residual)
        self.sample_graph_delta_scale = float(sample_graph_delta_scale)
        self.sample_lag_graph_mode = str(sample_lag_graph_mode or "abs").lower()
        if self.sample_lag_graph_mode not in ("abs", "positive", "signed_abs"):
            raise ValueError(
                f"Unsupported sample_lag_graph_mode={self.sample_lag_graph_mode!r}. "
                "Use 'abs', 'positive' or 'signed_abs'."
            )
        self.sample_graph_rank = max(int(sample_graph_rank), 1)
        self.lambda_pred = float(lambda_pred)
        self.lambda_dag = float(lambda_dag)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_group_sparse = float(lambda_group_sparse)
        self.lambda_lag_hierarchy = float(lambda_lag_hierarchy)
        self.prediction_loss_mode = str(prediction_loss_mode or "mse").lower()
        if self.prediction_loss_mode not in ("mse", "huber", "bold_alff"):
            raise ValueError(
                f"Unsupported temporal prediction loss mode={self.prediction_loss_mode!r}. "
                "Use 'mse', 'huber' or 'bold_alff'."
            )
        self.pred_huber_delta = max(float(pred_huber_delta), 1e-6)
        self.lambda_pred_delta = float(lambda_pred_delta)
        self.lambda_pred_lowfreq = float(lambda_pred_lowfreq)
        self.lambda_pred_corr = float(lambda_pred_corr)
        self.lowfreq_kernel_size = max(int(lowfreq_kernel_size), 1)
        if self.lowfreq_kernel_size % 2 == 0:
            self.lowfreq_kernel_size += 1
        self.a0_sparse_ratio = max(float(a0_sparse_ratio), 0.0)
        self.candidate_parent_topk = max(int(candidate_parent_topk), 0)
        self.lambda_sample_l1 = float(lambda_sample_l1)
        self.lambda_sample_deviation = float(lambda_sample_deviation)
        self.dagma_warmup_epochs = max(int(dagma_warmup_epochs), 0)
        self.dagma_barrier_epochs = max(int(dagma_barrier_epochs), 1)
        self.reg_warmup_epochs = max(int(reg_warmup_epochs), 0)
        self.graph_hidden_dim = int(graph_hidden_dim or 8)
        self.decoder_activation = str(decoder_activation or "identity").lower()
        if self.decoder_activation not in ("sigmoid", "tanh", "gelu", "identity", "none"):
            raise ValueError(
                f"Unsupported temporal decoder activation={self.decoder_activation!r}. "
                "Use 'tanh', 'sigmoid', 'gelu' or 'identity'."
            )
        self.prediction_target_mode = str(prediction_target_mode or "innovation").lower()
        if self.prediction_target_mode not in ("innovation", "next_value"):
            raise ValueError(
                f"Unsupported temporal prediction_target_mode={self.prediction_target_mode!r}. "
                "Use 'innovation' or 'next_value'."
            )
        self.a0_scale = min(max(float(a0_scale), 0.0), 1.0)
        self.current_epoch = 0

        init_edge_strength = torch.sigmoid(torch.tensor(float(init_logit))).item()
        init_weight_scale = math.sqrt(max(init_edge_strength, 1e-4) / self.graph_hidden_dim)
        lag_pos = torch.rand(
            self.lag_order, self.n_nodes, self.graph_hidden_dim, self.n_nodes
        ) * init_weight_scale
        lag_neg = torch.rand_like(lag_pos) * init_weight_scale
        a0_pos = torch.rand(self.n_nodes, self.graph_hidden_dim, self.n_nodes) * init_weight_scale
        a0_neg = torch.rand_like(a0_pos) * init_weight_scale
        self.lag_pos_raw = nn.Parameter(_inverse_softplus(lag_pos))
        self.lag_neg_raw = nn.Parameter(_inverse_softplus(lag_neg))
        self.a0_pos_raw = nn.Parameter(_inverse_softplus(a0_pos))
        self.a0_neg_raw = nn.Parameter(_inverse_softplus(a0_neg))

        self.lag_decoder = nn.ModuleList(
            [
                LocalLinear(self.n_nodes, self.graph_hidden_dim, 1, bias=False)
                for _ in range(self.lag_order)
            ]
        )
        self.a0_decoder = LocalLinear(self.n_nodes, self.graph_hidden_dim, 1, bias=False)

        self.sample_left = nn.Linear(1, self.sample_graph_rank)
        self.sample_right = nn.Linear(1, self.sample_graph_rank)

        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.n_nodes))
        self.register_buffer("child_parent_mask", (1.0 - torch.eye(self.n_nodes)).T)
        self.dag_constraint = DAGMALogDetConstraint(
            num_nodes=self.n_nodes,
            s=dagma_logdet_s,
            margin=dagma_logdet_margin,
            power_iters=dagma_power_iters,
        )

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _normalize_input(self, x):
        if self.input_norm in ("none", "off", "false", "0"):
            return x
        if self.input_norm in ("time_zscore", "zscore"):
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
            return (x - mean) / std
        if self.input_norm == "batch_zscore":
            mean = x.mean(dim=(0, 1), keepdim=True)
            std = x.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-5)
            return (x - mean) / std
        raise ValueError(
            f"Unsupported temporal input norm={self.input_norm!r}. "
            "Use 'none', 'time_zscore' or 'batch_zscore'."
        )

    def _as_4d(self, x):
        if x.ndim == 3:
            return x.unsqueeze(-1), True
        if x.ndim == 4:
            return x, False
        raise ValueError(f"Expected temporal input [B, T, N] or [B, T, N, D], got {tuple(x.shape)}.")

    def _off_diag(self, matrix):
        mask = self.off_diag_mask.to(device=matrix.device, dtype=matrix.dtype)
        if matrix.ndim == 2:
            return matrix * mask
        if matrix.ndim == 3:
            return matrix * mask.unsqueeze(0)
        raise ValueError(f"Unexpected graph shape {tuple(matrix.shape)}.")

    def _masked_lag_weights(self):
        pos = F.softplus(self.lag_pos_raw)
        neg = F.softplus(self.lag_neg_raw)
        mask = self.child_parent_mask.to(device=pos.device, dtype=pos.dtype).view(1, self.n_nodes, 1, self.n_nodes)
        return pos * mask, neg * mask

    def _masked_a0_weights(self):
        pos = F.softplus(self.a0_pos_raw)
        neg = F.softplus(self.a0_neg_raw)
        mask = self.child_parent_mask.to(device=pos.device, dtype=pos.dtype).view(self.n_nodes, 1, self.n_nodes)
        return pos * mask, neg * mask

    def first_layer_l1_loss(self):
        lag_pos, lag_neg = self._masked_lag_weights()
        a0_pos, a0_neg = self._masked_a0_weights()
        lag_count = self.lag_order * self.n_nodes * max(self.n_nodes - 1, 1) * self.graph_hidden_dim
        a0_count = self.n_nodes * max(self.n_nodes - 1, 1) * self.graph_hidden_dim
        return (lag_pos + lag_neg).sum() / max(float(lag_count), 1.0) + (
            a0_pos + a0_neg
        ).sum() / max(float(a0_count), 1.0)

    def shared_graphs(self, candidate_lag_mask=None):
        lag_pos, lag_neg = self._masked_lag_weights()
        a0_pos, a0_neg = self._masked_a0_weights()
        lag_weight = lag_pos - lag_neg
        a0_weight = a0_pos - a0_neg
        # Shape convention follows CausalGraphLearner: rows are parent/source,
        # columns are child/target after transpose.
        a_lag = torch.sqrt(torch.sum(lag_weight * lag_weight, dim=2).transpose(1, 2) + 1e-12)
        if candidate_lag_mask is not None:
            a_lag = a_lag * candidate_lag_mask.to(device=a_lag.device, dtype=a_lag.dtype)
        a0 = self.a0_scale * torch.sqrt(torch.sum(a0_weight * a0_weight, dim=1).T + 1e-12)
        return self._off_diag(a0), self._off_diag(a_lag)

    def _build_windows(self, x):
        if x.shape[1] <= self.lag_order:
            raise ValueError(
                f"Temporal length {x.shape[1]} must be larger than lag_order={self.lag_order}."
            )
        target = x[:, self.lag_order :, :, :]
        histories = []
        for lag_idx in range(self.lag_order):
            lag = lag_idx + 1
            histories.append(x[:, self.lag_order - lag : x.shape[1] - lag, :, :])
        return target, histories

    def _decode_hidden(self, hidden, decoder):
        bsz, steps, nodes, hidden_dim, feature_dim = hidden.shape
        hidden_flat = hidden.permute(0, 1, 4, 2, 3).reshape(bsz * steps, feature_dim, nodes, hidden_dim)
        if self.decoder_activation == "sigmoid":
            decoded = torch.sigmoid(hidden_flat)
        elif self.decoder_activation == "gelu":
            decoded = F.gelu(hidden_flat)
        elif self.decoder_activation == "tanh":
            decoded = torch.tanh(hidden_flat)
        else:
            decoded = hidden_flat
        decoded = decoder(decoded).squeeze(-1)
        return decoded.reshape(bsz, steps, feature_dim, nodes).permute(0, 1, 3, 2)

    def _predict(self, histories, candidate_lag_mask=None):
        lag_pos, lag_neg = self._masked_lag_weights()
        a0_pos, a0_neg = self._masked_a0_weights()
        if candidate_lag_mask is not None:
            weight_mask = candidate_lag_mask.to(device=lag_pos.device, dtype=lag_pos.dtype)
            weight_mask = weight_mask.permute(0, 2, 1).unsqueeze(2)
            lag_pos = lag_pos * weight_mask
            lag_neg = lag_neg * weight_mask
        # 对零均值 BOLD 使用持久性预测作为无因果基线，跨 ROI 参数只学习
        # x_t - x_{t-1} 的创新量，避免靠每个 ROI 的常数均值降低损失。
        prediction_baseline = (
            histories[0]
            if self.prediction_target_mode == "innovation"
            else torch.zeros_like(histories[0])
        )
        innovation_hat = torch.zeros_like(histories[0])
        for lag_idx, history in enumerate(histories):
            weight = lag_pos[lag_idx] - lag_neg[lag_idx]
            hidden = torch.einsum("btpd,jmp->btjmd", history, weight)
            innovation_hat = innovation_hat + self._decode_hidden(hidden, self.lag_decoder[lag_idx])

        # A0 uses the prediction itself as contemporaneous residual input. This
        # avoids leaking the true target into the prediction objective.
        a0_weight = self.a0_scale * (a0_pos - a0_neg)
        residual_hidden = torch.einsum("btpd,jmp->btjmd", innovation_hat, a0_weight)
        innovation_hat = innovation_hat + self._decode_hidden(residual_hidden, self.a0_decoder)
        return prediction_baseline + innovation_hat, prediction_baseline, innovation_hat

    def _lagged_correlation_stack(self, histories, target):
        lag_graphs = []
        for history in histories:
            history_centered = history - history.mean(dim=(1, 3), keepdim=True)
            target_centered = target - target.mean(dim=(1, 3), keepdim=True)
            bsz, steps, nodes, feat_dim = history_centered.shape
            history_flat = history_centered.permute(0, 2, 1, 3).reshape(bsz, nodes, steps * feat_dim)
            target_flat = target_centered.permute(0, 2, 1, 3).reshape(bsz, nodes, steps * feat_dim)
            numerator = torch.einsum("bpm,bcm->bpc", history_flat, target_flat)
            history_norm = history_flat.square().sum(dim=-1).clamp_min(1e-8).sqrt()
            target_norm = target_flat.square().sum(dim=-1).clamp_min(1e-8).sqrt()
            correlation = numerator / (history_norm.unsqueeze(-1) * target_norm.unsqueeze(-2))
            lag_graphs.append(correlation)
        return torch.stack(lag_graphs, dim=0)

    def _candidate_parent_mask(self, histories, target, lag_corr_stack=None):
        if self.candidate_parent_topk <= 0:
            return None
        if lag_corr_stack is None:
            lag_corr_stack = self._lagged_correlation_stack(histories, target)
        lag_corr = lag_corr_stack.abs().mean(dim=1)
        off_diag = self.off_diag_mask.to(device=lag_corr.device, dtype=lag_corr.dtype).unsqueeze(0)
        lag_corr = lag_corr * off_diag
        n_nodes = lag_corr.shape[-1]
        k = min(max(self.candidate_parent_topk, 1), max(n_nodes - 1, 1))
        _, top_indices = torch.topk(lag_corr, k=k, dim=1)
        mask = torch.zeros_like(lag_corr).scatter_(1, top_indices, 1.0)
        return mask * off_diag

    def _sample_lagged_correlation_graph(self, histories, target, lag_corr_stack=None):
        if lag_corr_stack is None:
            lag_corr_stack = self._lagged_correlation_stack(histories, target)
        signed_graph = lag_corr_stack.mean(dim=0)
        if self.sample_lag_graph_mode == "positive":
            graph = lag_corr_stack.clamp_min(0.0).mean(dim=0)
        elif self.sample_lag_graph_mode == "signed_abs":
            graph = (lag_corr_stack.abs() * lag_corr_stack.sign().clamp_min(0.0)).mean(dim=0)
        else:
            graph = lag_corr_stack.abs().mean(dim=0)
        off_diag = self.off_diag_mask.to(device=graph.device, dtype=graph.dtype).unsqueeze(0)
        return graph * off_diag, signed_graph * off_diag

    def _sample_residual_graph(self, x, a_shared, histories=None, target=None, lag_corr_stack=None):
        if not self.use_sample_graph_residual or self.sample_graph_delta_scale <= 0:
            return None, a_shared, None
        if histories is not None and target is not None:
            sample_graph, signed_graph = self._sample_lagged_correlation_graph(
                histories,
                target,
                lag_corr_stack=lag_corr_stack,
            )
            shared = a_shared.unsqueeze(0).to(device=sample_graph.device, dtype=sample_graph.dtype)
            shared_mass = shared.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            sample_mass = sample_graph.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            sample_graph = sample_graph * (shared_mass / sample_mass)
            signed_abs_mass = signed_graph.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            signed_graph = signed_graph * (shared_mass / signed_abs_mass)
            blend = min(max(self.sample_graph_delta_scale, 0.0), 1.0)
            a_delta = (sample_graph - shared) * blend
        else:
            signed_graph = None
            node_summary = x.std(dim=(1, 3), keepdim=False).unsqueeze(-1)
            left = self.sample_left(node_summary)
            right = self.sample_right(node_summary)
            low_rank = torch.einsum("bnr,bmr->bnm", left, right) / math.sqrt(float(self.sample_graph_rank))
            a_delta = torch.tanh(low_rank) * self.sample_graph_delta_scale
        off_diag = self.off_diag_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        a_delta = a_delta * off_diag
        a_effective = torch.clamp(a_shared.unsqueeze(0) + a_delta, min=0.0) * off_diag
        signed_graph = None if signed_graph is None else signed_graph * off_diag
        return a_delta, a_effective, signed_graph

    def _dagma_schedule(self):
        if self.current_epoch < self.dagma_warmup_epochs:
            denom = max(float(self.dagma_warmup_epochs), 1.0)
            scale = 0.1 + 0.9 * ((self.current_epoch + 1) / denom)
            return "warmup", min(scale, 1.0)
        barrier_start = self.dagma_warmup_epochs
        barrier_end = barrier_start + self.dagma_barrier_epochs
        if self.current_epoch < barrier_end:
            progress = (self.current_epoch - barrier_start + 1) / max(float(self.dagma_barrier_epochs), 1.0)
            return "barrier", 1.0 + 4.0 * min(max(progress, 0.0), 1.0)
        return "refine", 5.0

    def _reg_schedule(self):
        if self.reg_warmup_epochs <= 0:
            return 1.0
        scale = min(max((self.current_epoch + 1) / float(self.reg_warmup_epochs), 0.0), 1.0)
        return 0.1 + 0.9 * scale

    def smoothness_loss(self, a_lag):
        if a_lag.shape[0] <= 1:
            return torch.zeros((), device=a_lag.device, dtype=a_lag.dtype)
        lag_steps = torch.arange(
            1,
            a_lag.shape[0],
            device=a_lag.device,
            dtype=a_lag.dtype,
        ).view(-1, 1, 1)
        # 远 lag 的 BOLD 影响通常更弱、更不稳定，因此相邻 lag 差异按滞后阶数轻微加权。
        return ((a_lag[1:] - a_lag[:-1]).abs() * lag_steps.sqrt()).mean()

    def direct_graph_sparse_loss(self, a_lag, a0):
        off_diag = self.off_diag_mask.to(device=a_lag.device, dtype=a_lag.dtype)
        lag_weights = torch.arange(
            1,
            a_lag.shape[0] + 1,
            device=a_lag.device,
            dtype=a_lag.dtype,
        ).view(-1, 1, 1)
        # A_lag 是模块2的主因果图；越远的 lag 越容易是间接传播或噪声，稀疏惩罚略强。
        lag_sparse = (a_lag * off_diag.unsqueeze(0) * lag_weights.sqrt()).sum()
        lag_sparse = lag_sparse / (off_diag.sum().clamp_min(1.0) * lag_weights.sum().clamp_min(1.0))
        a0_sparse = (a0 * off_diag).sum() / off_diag.sum().clamp_min(1.0)
        return lag_sparse + self.a0_sparse_ratio * a0_sparse

    def group_sparse_loss(self, a_lag):
        off_diag = self.off_diag_mask.to(device=a_lag.device, dtype=a_lag.dtype)
        group_norm = torch.sqrt(a_lag.square().sum(dim=0) + 1e-12) * off_diag
        return group_norm.sum() / off_diag.sum().clamp_min(1.0)

    def lag_hierarchy_loss(self, a_lag):
        off_diag = self.off_diag_mask.to(device=a_lag.device, dtype=a_lag.dtype)
        parts = []
        for lag_idx in range(a_lag.shape[0]):
            tail_norm = torch.sqrt(a_lag[lag_idx:].square().sum(dim=0) + 1e-12) * off_diag
            parts.append(tail_norm.sum() / off_diag.sum().clamp_min(1.0))
        return torch.stack(parts).mean()

    def _low_frequency_view(self, x):
        if self.lowfreq_kernel_size <= 1 or x.shape[1] <= 1:
            return x
        kernel_size = min(self.lowfreq_kernel_size, x.shape[1])
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size <= 1:
            return x
        bsz, steps, nodes, feat_dim = x.shape
        series = x.permute(0, 2, 3, 1).reshape(bsz * nodes * feat_dim, 1, steps)
        padding = kernel_size // 2
        smooth = F.avg_pool1d(series, kernel_size=kernel_size, stride=1, padding=padding)
        return smooth.reshape(bsz, nodes, feat_dim, steps).permute(0, 3, 1, 2)

    def _loss_as_4d(self, x):
        if x.ndim == 3:
            return x.unsqueeze(-1)
        if x.ndim == 4:
            return x
        raise ValueError(
            f"Temporal prediction loss expects [B,T,N] or [B,T,N,D], got shape={tuple(x.shape)}"
        )

    def _temporal_corr_loss(self, x_hat, target):
        pred = x_hat - x_hat.mean(dim=1, keepdim=True)
        truth = target - target.mean(dim=1, keepdim=True)
        numerator = (pred * truth).sum(dim=1)
        pred_norm = pred.square().sum(dim=1).clamp_min(1e-8).sqrt()
        truth_norm = truth.square().sum(dim=1).clamp_min(1e-8).sqrt()
        corr = numerator / (pred_norm * truth_norm)
        return (1.0 - corr).mean()

    def prediction_loss(self, x_hat, target, innovation_hat=None, innovation_target=None):
        x_hat_4d = self._loss_as_4d(x_hat)
        target_4d = self._loss_as_4d(target)
        base_pred = x_hat_4d
        base_target = target_4d
        if (
            self.prediction_target_mode == "innovation"
            and innovation_hat is not None
            and innovation_target is not None
        ):
            base_pred = self._loss_as_4d(innovation_hat)
            base_target = self._loss_as_4d(innovation_target)
        if self.prediction_loss_mode == "mse":
            base = F.mse_loss(base_pred, base_target)
        else:
            base = F.huber_loss(base_pred, base_target, delta=self.pred_huber_delta)
        pred_std = x_hat_4d.std(dim=1, unbiased=False).mean()
        target_std = target_4d.std(dim=1, unbiased=False).mean().clamp_min(1e-8)
        std_ratio = pred_std / target_std
        if self.prediction_loss_mode != "bold_alff":
            zero = torch.zeros((), device=x_hat_4d.device, dtype=x_hat_4d.dtype)
            return base, {
                "temporal_pred_base_loss": base,
                "temporal_pred_delta_loss": zero,
                "temporal_pred_lowfreq_loss": zero,
                "temporal_pred_corr_loss": zero,
                "temporal_pred_std_ratio": std_ratio.detach(),
                "temporal_pred_corr_value": zero,
            }

        if x_hat_4d.shape[1] > 1:
            delta_pred = x_hat_4d[:, 1:] - x_hat_4d[:, :-1]
            delta_true = target_4d[:, 1:] - target_4d[:, :-1]
            delta_loss = F.huber_loss(delta_pred, delta_true, delta=self.pred_huber_delta)
        else:
            delta_loss = torch.zeros((), device=x_hat_4d.device, dtype=x_hat_4d.dtype)
        low_pred = self._low_frequency_view(x_hat_4d)
        low_true = self._low_frequency_view(target_4d)
        lowfreq_loss = F.huber_loss(low_pred, low_true, delta=self.pred_huber_delta)
        corr_loss = self._temporal_corr_loss(x_hat_4d, target_4d)
        total = (
            base
            + self.lambda_pred_delta * delta_loss
            + self.lambda_pred_lowfreq * lowfreq_loss
            + self.lambda_pred_corr * corr_loss
        )
        return total, {
            "temporal_pred_base_loss": base,
            "temporal_pred_delta_loss": delta_loss,
            "temporal_pred_lowfreq_loss": lowfreq_loss,
            "temporal_pred_corr_loss": corr_loss,
            "temporal_pred_std_ratio": std_ratio.detach(),
            "temporal_pred_corr_value": (1.0 - corr_loss).detach(),
        }

    def sample_graph_l1_loss(self, a_delta):
        if a_delta is None:
            return torch.zeros((), device=self.off_diag_mask.device)
        return normalized_l1_loss(a_delta)

    def sample_graph_deviation_loss(self, a_effective, a_shared):
        if a_effective.ndim != 3:
            return torch.zeros((), device=a_shared.device, dtype=a_shared.dtype)
        shared = a_shared.unsqueeze(0).to(device=a_effective.device, dtype=a_effective.dtype)
        off_diag = self.off_diag_mask.to(device=a_effective.device, dtype=a_effective.dtype).unsqueeze(0)
        return (((a_effective - shared) * off_diag) ** 2).mean()

    def compute_losses(self, output):
        pred_loss, pred_parts = self.prediction_loss(
            output.x_hat,
            output.target,
            innovation_hat=output.innovation_hat,
            innovation_target=output.innovation_target,
        )
        dag_loss = normalized_dag_loss(output.dag_penalty, self.n_nodes)
        parameter_sparse_loss = self.first_layer_l1_loss()
        sparse_loss = self.direct_graph_sparse_loss(output.a_lag, output.a0)
        smooth_loss = self.smoothness_loss(output.a_lag)
        group_sparse_loss = self.group_sparse_loss(output.a_lag)
        lag_hierarchy_loss = self.lag_hierarchy_loss(output.a_lag)
        sample_l1 = self.sample_graph_l1_loss(output.a_delta)
        sample_dev = self.sample_graph_deviation_loss(output.a_effective, output.a_shared)
        stage = output.dag_metadata["dagma_stage"]
        dag_scale = output.dag_metadata["dagma_effective_scale"]
        reg_scale = self._reg_schedule()
        aux_loss = (
            self.lambda_pred * pred_loss
            + reg_scale
            * (
                self.lambda_dag * dag_scale * dag_loss
                + self.lambda_sparse * sparse_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_group_sparse * group_sparse_loss
                + self.lambda_lag_hierarchy * lag_hierarchy_loss
                + self.lambda_sample_l1 * sample_l1
                + self.lambda_sample_deviation * sample_dev
            )
        )
        parts = {
            "temporal_pred_loss": pred_loss,
            **pred_parts,
            "causal_dag_loss": dag_loss,
            "temporal_sparse_loss": sparse_loss,
            "temporal_parameter_sparse_loss": parameter_sparse_loss,
            "temporal_smooth_loss": smooth_loss,
            "temporal_group_sparse_loss": group_sparse_loss,
            "temporal_lag_hierarchy_loss": lag_hierarchy_loss,
            "sample_graph_l1_loss": sample_l1,
            "sample_graph_deviation_loss": sample_dev,
            "causal_aux_loss": aux_loss,
            "temporal_reg_scale": torch.as_tensor(reg_scale, dtype=aux_loss.dtype, device=aux_loss.device),
            "causal_dag_weighted_loss": self.lambda_dag * dag_scale * dag_loss,
            "temporal_sparse_weighted_loss": self.lambda_sparse * sparse_loss,
            "temporal_parameter_sparse_weighted_loss": self.lambda_sparse * parameter_sparse_loss,
            "temporal_smooth_weighted_loss": self.lambda_smooth * smooth_loss,
            "temporal_group_sparse_weighted_loss": self.lambda_group_sparse * group_sparse_loss,
            "temporal_lag_hierarchy_weighted_loss": self.lambda_lag_hierarchy * lag_hierarchy_loss,
        }
        parts["causal_meta_dagma_stage_id"] = torch.as_tensor(
            {"warmup": 0, "barrier": 1, "refine": 2}[stage],
            dtype=aux_loss.dtype,
            device=aux_loss.device,
        )
        for key, value in output.dag_metadata.items():
            if isinstance(value, (int, float)):
                parts[f"causal_meta_{key}"] = torch.as_tensor(
                    value,
                    dtype=aux_loss.dtype,
                    device=aux_loss.device,
                )
        return aux_loss, parts

    def forward(self, x):
        x4, squeezed = self._as_4d(x)
        x4 = self._normalize_input(x4)
        target, histories = self._build_windows(x4)
        needs_lag_corr = self.candidate_parent_topk > 0 or (
            self.use_sample_graph_residual and self.sample_graph_delta_scale > 0
        )
        innovation_target = (
            target - histories[0]
            if self.prediction_target_mode == "innovation"
            else target
        )
        lag_corr_stack = (
            self._lagged_correlation_stack(histories, innovation_target)
            if needs_lag_corr
            else None
        )
        candidate_lag_mask = self._candidate_parent_mask(
            histories,
            innovation_target,
            lag_corr_stack=lag_corr_stack,
        )
        a0, a_lag_raw = self.shared_graphs(candidate_lag_mask=None)
        if candidate_lag_mask is not None:
            a_lag = a_lag_raw * candidate_lag_mask.to(device=a_lag_raw.device, dtype=a_lag_raw.dtype)
        else:
            a_lag = a_lag_raw
        x_hat, prediction_baseline, innovation_hat = self._predict(
            histories,
            candidate_lag_mask=candidate_lag_mask,
        )
        a_lag_mean = a_lag.mean(dim=0)
        a_shared = a_lag_mean * self.off_diag_mask.to(a_lag_mean.device, a_lag_mean.dtype)
        a_delta, a_effective, signed_sample_graph = self._sample_residual_graph(
            x4,
            a_shared,
            histories=histories,
            target=innovation_target,
            lag_corr_stack=lag_corr_stack,
        )
        penalty = self.dag_constraint(a0)
        stage, dag_scale = self._dagma_schedule()
        metadata = {
            "dag_method": "dagma_logdet",
            "graph_style": "temporal_nts_notears",
            "dagma_stage": stage,
            "dagma_effective_scale": float(dag_scale),
            "lag_order": self.lag_order,
            "decoder_activation_id": {"sigmoid": 0, "tanh": 1, "gelu": 2, "identity": 3, "none": 3}[
                self.decoder_activation
            ],
            "use_sample_graph_residual": self.use_sample_graph_residual,
            "candidate_parent_topk": self.candidate_parent_topk,
            "prediction_target_mode_id": 1.0 if self.prediction_target_mode == "innovation" else 0.0,
            "a0_scale": self.a0_scale,
        }
        metadata.update(self.dag_constraint.metadata())
        for key, value in adjacency_directionality(a0).items():
            metadata[f"a0_{key}"] = value.item()
        for key, value in adjacency_directionality(a_lag_mean).items():
            metadata[f"alag_mean_{key}"] = value.item()
        for key, value in adjacency_directionality(a_shared).items():
            metadata[f"shared_{key}"] = value.item()
        metadata["a0_to_alag_mass_ratio"] = float(
            a0.abs().mean().detach() / a_lag_mean.abs().mean().detach().clamp_min(1e-8)
        )
        if a_delta is not None:
            metadata["sample_delta_abs_mean"] = a_delta.abs().mean().detach().item()
            metadata["sample_delta_abs_max"] = a_delta.abs().max().detach().item()
        if candidate_lag_mask is not None:
            metadata["candidate_mask_density"] = candidate_lag_mask.mean().detach().item()

        if squeezed:
            x_hat = x_hat.squeeze(-1)
            target = target.squeeze(-1)
            normalized_input = x4.squeeze(-1)
            prediction_baseline = prediction_baseline.squeeze(-1)
            innovation_hat = innovation_hat.squeeze(-1)
            innovation_target = innovation_target.squeeze(-1)
        else:
            normalized_input = x4

        return TemporalSEMOutput(
            x_hat=x_hat,
            target=target,
            a0=a0,
            a_lag=a_lag,
            a_lag_raw=a_lag_raw,
            a_shared=a_shared,
            a_effective=a_effective,
            dag_penalty=penalty,
            dag_metadata=metadata,
            a_delta=a_delta,
            normalized_input=normalized_input,
            signed_sample_graph=signed_sample_graph,
            candidate_lag_mask=candidate_lag_mask,
            prediction_baseline=prediction_baseline,
            innovation_hat=innovation_hat,
            innovation_target=innovation_target,
        )
