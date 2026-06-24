from dataclasses import dataclass
import math
from typing import Optional

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
    a_shared: torch.Tensor
    a_effective: torch.Tensor
    dag_penalty: torch.Tensor
    dag_metadata: dict
    a_delta: Optional[torch.Tensor] = None
    normalized_input: Optional[torch.Tensor] = None
    counterfactual_loss: Optional[torch.Tensor] = None
    counterfactual_effect_mean: Optional[torch.Tensor] = None
    counterfactual_effect_max: Optional[torch.Tensor] = None
    counterfactual_edge_mean: Optional[torch.Tensor] = None

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
        sample_graph_rank=4,
        lambda_pred=1.0,
        lambda_dag=0.001,
        lambda_sparse=0.0005,
        lambda_smooth=0.0001,
        lambda_counterfactual=0.0,
        counterfactual_edges=4,
        counterfactual_temperature=0.1,
        counterfactual_interval=1,
        counterfactual_baseline="zero",
        lambda_sample_l1=0.0,
        lambda_sample_deviation=0.0,
        dagma_warmup_epochs=5,
        dagma_barrier_epochs=20,
        reg_warmup_epochs=0,
        graph_hidden_dim=None,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.lag_order = max(int(lag_order), 1)
        self.input_norm = str(input_norm or "none").lower()
        self.use_sample_graph_residual = bool(use_sample_graph_residual)
        self.sample_graph_delta_scale = float(sample_graph_delta_scale)
        self.sample_graph_rank = max(int(sample_graph_rank), 1)
        self.lambda_pred = float(lambda_pred)
        self.lambda_dag = float(lambda_dag)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_counterfactual = float(lambda_counterfactual)
        self.counterfactual_edges = max(int(counterfactual_edges), 0)
        self.counterfactual_temperature = max(float(counterfactual_temperature), 1e-6)
        self.counterfactual_interval = max(int(counterfactual_interval), 1)
        self.counterfactual_baseline = str(counterfactual_baseline or "zero").lower()
        if self.counterfactual_baseline not in ("zero", "shuffle"):
            raise ValueError(
                f"Unsupported counterfactual_baseline={self.counterfactual_baseline!r}. "
                "Use 'zero' or 'shuffle'."
            )
        self.lambda_sample_l1 = float(lambda_sample_l1)
        self.lambda_sample_deviation = float(lambda_sample_deviation)
        self.dagma_warmup_epochs = max(int(dagma_warmup_epochs), 0)
        self.dagma_barrier_epochs = max(int(dagma_barrier_epochs), 1)
        self.reg_warmup_epochs = max(int(reg_warmup_epochs), 0)
        self.graph_hidden_dim = int(graph_hidden_dim or 8)
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
            [LocalLinear(self.n_nodes, self.graph_hidden_dim, 1) for _ in range(self.lag_order)]
        )
        self.a0_decoder = LocalLinear(self.n_nodes, self.graph_hidden_dim, 1)

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

    def shared_graphs(self):
        lag_pos, lag_neg = self._masked_lag_weights()
        a0_pos, a0_neg = self._masked_a0_weights()
        lag_weight = lag_pos - lag_neg
        a0_weight = a0_pos - a0_neg
        # Shape convention follows CausalGraphLearner: rows are parent/source,
        # columns are child/target after transpose.
        a_lag = torch.sqrt(torch.sum(lag_weight * lag_weight, dim=2).transpose(1, 2) + 1e-12)
        a0 = torch.sqrt(torch.sum(a0_weight * a0_weight, dim=1).T + 1e-12)
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
        decoded = torch.sigmoid(hidden_flat)
        decoded = decoder(decoded).squeeze(-1)
        return decoded.reshape(bsz, steps, feature_dim, nodes).permute(0, 1, 3, 2)

    def _predict(self, histories):
        lag_pos, lag_neg = self._masked_lag_weights()
        a0_pos, a0_neg = self._masked_a0_weights()
        pred = torch.zeros_like(histories[0])
        for lag_idx, history in enumerate(histories):
            weight = lag_pos[lag_idx] - lag_neg[lag_idx]
            hidden = torch.einsum("btpd,jmp->btjmd", history, weight)
            pred = pred + self._decode_hidden(hidden, self.lag_decoder[lag_idx])

        # A0 uses the prediction itself as contemporaneous residual input. This
        # avoids leaking the true target into the prediction objective.
        a0_weight = a0_pos - a0_neg
        residual_hidden = torch.einsum("btpd,jmp->btjmd", pred, a0_weight)
        return pred + self._decode_hidden(residual_hidden, self.a0_decoder)

    def _sample_residual_graph(self, x, a_shared):
        if not self.use_sample_graph_residual or self.sample_graph_delta_scale <= 0:
            return None, a_shared
        node_summary = x.mean(dim=(1, 3), keepdim=False).unsqueeze(-1)
        left = self.sample_left(node_summary)
        right = self.sample_right(node_summary)
        low_rank = torch.einsum("bnr,bmr->bnm", left, right) / math.sqrt(float(self.sample_graph_rank))
        a_delta = torch.tanh(low_rank) * self.sample_graph_delta_scale
        off_diag = self.off_diag_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        a_delta = a_delta * off_diag
        a_effective = torch.clamp(a_shared.unsqueeze(0) + a_delta, min=0.0) * off_diag
        return a_delta, a_effective

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
        return (a_lag[1:] - a_lag[:-1]).abs().mean()

    def _zero_counterfactual_stats(self, device, dtype):
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, zero, zero, zero

    def _intervene_history(self, histories, lag_idx, parent_idx):
        intervened = list(histories)
        history = histories[lag_idx]
        history_cf = history.clone()
        if self.counterfactual_baseline == "shuffle" and history.shape[0] > 1:
            perm = torch.randperm(history.shape[0], device=history.device)
            history_cf[:, :, parent_idx, :] = history[perm, :, parent_idx, :]
        else:
            history_cf[:, :, parent_idx, :] = 0.0
        intervened[lag_idx] = history_cf
        return intervened

    def counterfactual_granger_loss(self, histories, x_hat, a_lag):
        """Align strong lagged edges with their do-style predictive effect.

        For a candidate parent i, lag l, and child j, we intervene on the
        historical parent signal x_i(t-l) by replacing it with a baseline and
        measure how much the prediction of x_j(t) changes. This gives each
        selected edge a lightweight counterfactual Granger score; the learned
        lag graph is then encouraged to rank edges similarly to that effect.
        """

        if (
            not self.training
            or self.lambda_counterfactual <= 0
            or self.counterfactual_edges <= 0
            or self.current_epoch % self.counterfactual_interval != 0
        ):
            return self._zero_counterfactual_stats(x_hat.device, x_hat.dtype)

        with torch.no_grad():
            scores = a_lag.detach().clone()
            for lag_idx in range(scores.shape[0]):
                scores[lag_idx].fill_diagonal_(0.0)
            flat_scores = scores.reshape(-1)
            positive = flat_scores > 0
            if not torch.any(positive):
                return self._zero_counterfactual_stats(x_hat.device, x_hat.dtype)
            k = min(self.counterfactual_edges, int(positive.sum().item()))
            _, flat_indices = torch.topk(flat_scores, k=k)

        effects = []
        edge_values = []
        n = self.n_nodes
        for flat_index in flat_indices.tolist():
            lag_idx = flat_index // (n * n)
            rem = flat_index % (n * n)
            parent_idx = rem // n
            child_idx = rem % n
            if parent_idx == child_idx:
                continue
            cf_histories = self._intervene_history(histories, lag_idx, parent_idx)
            with torch.no_grad():
                x_hat_cf = self._predict(cf_histories)
                effect = (x_hat[:, :, child_idx, :] - x_hat_cf[:, :, child_idx, :]).abs().mean()
            effects.append(effect)
            edge_values.append(a_lag[lag_idx, parent_idx, child_idx])

        if not effects:
            return self._zero_counterfactual_stats(x_hat.device, x_hat.dtype)

        effects = torch.stack(effects)
        edge_values = torch.stack(edge_values)
        target = effects.detach()
        if effects.numel() == 1:
            scale_loss = F.smooth_l1_loss(edge_values, target)
            rank_loss = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        else:
            target_norm = target / target.max().clamp_min(1e-8)
            edge_norm = edge_values / edge_values.detach().max().clamp_min(1e-8)
            scale_loss = F.smooth_l1_loss(edge_norm, target_norm)
            edge_log_prob = F.log_softmax(edge_values / self.counterfactual_temperature, dim=0)
            target_prob = F.softmax(target / self.counterfactual_temperature, dim=0)
            rank_loss = F.kl_div(edge_log_prob, target_prob, reduction="batchmean")
        loss = scale_loss + rank_loss
        return loss, target.mean(), target.max(), edge_values.detach().mean()

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
        pred_loss = F.mse_loss(output.x_hat, output.target)
        dag_loss = normalized_dag_loss(output.dag_penalty, self.n_nodes)
        sparse_loss = self.first_layer_l1_loss()
        smooth_loss = self.smoothness_loss(output.a_lag)
        counterfactual_loss = (
            output.counterfactual_loss
            if output.counterfactual_loss is not None
            else torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
        )
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
                + self.lambda_counterfactual * counterfactual_loss
                + self.lambda_sample_l1 * sample_l1
                + self.lambda_sample_deviation * sample_dev
            )
        )
        parts = {
            "temporal_pred_loss": pred_loss,
            "causal_dag_loss": dag_loss,
            "temporal_sparse_loss": sparse_loss,
            "temporal_smooth_loss": smooth_loss,
            "temporal_counterfactual_loss": counterfactual_loss,
            "sample_graph_l1_loss": sample_l1,
            "sample_graph_deviation_loss": sample_dev,
            "causal_aux_loss": aux_loss,
            "temporal_reg_scale": torch.as_tensor(reg_scale, dtype=aux_loss.dtype, device=aux_loss.device),
            "causal_dag_weighted_loss": self.lambda_dag * dag_scale * dag_loss,
            "temporal_sparse_weighted_loss": self.lambda_sparse * sparse_loss,
            "temporal_smooth_weighted_loss": self.lambda_smooth * smooth_loss,
            "temporal_counterfactual_weighted_loss": self.lambda_counterfactual * counterfactual_loss,
        }
        if output.counterfactual_effect_mean is not None:
            parts["temporal_counterfactual_effect_mean"] = output.counterfactual_effect_mean
        if output.counterfactual_effect_max is not None:
            parts["temporal_counterfactual_effect_max"] = output.counterfactual_effect_max
        if output.counterfactual_edge_mean is not None:
            parts["temporal_counterfactual_edge_mean"] = output.counterfactual_edge_mean
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
        a0, a_lag = self.shared_graphs()
        x_hat = self._predict(histories)
        (
            counterfactual_loss,
            counterfactual_effect_mean,
            counterfactual_effect_max,
            counterfactual_edge_mean,
        ) = self.counterfactual_granger_loss(histories, x_hat, a_lag)
        a_lag_mean = a_lag.mean(dim=0)
        a_shared = a_lag_mean * self.off_diag_mask.to(a_lag_mean.device, a_lag_mean.dtype)
        a_delta, a_effective = self._sample_residual_graph(x4, a_shared)
        penalty = self.dag_constraint(a0)
        stage, dag_scale = self._dagma_schedule()
        metadata = {
            "dag_method": "dagma_logdet",
            "graph_style": "temporal_nts_notears",
            "dagma_stage": stage,
            "dagma_effective_scale": float(dag_scale),
            "lag_order": self.lag_order,
            "use_sample_graph_residual": self.use_sample_graph_residual,
        }
        metadata.update(self.dag_constraint.metadata())
        for key, value in adjacency_directionality(a0).items():
            metadata[f"a0_{key}"] = value.item()
        for key, value in adjacency_directionality(a_lag_mean).items():
            metadata[f"alag_mean_{key}"] = value.item()
        for key, value in adjacency_directionality(a_shared).items():
            metadata[f"shared_{key}"] = value.item()
        if a_delta is not None:
            metadata["sample_delta_abs_mean"] = a_delta.abs().mean().detach().item()
            metadata["sample_delta_abs_max"] = a_delta.abs().max().detach().item()

        if squeezed:
            x_hat = x_hat.squeeze(-1)
            target = target.squeeze(-1)
            normalized_input = x4.squeeze(-1)
        else:
            normalized_input = x4

        return TemporalSEMOutput(
            x_hat=x_hat,
            target=target,
            a0=a0,
            a_lag=a_lag,
            a_shared=a_shared,
            a_effective=a_effective,
            dag_penalty=penalty,
            dag_metadata=metadata,
            a_delta=a_delta,
            normalized_input=normalized_input,
            counterfactual_loss=counterfactual_loss,
            counterfactual_effect_mean=counterfactual_effect_mean,
            counterfactual_effect_max=counterfactual_effect_max,
            counterfactual_edge_mean=counterfactual_edge_mean,
        )
