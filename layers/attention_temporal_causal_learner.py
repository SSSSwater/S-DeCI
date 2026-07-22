import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.causal_graph_learner import (
    DAGMALogDetConstraint,
    adjacency_directionality,
    normalized_dag_loss,
    normalized_l1_loss,
)
from layers.temporal_sem_causal_learner import TemporalSEMOutput


class AttentionGuidedTemporalCausalLearner(nn.Module):
    """Attention-guided Temporal NTS-NOTEARS learner for S-DeCI module 2.

    A_lag is the cross-time graph used by downstream graph modules. A0 only
    models same-time residual dependency and carries the DAG penalty.
    """

    def __init__(
        self,
        n_nodes,
        lag_order=3,
        input_dim=1,
        attention_heads=2,
        attention_head_dim=8,
        attention_dropout=0.0,
        classification_graph_scale=1.0,
        init_logit=-3.0,
        dagma_logdet_s=1.0,
        dagma_logdet_margin=0.1,
        dagma_power_iters=5,
        input_norm="time_zscore",
        lambda_pred=1.0,
        lambda_dag=0.001,
        lambda_sparse=0.0005,
        lambda_smooth=0.0001,
        prediction_loss_mode="bold_alff",
        pred_huber_delta=1.0,
        lambda_pred_delta=0.2,
        lambda_pred_lowfreq=0.2,
        lambda_pred_corr=0.05,
        lowfreq_kernel_size=9,
        a0_sparse_ratio=0.2,
        dagma_warmup_epochs=5,
        dagma_barrier_epochs=20,
        reg_warmup_epochs=0,
        decoder_activation=None,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.lag_order = max(int(lag_order), 1)
        self.input_dim = max(int(input_dim), 1)
        self.attention_heads = max(int(attention_heads), 1)
        self.attention_head_dim = max(int(attention_head_dim), 1)
        self.inner_dim = self.attention_heads * self.attention_head_dim
        self.classification_graph_scale = float(classification_graph_scale)
        self.input_norm = str(input_norm or "none").lower()
        self.lambda_pred = float(lambda_pred)
        self.lambda_dag = float(lambda_dag)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_smooth = float(lambda_smooth)
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
        self.dagma_warmup_epochs = max(int(dagma_warmup_epochs), 0)
        self.dagma_barrier_epochs = max(int(dagma_barrier_epochs), 1)
        self.reg_warmup_epochs = max(int(reg_warmup_epochs), 0)
        self.current_epoch = 0

        self.query_proj = nn.Linear(self.input_dim, self.inner_dim)
        self.key_proj = nn.Linear(self.input_dim, self.inner_dim)
        self.value_proj = nn.Linear(self.input_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.input_dim)
        self.dropout = nn.Dropout(float(attention_dropout))

        init_value = torch.full(
            (self.attention_heads, self.lag_order, self.n_nodes, self.n_nodes),
            float(init_logit),
        )
        init_value = init_value + 0.01 * torch.randn_like(init_value)
        self.gate_logits = nn.Parameter(init_value)

        a0_init = torch.full((self.n_nodes, self.n_nodes), float(init_logit))
        a0_init = a0_init + 0.01 * torch.randn_like(a0_init)
        self.a0_logits = nn.Parameter(a0_init)
        self.a0_residual_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        self.dag_constraint = DAGMALogDetConstraint(
            num_nodes=self.n_nodes,
            s=dagma_logdet_s,
            margin=dagma_logdet_margin,
            power_iters=dagma_power_iters,
        )
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.n_nodes))

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _as_4d(self, x):
        if x.ndim == 3:
            return x.unsqueeze(-1), True
        if x.ndim == 4:
            return x, False
        raise ValueError(f"Expected temporal input [B, T, N] or [B, T, N, D], got {tuple(x.shape)}.")

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

    def _off_diag(self, matrix):
        mask = self.off_diag_mask.to(device=matrix.device, dtype=matrix.dtype)
        if matrix.ndim == 2:
            return matrix * mask
        if matrix.ndim == 3:
            return matrix * mask.unsqueeze(0)
        if matrix.ndim == 4:
            return matrix * mask.view(1, 1, self.n_nodes, self.n_nodes)
        raise ValueError(f"Unexpected graph shape {tuple(matrix.shape)}.")

    def a0_adjacency(self):
        return self._off_diag(torch.sigmoid(self.a0_logits))

    def gate_adjacency(self):
        return self._off_diag(torch.sigmoid(self.gate_logits))

    def _project_heads(self, tensor, projection):
        projected = projection(tensor)
        return projected.view(*tensor.shape[:-1], self.attention_heads, self.attention_head_dim)

    def _attention_predict(self, histories):
        # 将 lag、batch、time 合并为批量矩阵乘法，避免在 Python 循环里反复创建
        # [B,T,H,N,N] 五维 attention 张量导致 GPU 吃不满。
        bsz, steps, nodes, feat_dim = histories[0].shape
        hist_stack = torch.stack(histories, dim=0).reshape(
            self.lag_order,
            bsz * steps,
            nodes,
            feat_dim,
        )
        history_summary = hist_stack.mean(dim=0)

        q = self._project_heads(history_summary, self.query_proj).permute(0, 2, 1, 3)
        k = self._project_heads(hist_stack, self.key_proj).permute(0, 1, 3, 2, 4)
        v = self._project_heads(hist_stack, self.value_proj).permute(0, 1, 3, 2, 4)
        q = q.unsqueeze(0).expand(self.lag_order, -1, -1, -1, -1)

        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.attention_head_dim))
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # gate 参数语义为 [head, lag, parent, child]，预测时转为
        # [lag, 1, head, child, parent] 后直接广播到 attention。
        gate = self.gate_adjacency()
        gate_child_parent = gate.permute(1, 0, 3, 2).unsqueeze(1)
        edge = attn * gate_child_parent

        context = torch.matmul(edge, v).mean(dim=0)
        context = context.transpose(1, 2).reshape(bsz, steps, nodes, self.inner_dim)
        x_hat = self.out_proj(context)
        a_lag = self._off_diag(edge.mean(dim=(1, 2)).transpose(-1, -2))
        attention_entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return x_hat, a_lag, attention_entropy
    def _dagma_schedule(self):
        if self.current_epoch < self.dagma_warmup_epochs:
            denom = max(float(self.dagma_warmup_epochs), 1.0)
            scale = 0.1 + 0.9 * ((self.current_epoch + 1) / denom)
            return "warmup", min(scale, 1.0)
        barrier_start = self.dagma_warmup_epochs
        barrier_end = barrier_start + self.dagma_barrier_epochs
        if self.current_epoch < barrier_end:
            progress = (self.current_epoch - barrier_start + 1) / max(
                float(self.dagma_barrier_epochs), 1.0
            )
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
        lag_weights = torch.arange(
            2,
            a_lag.shape[0] + 1,
            device=a_lag.device,
            dtype=a_lag.dtype,
        ).view(-1, 1, 1).sqrt()
        return ((a_lag[1:] - a_lag[:-1]).abs() * lag_weights).mean()

    def first_layer_l1_loss(self, a_lag=None, a0=None):
        gate_l1 = normalized_l1_loss(self.gate_adjacency().mean(dim=0))
        parts = [gate_l1]
        if a_lag is not None:
            parts.append(normalized_l1_loss(a_lag))
        if a0 is not None:
            parts.append(normalized_l1_loss(a0))
        return torch.stack(parts).mean()

    def direct_graph_sparse_loss(self, a_lag, a0):
        off_diag = self.off_diag_mask.to(device=a_lag.device, dtype=a_lag.dtype)
        lag_weights = torch.arange(
            1,
            a_lag.shape[0] + 1,
            device=a_lag.device,
            dtype=a_lag.dtype,
        ).view(-1, 1, 1).sqrt()
        lag_sparse = (a_lag * off_diag.unsqueeze(0) * lag_weights).sum()
        lag_sparse = lag_sparse / (off_diag.sum().clamp_min(1.0) * lag_weights.sum().clamp_min(1.0))
        a0_sparse = (a0 * off_diag).sum() / off_diag.sum().clamp_min(1.0)
        return lag_sparse + self.a0_sparse_ratio * a0_sparse

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

    def prediction_loss(self, x_hat, target):
        x_hat_4d = self._loss_as_4d(x_hat)
        target_4d = self._loss_as_4d(target)
        if self.prediction_loss_mode == "mse":
            base = F.mse_loss(x_hat_4d, target_4d)
        else:
            base = F.huber_loss(x_hat_4d, target_4d, delta=self.pred_huber_delta)
        if self.prediction_loss_mode != "bold_alff":
            zero = torch.zeros((), device=x_hat_4d.device, dtype=x_hat_4d.dtype)
            return base, {
                "temporal_pred_base_loss": base,
                "temporal_pred_delta_loss": zero,
                "temporal_pred_lowfreq_loss": zero,
                "temporal_pred_corr_loss": zero,
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
        }

    def compute_losses(self, output):
        pred_loss, pred_parts = self.prediction_loss(output.x_hat, output.target)
        dag_loss = normalized_dag_loss(output.dag_penalty, self.n_nodes)
        parameter_sparse_loss = self.first_layer_l1_loss(output.a_lag, output.a0)
        sparse_loss = self.direct_graph_sparse_loss(output.a_lag, output.a0)
        smooth_loss = self.smoothness_loss(output.a_lag)
        stage = output.dag_metadata["dagma_stage"]
        dag_scale = output.dag_metadata["dagma_effective_scale"]
        reg_scale = self._reg_schedule()
        aux_loss = self.lambda_pred * pred_loss + reg_scale * (
            self.lambda_dag * dag_scale * dag_loss
            + self.lambda_sparse * sparse_loss
            + self.lambda_smooth * smooth_loss
        )
        parts = {
            "temporal_pred_loss": pred_loss,
            **pred_parts,
            "causal_dag_loss": dag_loss,
            "temporal_sparse_loss": sparse_loss,
            "temporal_parameter_sparse_loss": parameter_sparse_loss,
            "temporal_smooth_loss": smooth_loss,
            "causal_aux_loss": aux_loss,
            "temporal_reg_scale": torch.as_tensor(reg_scale, dtype=aux_loss.dtype, device=aux_loss.device),
            "causal_dag_weighted_loss": self.lambda_dag * dag_scale * dag_loss,
            "temporal_sparse_weighted_loss": self.lambda_sparse * sparse_loss,
            "temporal_parameter_sparse_weighted_loss": self.lambda_sparse * parameter_sparse_loss,
            "temporal_smooth_weighted_loss": self.lambda_smooth * smooth_loss,
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
        if x4.shape[-1] != self.input_dim:
            if self.input_dim != 1:
                raise ValueError(
                    f"Expected input_dim={self.input_dim}, got feature dim {x4.shape[-1]}."
                )
            x4 = x4.mean(dim=-1, keepdim=True)
        target, histories = self._build_windows(x4)
        x_hat, a_lag, attention_entropy = self._attention_predict(histories)

        # A0 表示同时间片残余依赖，只用预测值自身做残差修正，不读取真实 target。
        a0 = self.a0_adjacency()
        residual = torch.einsum("btpd,pc->btcd", x_hat, a0)
        x_hat = x_hat + torch.tanh(self.a0_residual_scale) * residual / max(float(self.n_nodes), 1.0)

        a_lag_mean = self._off_diag(a_lag.mean(dim=0))
        # raw A_lag 来自 parent 维度 softmax，单条边天然约为 1/N；
        # 下游分类图允许单独设置尺度，原始 A_lag 保留用于解释和稀疏约束。
        graph_scale = max(self.classification_graph_scale, 0.0)
        a_shared = self._off_diag(a_lag_mean * graph_scale)
        a_effective = a_shared
        penalty = self.dag_constraint(a0)
        stage, dag_scale = self._dagma_schedule()
        gate = self.gate_adjacency()
        metadata = {
            "dag_method": "dagma_logdet",
            "graph_style": "attention_guided_temporal_nts_notears",
            "dagma_stage": stage,
            "dagma_effective_scale": float(dag_scale),
            "lag_order": self.lag_order,
            "attention_heads": self.attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "attention_entropy": attention_entropy.detach().item(),
            "attention_graph_scale": graph_scale,
            "gate_mass": gate.detach().mean().item(),
            "gate_abs_max": gate.detach().max().item(),
            "a0_role_id": 0,
            "use_sample_graph_residual": False,
        }
        metadata.update(self.dag_constraint.metadata())
        for key, value in adjacency_directionality(a0).items():
            metadata[f"a0_{key}"] = value.item()
        for key, value in adjacency_directionality(a_lag_mean).items():
            metadata[f"alag_mean_{key}"] = value.item()
        for key, value in adjacency_directionality(a_shared).items():
            metadata[f"shared_{key}"] = value.item()

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
            a_delta=None,
            normalized_input=normalized_input,
        )

