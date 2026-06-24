from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalGraphOutput:
    c_hat: torch.Tensor
    adjacency: torch.Tensor
    dag_penalty: torch.Tensor
    dag_metadata: dict
    a_shared: torch.Tensor
    a_effective: torch.Tensor
    a_delta: Optional[torch.Tensor] = None
    normalized_input: Optional[torch.Tensor] = None


class LocalLinear(nn.Module):
    """NTS-NOTEARS 风格的 node-wise local linear layer。"""

    def __init__(self, n_nodes, in_features, out_features, bias=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(n_nodes, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_nodes, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        bound = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = torch.einsum("bfni,nio->bfno", x, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, self.n_nodes, self.out_features)
        return out


class AnalyticDAGConstraint(nn.Module):
    """使用缩放矩阵逆的稳定 DAG constraint。"""

    def __init__(self, num_nodes, margin=0.1, power_iters=5, eps=1e-5):
        super().__init__()
        self.num_nodes = num_nodes
        self.margin = margin
        self.power_iters = power_iters
        self.eps = eps
        self.register_buffer("identity", torch.eye(num_nodes))
        self.last_spectral_radius = None
        self.last_scale = None

    def estimate_spectral_radius(self, matrix):
        x = torch.ones((self.num_nodes, 1), device=matrix.device, dtype=matrix.dtype)
        for _ in range(self.power_iters):
            x = torch.matmul(matrix, x)
            x = x / (torch.norm(x, p=2) + self.eps)

        numerator = torch.matmul(torch.matmul(x.T, matrix), x)
        denominator = torch.matmul(x.T, x).clamp_min(self.eps)
        return (numerator / denominator).squeeze()

    def forward(self, adjacency):
        identity = self.identity.to(device=adjacency.device, dtype=adjacency.dtype)
        off_diag = 1.0 - identity
        nonnegative_matrix = (adjacency * off_diag) ** 2

        spectral_radius = self.estimate_spectral_radius(nonnegative_matrix)
        scale = ((1.0 + self.margin) * spectral_radius).clamp_min(self.eps)
        scaled_matrix = nonnegative_matrix / scale

        inverse = torch.linalg.solve(identity - scaled_matrix, identity)
        dag_loss = torch.trace(inverse) - self.num_nodes

        self.last_spectral_radius = spectral_radius.detach()
        self.last_scale = scale.detach()
        return dag_loss

    def metadata(self):
        metadata = {}
        if self.last_spectral_radius is not None:
            metadata["analytic_spectral_radius"] = self.last_spectral_radius.item()
        if self.last_scale is not None:
            metadata["analytic_scale"] = self.last_scale.item()
        metadata["analytic_margin"] = self.margin
        metadata["analytic_power_iters"] = self.power_iters
        return metadata


class DAGMALogDetConstraint(nn.Module):
    """DAGMA 风格的 log-det DAG penalty。

    h(A) = -logdet(sI - A*A) + d log(s)。为了避免训练早期数值爆炸，
    s 会按谱半径做下界保护；这保持 log-det 约束可微，同时比 matrix_exp 更稳定。
    """

    def __init__(self, num_nodes, s=1.0, margin=0.1, power_iters=5, eps=1e-5):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.s = float(s)
        self.margin = float(margin)
        self.power_iters = int(power_iters)
        self.eps = float(eps)
        self.register_buffer("identity", torch.eye(num_nodes))
        self.last_spectral_radius = None
        self.last_scale = None
        self.last_logdet_sign = None

    def estimate_spectral_radius(self, matrix):
        x = torch.ones((self.num_nodes, 1), device=matrix.device, dtype=matrix.dtype)
        for _ in range(self.power_iters):
            x = torch.matmul(matrix, x)
            x = x / (torch.norm(x, p=2) + self.eps)
        numerator = torch.matmul(torch.matmul(x.T, matrix), x)
        denominator = torch.matmul(x.T, x).clamp_min(self.eps)
        return (numerator / denominator).squeeze()

    def forward(self, adjacency):
        identity = self.identity.to(device=adjacency.device, dtype=adjacency.dtype)
        off_diag = 1.0 - identity
        nonnegative_matrix = (adjacency * off_diag) ** 2
        spectral_radius = self.estimate_spectral_radius(nonnegative_matrix)
        base_scale = torch.as_tensor(self.s, device=adjacency.device, dtype=adjacency.dtype)
        scale = torch.maximum(base_scale, spectral_radius.detach() + self.margin).clamp_min(self.eps)
        constrained = scale * identity - nonnegative_matrix
        sign, logabsdet = torch.linalg.slogdet(constrained)
        penalty = -logabsdet + self.num_nodes * torch.log(scale)
        if sign <= 0:
            penalty = penalty + (1.0 - sign).abs()

        self.last_spectral_radius = spectral_radius.detach()
        self.last_scale = scale.detach()
        self.last_logdet_sign = sign.detach()
        return penalty

    def metadata(self):
        metadata = {
            "dagma_logdet_s": self.s,
            "dagma_logdet_margin": self.margin,
            "dagma_power_iters": self.power_iters,
        }
        if self.last_spectral_radius is not None:
            metadata["dagma_spectral_radius"] = self.last_spectral_radius.item()
        if self.last_scale is not None:
            metadata["dagma_scale"] = self.last_scale.item()
        if self.last_logdet_sign is not None:
            metadata["dagma_logdet_sign"] = self.last_logdet_sign.item()
        return metadata


def dag_penalty(adjacency):
    """NOTEARS-style differentiable acyclicity penalty."""
    n_nodes = adjacency.shape[0]
    expm = torch.matrix_exp(adjacency * adjacency)
    return torch.trace(expm) - n_nodes


def threshold_adjacency(adjacency, threshold=0.5):
    return (adjacency >= threshold).to(adjacency.dtype)


def normalized_dag_loss(dag_value, n_nodes):
    return dag_value / max(float(n_nodes), 1.0)


def normalized_l1_loss(adjacency):
    if adjacency.ndim == 2:
        off_diag_count = adjacency.numel() - adjacency.shape[0]
    elif adjacency.ndim == 3:
        off_diag_count = adjacency.numel() - adjacency.shape[0] * adjacency.shape[1]
    else:
        off_diag_count = adjacency.numel()
    return adjacency.abs().sum() / max(float(off_diag_count), 1.0)


def adjacency_directionality(adjacency):
    if adjacency.ndim == 2:
        n_nodes = adjacency.shape[0]
        identity = torch.eye(n_nodes, device=adjacency.device, dtype=adjacency.dtype)
        off_diag = 1.0 - identity
        directed_delta = (adjacency - adjacency.T).abs() * off_diag
        directed_mass = adjacency.abs() * off_diag
        pair_mass = (adjacency.abs() + adjacency.T.abs()) * off_diag
        denominator = pair_mass.sum().clamp_min(1e-12)
        off_diag_sum = off_diag.sum().clamp_min(1.0)
    elif adjacency.ndim == 3:
        n_nodes = adjacency.shape[-1]
        identity = torch.eye(n_nodes, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        off_diag = 1.0 - identity
        directed_delta = (adjacency - adjacency.transpose(-1, -2)).abs() * off_diag
        directed_mass = adjacency.abs() * off_diag
        pair_mass = (adjacency.abs() + adjacency.transpose(-1, -2).abs()) * off_diag
        denominator = pair_mass.sum().clamp_min(1e-12)
        off_diag_sum = (off_diag.sum() * adjacency.shape[0]).clamp_min(1.0)
    else:
        raise ValueError(f"Expected adjacency with shape [N, N] or [B, N, N], got {tuple(adjacency.shape)}.")
    return {
        "adjacency_asymmetry_mean": (directed_delta.sum() / off_diag_sum).detach(),
        "adjacency_directionality_ratio": (directed_delta.sum() / denominator).detach(),
        "adjacency_mass_mean": (directed_mass.sum() / off_diag_sum).detach(),
    }


def sinkhorn(log_alpha, n_iters=20):
    n = log_alpha.shape[-1]
    log_alpha = log_alpha.reshape(-1, n, n)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha).squeeze(0)


def straight_through_hard_permutation(soft_permutation, scores):
    """用 Hungarian matching 生成 hard permutation，并保留 soft permutation 的梯度。"""
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        row_ind, col_ind = linear_sum_assignment((-scores.detach().cpu()).numpy())
        hard = torch.zeros_like(soft_permutation)
        hard[row_ind, col_ind] = 1.0
    return hard - soft_permutation.detach() + soft_permutation


def sample_gumbel_like(tensor, eps=1e-20):
    uniform = torch.rand_like(tensor)
    return -torch.log(-torch.log(uniform + eps) + eps)


def _inverse_softplus(value):
    value = torch.as_tensor(value)
    return torch.log(torch.expm1(value).clamp_min(1e-12))


class CausalGraphLearner(nn.Module):
    """S-DeCI 模块 2：共享因果图 + 可选样本残差图学习器。"""

    def __init__(
        self,
        n_nodes,
        feature_dim=64,
        init_logit=-4.0,
        dag_method="notears",
        analytic_margin=0.1,
        analytic_power_iters=5,
        graph_hidden_dim=None,
        graph_method="nts_notears",
        dag_sampling_temperature=1.0,
        dag_sampling_noise=0.0,
        dag_sampling_sinkhorn_iters=20,
        dag_sampling_hard=True,
        causal_input_norm="none",
        use_sample_graph_residual=False,
        sample_graph_delta_scale=0.05,
        sample_graph_hidden_dim=None,
        dagma_logdet_s=1.0,
        dagma_logdet_margin=0.1,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.feature_dim = int(feature_dim)
        self.dag_method = dag_method
        self.graph_method = graph_method
        self.causal_input_norm = str(causal_input_norm or "none").lower()
        self.use_sample_graph_residual = bool(use_sample_graph_residual)
        self.sample_graph_delta_scale = float(sample_graph_delta_scale)
        self.graph_hidden_dim = int(graph_hidden_dim or min(max(feature_dim // 4, 4), 16))
        if self.graph_hidden_dim <= 0:
            self.graph_hidden_dim = min(max(feature_dim // 4, 4), 16)
        self.dag_sampling_temperature = dag_sampling_temperature
        self.dag_sampling_noise = dag_sampling_noise
        self.dag_sampling_sinkhorn_iters = dag_sampling_sinkhorn_iters
        self.dag_sampling_hard = bool(dag_sampling_hard)

        self.register_buffer("off_diag_mask", 1.0 - torch.eye(n_nodes))
        self.register_buffer("child_parent_mask", (1.0 - torch.eye(n_nodes)).T)
        self.register_buffer("upper_mask", torch.triu(torch.ones(n_nodes, n_nodes), 1))

        if graph_method in ("nts_notears", "dagma_logdet"):
            init_edge_strength = torch.sigmoid(torch.tensor(float(init_logit))).item()
            init_weight_scale = math.sqrt(max(init_edge_strength, 1e-4) / self.graph_hidden_dim)
            pos_init = torch.rand(n_nodes, self.graph_hidden_dim, n_nodes) * init_weight_scale
            neg_init = torch.rand(n_nodes, self.graph_hidden_dim, n_nodes) * init_weight_scale
            self.fc1_pos_raw = nn.Parameter(_inverse_softplus(pos_init))
            self.fc1_neg_raw = nn.Parameter(_inverse_softplus(neg_init))
            self.decoder = nn.ModuleList([LocalLinear(n_nodes, self.graph_hidden_dim, 1)])
        elif graph_method == "dag_sampling":
            self.perm_weights = nn.Parameter(0.01 * torch.randn(n_nodes, n_nodes))
            edge_logits = torch.full((n_nodes, n_nodes), float(init_logit))
            edge_logits = edge_logits + 0.01 * torch.randn(n_nodes, n_nodes)
            edge_logits.fill_diagonal_(-300.0)
            self.edge_log_params = nn.Parameter(edge_logits)
            self.decoder = nn.ModuleList(
                [
                    LocalLinear(n_nodes, n_nodes, self.graph_hidden_dim),
                    LocalLinear(n_nodes, self.graph_hidden_dim, 1),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported graph_method={graph_method!r}. "
                "Use 'nts_notears', 'dagma_logdet' or 'dag_sampling'."
            )

        sample_hidden = int(sample_graph_hidden_dim or min(max(feature_dim // 2, 8), 64))
        self.sample_graph_query = nn.Linear(feature_dim, sample_hidden)
        self.sample_graph_key = nn.Linear(feature_dim, sample_hidden)

        if graph_method == "dagma_logdet" or dag_method == "dagma_logdet":
            self.dag_constraint = DAGMALogDetConstraint(
                num_nodes=n_nodes,
                s=dagma_logdet_s,
                margin=dagma_logdet_margin,
                power_iters=analytic_power_iters,
            )
            self.dag_method = "dagma_logdet"
        elif dag_method == "notears":
            self.dag_constraint = None
        elif dag_method == "analytic":
            self.dag_constraint = AnalyticDAGConstraint(
                num_nodes=n_nodes,
                margin=analytic_margin,
                power_iters=analytic_power_iters,
            )
        else:
            raise ValueError(
                f"Unsupported dag_method={dag_method!r}. "
                "Use 'notears', 'analytic' or 'dagma_logdet'."
            )

    def _normalize_input(self, c):
        if self.causal_input_norm in ("none", "off", "false", "0"):
            return c
        if self.causal_input_norm in ("feature_zscore", "zscore", "node_zscore"):
            mean = c.mean(dim=-1, keepdim=True)
            std = c.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            return (c - mean) / std
        if self.causal_input_norm in ("batch_node_zscore", "batch_zscore"):
            mean = c.mean(dim=(0, 2), keepdim=True)
            std = c.std(dim=(0, 2), keepdim=True, unbiased=False).clamp_min(1e-5)
            return (c - mean) / std
        raise ValueError(
            f"Unsupported causal_input_norm={self.causal_input_norm!r}. "
            "Use 'none', 'feature_zscore' or 'batch_node_zscore'."
        )

    def _positive_negative_weights(self):
        if self.graph_method == "dag_sampling":
            raise RuntimeError("_positive_negative_weights is not available for dag_sampling.")
        pos = F.softplus(self.fc1_pos_raw)
        neg = F.softplus(self.fc1_neg_raw)
        mask = self.child_parent_mask.to(device=pos.device, dtype=pos.dtype).unsqueeze(1)
        return pos * mask, neg * mask

    def first_layer_weight(self):
        pos, neg = self._positive_negative_weights()
        return pos - neg

    def _dag_sampling_permutation(self):
        log_alpha = F.logsigmoid(self.perm_weights)
        if self.training and self.dag_sampling_noise > 0:
            log_alpha = log_alpha + sample_gumbel_like(log_alpha) * self.dag_sampling_noise
        temperature = max(float(self.dag_sampling_temperature), 1e-6)
        permutation = sinkhorn(log_alpha / temperature, n_iters=self.dag_sampling_sinkhorn_iters)
        if self.dag_sampling_hard:
            return straight_through_hard_permutation(permutation, log_alpha)
        return permutation

    def _dag_sampling_adjacency(self):
        permutation = self._dag_sampling_permutation()
        order_mask = permutation.T @ self.upper_mask.to(permutation.device, permutation.dtype) @ permutation
        edge_prob = torch.sigmoid(self.edge_log_params)
        edge_prob = edge_prob * self.off_diag_mask.to(edge_prob.device, edge_prob.dtype)
        return edge_prob * order_mask

    def effective_adjacency(self):
        if self.graph_method == "dag_sampling":
            return self._dag_sampling_adjacency()
        weight = self.first_layer_weight()
        adjacency_squared = torch.sum(weight * weight, dim=1).T
        adjacency = torch.sqrt(adjacency_squared + 1e-12)
        return adjacency * self.off_diag_mask.to(device=adjacency.device, dtype=adjacency.dtype)

    def _sample_residual_graph(self, c, a_shared):
        if not self.use_sample_graph_residual or self.sample_graph_delta_scale <= 0:
            return None, a_shared
        query = F.gelu(self.sample_graph_query(c))
        key = F.gelu(self.sample_graph_key(c))
        pair_score = torch.einsum("bih,bjh->bij", query, key) / math.sqrt(max(query.shape[-1], 1))
        a_delta = torch.tanh(pair_score) * self.sample_graph_delta_scale
        off_diag = self.off_diag_mask.to(device=c.device, dtype=c.dtype).unsqueeze(0)
        a_delta = a_delta * off_diag
        a_effective = torch.clamp(a_shared.unsqueeze(0) + a_delta, min=0.0) * off_diag
        return a_delta, a_effective

    def first_layer_l1_loss(self):
        if self.graph_method == "dag_sampling":
            return normalized_l1_loss(self.effective_adjacency())
        pos, neg = self._positive_negative_weights()
        off_diag_count = self.n_nodes * max(self.n_nodes - 1, 1) * self.graph_hidden_dim
        return (pos + neg).sum() / float(off_diag_count)

    def sample_graph_l1_loss(self, a_delta):
        if a_delta is None:
            return torch.zeros((), device=self.off_diag_mask.device)
        return normalized_l1_loss(a_delta)

    def sample_graph_deviation_loss(self, a_effective, a_shared):
        if a_effective.ndim != 3:
            return torch.zeros((), device=a_shared.device, dtype=a_shared.dtype)
        shared = a_shared.unsqueeze(0).to(device=a_effective.device, dtype=a_effective.dtype)
        off_diag = self.off_diag_mask.to(device=a_effective.device, dtype=a_effective.dtype).unsqueeze(0)
        return (((a_effective - shared) * off_diag) ** 2).sum() / (
            a_effective.shape[0] * max(float(self.n_nodes * (self.n_nodes - 1)), 1.0)
        )

    def reconstruct(self, c):
        if c.ndim != 3:
            raise ValueError(f"Expected C with shape [B, N, F], got {tuple(c.shape)}.")
        if c.shape[1] != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {c.shape[1]}.")
        if c.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {c.shape[2]}.")

        x = c.transpose(1, 2)
        if self.graph_method == "dag_sampling":
            adjacency = self.effective_adjacency()
            hidden = x.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
            hidden = hidden * adjacency.T.to(device=x.device, dtype=x.dtype).view(1, 1, self.n_nodes, self.n_nodes)
        else:
            weight = self.first_layer_weight()
            hidden = torch.einsum("bfp,jmp->bfjm", x, weight)
        for decoder_layer in self.decoder:
            hidden = torch.sigmoid(hidden)
            hidden = decoder_layer(hidden)
        return hidden.squeeze(-1).transpose(1, 2)

    def _dag_loss_and_metadata(self, adjacency):
        if self.dag_method == "notears":
            penalty = dag_penalty(adjacency)
            metadata = {"dag_method": "notears", "graph_style": self.graph_method}
        elif self.dag_method == "dagma_logdet":
            penalty = self.dag_constraint(adjacency)
            metadata = {"dag_method": "dagma_logdet", "graph_style": self.graph_method}
            metadata.update(self.dag_constraint.metadata())
        else:
            penalty = self.dag_constraint(adjacency)
            metadata = {"dag_method": "analytic", "graph_style": self.graph_method}
            metadata.update(self.dag_constraint.metadata())
        return penalty, metadata

    def forward(self, c):
        normalized_input = self._normalize_input(c)
        a_shared = self.effective_adjacency()
        a_delta, a_effective = self._sample_residual_graph(normalized_input, a_shared)
        c_hat = self.reconstruct(normalized_input)
        penalty, metadata = self._dag_loss_and_metadata(a_shared)
        if self.graph_method == "dag_sampling":
            metadata["dag_sampling_temperature"] = self.dag_sampling_temperature
            metadata["dag_sampling_noise"] = self.dag_sampling_noise
            metadata["dag_sampling_sinkhorn_iters"] = self.dag_sampling_sinkhorn_iters
            metadata["dag_sampling_hard"] = self.dag_sampling_hard
        metadata["causal_input_norm"] = self.causal_input_norm
        metadata["use_sample_graph_residual"] = self.use_sample_graph_residual
        metadata["sample_graph_delta_scale"] = self.sample_graph_delta_scale
        for key, value in adjacency_directionality(a_shared).items():
            metadata[f"shared_{key}"] = value.item()
        effective_directionality = adjacency_directionality(a_effective)
        for key, value in effective_directionality.items():
            metadata[f"effective_{key}"] = value.item()
        if a_delta is not None:
            metadata["sample_delta_abs_mean"] = a_delta.abs().mean().detach().item()
            metadata["sample_delta_abs_max"] = a_delta.abs().max().detach().item()
            metadata["effective_graph_abs_mean"] = a_effective.abs().mean().detach().item()
        return CausalGraphOutput(
            c_hat=c_hat,
            adjacency=a_effective,
            dag_penalty=penalty,
            dag_metadata=metadata,
            a_shared=a_shared,
            a_effective=a_effective,
            a_delta=a_delta,
            normalized_input=normalized_input,
        )
