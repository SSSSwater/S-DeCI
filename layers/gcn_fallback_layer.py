from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GCNFallbackOutput:
    normalized_adjacency: torch.Tensor
    h_gcn: torch.Tensor
    readout: torch.Tensor
    logits: torch.Tensor


class EuclideanGraphConvolution(nn.Module):
    """普通 GCN 层：在欧式空间中用 adjacency 聚合节点特征。"""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x, adjacency):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, N, D], got {tuple(x.shape)}.")
        if adjacency.ndim == 2:
            aggregated = torch.einsum("ij,bjd->bid", adjacency.T, x)
        elif adjacency.ndim == 3:
            aggregated = torch.einsum("bij,bjd->bid", adjacency.transpose(1, 2), x)
        else:
            raise ValueError(
                f"Expected adjacency with shape [N, N] or [B, N, N], got {tuple(adjacency.shape)}."
            )
        return self.dropout(F.gelu(self.linear(aggregated)))


class ModuleGCNFallback(nn.Module):
    """S-DeCI 模块 3/4 关闭时的普通 GCN 退化分类路径。"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        dropout=0.0,
        add_self_loop=True,
        adjacency_normalization="row",
        sample_correlation_mode="abs",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = max(int(num_layers), 1)
        self.add_self_loop = bool(add_self_loop)
        self.adjacency_normalization = adjacency_normalization
        self.sample_correlation_mode = sample_correlation_mode

        layers = []
        for idx in range(self.num_layers):
            in_dim = self.input_dim if idx == 0 else self.hidden_dim
            layers.append(EuclideanGraphConvolution(in_dim, self.hidden_dim, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)

    def _prepare_adjacency_values(self, adjacency, is_sample_correlation=False):
        adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        if is_sample_correlation:
            if self.sample_correlation_mode == "abs":
                return adjacency.abs()
            if self.sample_correlation_mode == "positive":
                return adjacency.clamp_min(0.0)
            if self.sample_correlation_mode == "raw":
                return adjacency
            raise ValueError(
                f"Unsupported sample_correlation_mode={self.sample_correlation_mode!r}. "
                "Use 'abs', 'positive' or 'raw'."
            )
        return adjacency.clamp_min(0.0)

    def _normalize_adjacency(self, adjacency, is_sample_correlation=False):
        adjacency = self._prepare_adjacency_values(
            adjacency,
            is_sample_correlation=is_sample_correlation,
        )
        if adjacency.ndim == 2:
            n_nodes = adjacency.shape[0]
        elif adjacency.ndim == 3:
            n_nodes = adjacency.shape[1]
        else:
            raise ValueError(
                f"Expected adjacency with shape [N, N] or [B, N, N], got {tuple(adjacency.shape)}."
            )

        if self.add_self_loop:
            identity = torch.eye(n_nodes, device=adjacency.device, dtype=adjacency.dtype)
            adjacency = adjacency + (identity if adjacency.ndim == 2 else identity.unsqueeze(0))

        if self.adjacency_normalization == "none":
            return adjacency
        if self.adjacency_normalization == "sym":
            degree = adjacency.sum(dim=-1).clamp_min(1e-8)
            inv_sqrt = degree.rsqrt()
            if adjacency.ndim == 2:
                return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
            return inv_sqrt[:, :, None] * adjacency * inv_sqrt[:, None, :]
        if self.adjacency_normalization != "row":
            raise ValueError(
                f"Unsupported adjacency_normalization={self.adjacency_normalization!r}. "
                "Use 'row', 'sym' or 'none'."
            )

        degree = adjacency.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        return adjacency / degree

    def forward(self, node_features, adjacency, is_sample_correlation=False):
        if node_features.ndim != 3:
            raise ValueError(
                f"Expected node_features with shape [B, N, D], got {tuple(node_features.shape)}."
            )
        expected_2d = node_features.shape[1:2] * 2
        expected_3d = (node_features.shape[0], node_features.shape[1], node_features.shape[1])
        if adjacency.ndim == 2 and adjacency.shape != expected_2d:
            raise ValueError(
                f"Expected adjacency shape [{node_features.shape[1]}, {node_features.shape[1]}], "
                f"got {tuple(adjacency.shape)}."
            )
        if adjacency.ndim == 3 and adjacency.shape != expected_3d:
            raise ValueError(f"Expected batch adjacency shape {expected_3d}, got {tuple(adjacency.shape)}.")

        normalized_adjacency = self._normalize_adjacency(
            adjacency.to(device=node_features.device, dtype=node_features.dtype),
            is_sample_correlation=is_sample_correlation,
        )
        h = node_features
        for layer in self.layers:
            h = layer(h, normalized_adjacency)
        readout = h.mean(dim=1)
        logits = self.classifier(readout)
        return GCNFallbackOutput(
            normalized_adjacency=normalized_adjacency,
            h_gcn=h,
            readout=readout,
            logits=logits,
        )
