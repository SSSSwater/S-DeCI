from dataclasses import dataclass

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Module3HGCNOutput:
    c_clipped: torch.Tensor
    h0: torch.Tensor
    h_gcn: torch.Tensor
    z_global: torch.Tensor
    z_tangent: torch.Tensor
    node_attention: torch.Tensor
    network_summary: torch.Tensor
    network_attention: torch.Tensor
    normalized_adjacency: torch.Tensor


def _aal116_network_masks():
    """返回基于 MDD 文献常见判别网络的 AAL116 分组，索引为 0-based。"""

    groups = [
        # Default mode network: mPFC / PCC / precuneus / angular / medial temporal.
        ("dmn", [22, 23, 24, 25, 34, 35, 36, 37, 38, 39, 64, 65, 66, 67, 84, 85, 86, 87]),
        # Fronto-limbic / affective network: OFC, ACC/MCC, insula, hippocampus, amygdala.
        ("fronto_limbic", [4, 5, 8, 9, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41]),
        # Cognitive control / frontoparietal network.
        ("control", [2, 3, 6, 7, 10, 11, 12, 13, 18, 19, 58, 59, 60, 61, 62, 63]),
        # Salience network: anterior insula, ACC/MCC, SMA, thalamus.
        ("salience", [18, 19, 28, 29, 30, 31, 32, 33, 76, 77]),
        # Subcortical-thalamic/striatal loop.
        ("subcortical", [70, 71, 72, 73, 74, 75, 76, 77]),
        # Sensorimotor network.
        ("sensorimotor", [0, 1, 16, 17, 18, 19, 56, 57, 68, 69]),
        # Visual network.
        ("visual", [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]),
        # Cerebellar regions are often reported in whole-brain MDD FC classifiers.
        ("cerebellum", list(range(90, 116))),
    ]
    prior_weights = torch.tensor([1.35, 1.35, 1.15, 1.2, 1.2, 0.9, 0.9, 0.85], dtype=torch.float32)
    return groups, prior_weights


class Backclip(nn.Module):
    """将 Cycle feature 控制在进入 Poincare Ball 前的稳定半径内。"""

    def __init__(self, radius=1.0, eps=1e-8):
        super().__init__()
        self.radius = float(radius)
        self.eps = float(eps)

    def forward(self, x):
        if self.radius <= 0:
            return x
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(self.radius / norm, max=1.0)
        return x * scale


class HyperbolicGraphConvolution(nn.Module):
    """基于模块 2 因果图的 HGCN 层。

    节点特征先用 Mobius matvec 完成升维/变换，再回到切空间按有向邻接矩阵聚合。
    tangent-space 聚合是对逐边 Mobius add 的稳定近似，保留可微路径并适合 116 ROI 训练。
    """

    def __init__(self, in_dim, out_dim, manifold, dropout=0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.manifold = manifold
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, N, D], got {tuple(x.shape)}.")
        if adjacency.ndim not in (2, 3):
            raise ValueError(
                f"Expected adjacency with shape [N, N] or [B, N, N], got {tuple(adjacency.shape)}."
            )

        transformed = self.manifold.mobius_matvec(self.weight, x, dim=-1, project=True)
        transformed_tan = self.manifold.logmap0(transformed, dim=-1)
        transformed_tan = transformed_tan + self.bias.view(1, 1, -1)
        transformed_tan = self.dropout(F.gelu(transformed_tan))

        # adjacency[parent, child]，因此 child 聚合 parent 时使用 A.T。
        if adjacency.ndim == 2:
            agg_tan = torch.einsum("ij,bjd->bid", adjacency.T, transformed_tan)
        else:
            agg_tan = torch.einsum("bij,bjd->bid", adjacency.transpose(1, 2), transformed_tan)
        out = self.manifold.expmap0(agg_tan, dim=-1, project=True)
        return self.manifold.projx(out, dim=-1)


class TangentFrechetReadout(nn.Module):
    """可微切空间 readout。

    使用低自由度的图级统计量生成 `z_global`。相比可学习节点注意力，mean/std/max
    不容易记住训练样本，也能把 ROI 分布的离散程度和极值响应带入全局双曲中心点。
    """

    def __init__(
        self,
        manifold,
        hidden_dim,
        dropout=0.0,
        output_radius=1.0,
        use_brain_network_prior=True,
        graph_readout_alpha=0.0,
        delta_readout_alpha=0.0,
        readout_mode="node_stats",
    ):
        super().__init__()
        self.manifold = manifold
        self.output_radius = float(output_radius)
        self.use_brain_network_prior = bool(use_brain_network_prior)
        self.graph_readout_alpha = min(max(float(graph_readout_alpha), 0.0), 1.0)
        self.delta_readout_alpha = min(max(float(delta_readout_alpha), 0.0), 1.0)
        self.readout_mode = str(readout_mode or "node_stats").lower()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        groups, prior_weights = _aal116_network_masks()
        self.network_names = [name for name, _ in groups]
        max_nodes = 116
        masks = torch.zeros(len(groups), max_nodes, dtype=torch.float32)
        for group_idx, (_, indices) in enumerate(groups):
            masks[group_idx, indices] = 1.0
        self.register_buffer("aal116_network_masks", masks)
        self.register_buffer("network_prior_weights", prior_weights)

    def forward(self, x, node_weights=None, reference=None):
        tangent = self.manifold.logmap0(x, dim=-1)
        reference_tangent = self.manifold.logmap0(reference, dim=-1) if reference is not None else None
        batch_size, n_nodes, hidden_dim = tangent.shape
        if node_weights is None:
            # 无参数显著性只用于可视化和轻量加权，避免训练集专属 attention 过拟合。
            saliency = tangent.norm(dim=-1)
            node_weights = torch.softmax(saliency, dim=-1)
        else:
            weights = node_weights.to(device=x.device, dtype=x.dtype)
            node_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        plain_mean = tangent.mean(dim=1)
        weighted_mean = torch.einsum("bn,bnd->bd", node_weights, tangent)
        readout_mean = (
            (1.0 - self.graph_readout_alpha) * plain_mean
            + self.graph_readout_alpha * weighted_mean
        )
        readout_source = tangent
        if reference_tangent is not None and self.delta_readout_alpha > 0:
            # 图传播差分强调“因果图改变了哪些 ROI 表征”，默认关闭，避免在小样本上引入不稳。
            delta_source = tangent - reference_tangent
            readout_source = (
                (1.0 - self.delta_readout_alpha) * tangent
                + self.delta_readout_alpha * delta_source
            )
        centered = readout_source - readout_source.mean(dim=1, keepdim=True)
        std_pool = centered.pow(2).mean(dim=1).clamp_min(1e-8).sqrt()
        max_pool = readout_source.max(dim=1).values
        if self.use_brain_network_prior and n_nodes == self.aal116_network_masks.shape[1]:
            masks = self.aal116_network_masks.to(device=x.device, dtype=x.dtype)
            masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
            network_summary = torch.einsum("gn,bnd->bgd", masks, tangent)
            network_attention = self.network_prior_weights.to(device=x.device, dtype=x.dtype)
            network_attention = network_attention / network_attention.sum().clamp_min(1e-8)
            node_prior = torch.einsum("g,gn->n", network_attention, masks)
            node_prior = node_prior / node_prior.mean().clamp_min(1e-8)
            # 固定 ROI gating 只轻微放大文献相关网络，不新增可学习网络权重。
            gate = 1.0 + 0.25 * (node_prior - 1.0)
            gated_tangent = tangent * gate.view(1, -1, 1)
            gated_source = readout_source * gate.view(1, -1, 1)
            plain_mean = gated_tangent.mean(dim=1)
            centered = gated_source - gated_source.mean(dim=1, keepdim=True)
            std_pool = centered.pow(2).mean(dim=1).clamp_min(1e-8).sqrt()
            max_pool = gated_source.max(dim=1).values
            saliency = gated_tangent.norm(dim=-1)
            feature_weights = torch.softmax(saliency, dim=-1)
            node_weights = 0.5 * node_weights + 0.5 * feature_weights
            node_weights = node_weights / node_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weighted_mean = torch.einsum("bn,bnd->bd", node_weights, gated_tangent)
            readout_mean = (
                (1.0 - self.graph_readout_alpha) * plain_mean
                + self.graph_readout_alpha * weighted_mean
            )
            if self.readout_mode in ("network", "network_stats", "aal_network"):
                # 小样本时直接从 116 个 ROI 做 max/std readout 很容易记住局部噪声。
                # 网络级 readout 先聚合到少量 AAL 功能网络，再计算双曲中心点，
                # 让模块 3 学习更稳定的脑网络模式，而不是单个 ROI 的偶然极值。
                network_source = network_summary
                network_mean = torch.einsum("g,bgd->bd", network_attention, network_source)
                network_centered = network_source - network_source.mean(dim=1, keepdim=True)
                readout_mean = network_mean
                std_pool = network_centered.pow(2).mean(dim=1).clamp_min(1e-8).sqrt()
                max_pool = network_source.max(dim=1).values
        else:
            network_summary = tangent.new_zeros(batch_size, 1, hidden_dim)
            network_attention = tangent.new_ones(1)

        stat_tangent = self.fusion(torch.cat([readout_mean, std_pool, max_pool], dim=-1))
        z_tangent = stat_tangent + readout_mean
        if self.output_radius > 0:
            norm = z_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.clamp(self.output_radius / norm, max=1.0)
            z_tangent = z_tangent * scale
        z_global = self.manifold.expmap0(z_tangent, dim=-1, project=True)
        z_global = self.manifold.projx(z_global, dim=-1)
        z_tangent = self.manifold.logmap0(z_global, dim=-1)
        return z_global, z_tangent, node_weights, network_summary, network_attention


class Module3HGCNReadout(nn.Module):
    """S-DeCI 模块 3：Backclip、Poincare 投影、HGCN 和全脑中心读取。"""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        curvature=1.0,
        backclip_radius=1.0,
        dropout=0.0,
        add_self_loop=True,
        adjacency_normalization="row",
        sample_correlation_mode="abs",
        use_brain_network_prior=True,
        causal_edge_dropout=0.0,
        residual_alpha=0.35,
        graph_readout_alpha=0.0,
        delta_readout_alpha=0.0,
        readout_mode="node_stats",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(int(num_layers), 1)
        self.add_self_loop = bool(add_self_loop)
        self.adjacency_normalization = adjacency_normalization
        self.sample_correlation_mode = sample_correlation_mode
        self.use_brain_network_prior = bool(use_brain_network_prior)
        self.causal_edge_dropout = float(causal_edge_dropout)
        self.residual_alpha = float(residual_alpha)
        self.graph_readout_alpha = float(graph_readout_alpha)
        self.delta_readout_alpha = float(delta_readout_alpha)
        self.manifold = geoopt.PoincareBall(c=float(curvature))
        self.backclip = Backclip(radius=backclip_radius)
        self.residual_projection = (
            nn.Identity() if self.input_dim == self.hidden_dim else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.readout = TangentFrechetReadout(
            self.manifold,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            output_radius=backclip_radius,
            use_brain_network_prior=self.use_brain_network_prior,
            graph_readout_alpha=self.graph_readout_alpha,
            delta_readout_alpha=self.delta_readout_alpha,
            readout_mode=readout_mode,
        )

        layers = []
        for idx in range(self.num_layers):
            in_dim = self.input_dim if idx == 0 else self.hidden_dim
            layers.append(
                HyperbolicGraphConvolution(
                    in_dim=in_dim,
                    out_dim=self.hidden_dim,
                    manifold=self.manifold,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

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
        adjacency = self._prepare_adjacency_values(adjacency, is_sample_correlation=is_sample_correlation)
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
            if adjacency.ndim == 2:
                adjacency = adjacency + identity
            else:
                adjacency = adjacency + identity.unsqueeze(0)

        if self.adjacency_normalization == "none":
            return adjacency
        if self.adjacency_normalization == "sym":
            degree = adjacency.sum(dim=-1).clamp_min(1e-8)
            inv_sqrt = degree.rsqrt()
            if adjacency.ndim == 2:
                return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
            return inv_sqrt[:, :, None] * adjacency * inv_sqrt[:, None, :]

        degree = adjacency.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        return adjacency / degree

    def _apply_edge_dropout(self, adjacency):
        if not self.training or self.causal_edge_dropout <= 0:
            return adjacency
        drop_prob = min(max(self.causal_edge_dropout, 0.0), 0.95)
        keep_prob = 1.0 - drop_prob
        if adjacency.ndim == 2:
            n_nodes = adjacency.shape[0]
            off_diag = (1.0 - torch.eye(n_nodes, device=adjacency.device, dtype=adjacency.dtype)).bool()
            mask = torch.rand_like(adjacency).lt(keep_prob) | (~off_diag)
            return adjacency * mask.to(adjacency.dtype) / keep_prob
        n_nodes = adjacency.shape[1]
        off_diag = (1.0 - torch.eye(n_nodes, device=adjacency.device, dtype=adjacency.dtype)).bool()
        off_diag = off_diag.unsqueeze(0)
        mask = torch.rand_like(adjacency).lt(keep_prob) | (~off_diag)
        return adjacency * mask.to(adjacency.dtype) / keep_prob

    def forward(self, cycle_features, adjacency, is_sample_correlation=False):
        if cycle_features.ndim != 3:
            raise ValueError(
                f"Expected cycle_features with shape [B, N, D], got {tuple(cycle_features.shape)}."
            )
        expected_2d = cycle_features.shape[1:2] * 2
        expected_3d = (cycle_features.shape[0], cycle_features.shape[1], cycle_features.shape[1])
        if adjacency.ndim == 2 and adjacency.shape != expected_2d:
            raise ValueError(
                f"Expected adjacency shape [{cycle_features.shape[1]}, {cycle_features.shape[1]}], "
                f"got {tuple(adjacency.shape)}."
            )
        if adjacency.ndim == 3 and adjacency.shape != expected_3d:
            raise ValueError(
                f"Expected batch adjacency shape {expected_3d}, got {tuple(adjacency.shape)}."
            )
        if adjacency.ndim not in (2, 3):
            raise ValueError(
                f"Expected adjacency with shape [N, N] or [B, N, N], got {tuple(adjacency.shape)}."
            )

        c_clipped = self.backclip(cycle_features)
        h = self.manifold.expmap0(c_clipped, dim=-1, project=True)
        h0 = self.manifold.projx(h, dim=-1)
        adjacency = adjacency.to(device=cycle_features.device, dtype=cycle_features.dtype)
        adjacency = self._apply_edge_dropout(adjacency)
        normalized_adjacency = self._normalize_adjacency(
            adjacency,
            is_sample_correlation=is_sample_correlation,
        )

        h_gcn = h0
        for layer in self.layers:
            h_gcn = layer(h_gcn, normalized_adjacency)

        if self.residual_alpha > 0:
            alpha = min(max(self.residual_alpha, 0.0), 1.0)
            h0_tangent = self.manifold.logmap0(h0, dim=-1)
            h0_residual = self.residual_projection(h0_tangent)
            h0_residual = self.manifold.expmap0(h0_residual, dim=-1, project=True)
            h0_residual = self.manifold.projx(h0_residual, dim=-1)
            h_gcn = (1.0 - alpha) * h_gcn + alpha * h0_residual

        prepared_adjacency = self._prepare_adjacency_values(
            normalized_adjacency,
            is_sample_correlation=is_sample_correlation,
        )
        if prepared_adjacency.ndim == 2:
            graph_saliency = prepared_adjacency.sum(dim=0) + prepared_adjacency.sum(dim=1)
            graph_saliency = graph_saliency.unsqueeze(0).expand(cycle_features.shape[0], -1)
        else:
            graph_saliency = prepared_adjacency.sum(dim=1) + prepared_adjacency.sum(dim=2)
        node_weights = torch.softmax(graph_saliency, dim=-1)

        z_global, z_tangent, node_attention, network_summary, network_attention = self.readout(
            h_gcn,
            node_weights=node_weights,
            reference=h0_residual if self.residual_alpha > 0 else h0,
        )
        return Module3HGCNOutput(
            c_clipped=c_clipped,
            h0=h0,
            h_gcn=h_gcn,
            z_global=z_global,
            z_tangent=z_tangent,
            node_attention=node_attention,
            network_summary=network_summary,
            network_attention=network_attention,
            normalized_adjacency=normalized_adjacency,
        )
