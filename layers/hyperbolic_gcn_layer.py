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
    causal_role_poincare: torch.Tensor = None
    causal_role_tangent: torch.Tensor = None
    causal_role_weights: torch.Tensor = None


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
        readout_mode="mean_std",
        use_radius_head=False,
        radius_min_ratio=0.25,
        causal_attention_heads=4,
        causal_attention_graph_weight=0.5,
        network_gate_strength=0.35,
        use_graph_degree_encoding=False,
        graph_degree_encoding_weight=0.1,
        causal_subnetwork_count=4,
        causal_subnetwork_topk=12,
        causal_subnetwork_weight=0.5,
        einstein_readout_weight=0.0,
        causal_role_temperature=1.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.output_radius = float(output_radius)
        self.use_radius_head = bool(use_radius_head)
        self.radius_min_ratio = min(max(float(radius_min_ratio), 0.0), 1.0)
        self.use_brain_network_prior = bool(use_brain_network_prior)
        self.graph_readout_alpha = min(max(float(graph_readout_alpha), 0.0), 1.0)
        self.delta_readout_alpha = min(max(float(delta_readout_alpha), 0.0), 1.0)
        self.readout_mode = str(readout_mode or "mean_std").lower()
        self.causal_attention_heads = max(int(causal_attention_heads), 1)
        self.causal_attention_graph_weight = max(float(causal_attention_graph_weight), 0.0)
        self.network_gate_strength = max(float(network_gate_strength), 0.0)
        self.causal_subnetwork_count = max(int(causal_subnetwork_count), 1)
        self.causal_subnetwork_topk = max(int(causal_subnetwork_topk), 1)
        self.causal_subnetwork_weight = min(max(float(causal_subnetwork_weight), 0.0), 1.0)
        self.einstein_readout_weight = min(max(float(einstein_readout_weight), 0.0), 1.0)
        self.causal_role_temperature = max(float(causal_role_temperature), 1e-4)
        fusion_input_dim = hidden_dim * 2 if self.readout_mode in (
            "mean_std",
            "causal_weighted_mean_std",
            "graph_weighted_mean_std",
        ) else hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.causal_query = nn.Parameter(torch.empty(self.causal_attention_heads, hidden_dim))
        self.causal_attention_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.causal_attention_heads),
            nn.Linear(hidden_dim * self.causal_attention_heads, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.use_radius_head:
            self.radius_head = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, max(hidden_dim // 2, 1)),
                nn.GELU(),
                nn.Linear(max(hidden_dim // 2, 1), 1),
            )
            nn.init.zeros_(self.radius_head[-1].weight)
            nn.init.zeros_(self.radius_head[-1].bias)
        else:
            self.radius_head = None
        groups, prior_weights = _aal116_network_masks()
        self.network_names = [name for name, _ in groups]
        max_nodes = 116
        masks = torch.zeros(len(groups), max_nodes, dtype=torch.float32)
        for group_idx, (_, indices) in enumerate(groups):
            masks[group_idx, indices] = 1.0
        self.register_buffer("aal116_network_masks", masks)
        self.register_buffer("network_prior_weights", prior_weights)
        nn.init.xavier_uniform_(self.causal_query)

    def _einstein_midpoint(self, points, weights):
        """用 Poincare Ball 的 Einstein midpoint 近似双曲中心。

        原点切空间均值在节点远离原点时会产生几何失真；Einstein midpoint 先转到 Klein
        坐标做 Lorentz gamma 加权中心，再映回 Poincare Ball。
        """

        if points.ndim != 3 or weights.ndim != 2:
            return None
        weights = weights.to(device=points.device, dtype=points.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        curvature = self.manifold.c.to(device=points.device, dtype=points.dtype)
        norm_sq = points.pow(2).sum(dim=-1, keepdim=True)
        klein = 2.0 * points / (1.0 + curvature * norm_sq).clamp_min(1e-8)
        klein_norm_sq = klein.pow(2).sum(dim=-1, keepdim=True)
        scaled_klein_norm = (curvature * klein_norm_sq).clamp(max=1.0 - 1e-6)
        gamma = torch.rsqrt((1.0 - scaled_klein_norm).clamp_min(1e-8))
        weighted = weights.unsqueeze(-1) * gamma
        midpoint_klein = (weighted * klein).sum(dim=1) / weighted.sum(dim=1).clamp_min(1e-8)
        midpoint_klein_norm_sq = midpoint_klein.pow(2).sum(dim=-1, keepdim=True)
        denom = 1.0 + torch.sqrt(
            (1.0 - curvature * midpoint_klein_norm_sq).clamp_min(1e-8)
        )
        midpoint_poincare = midpoint_klein / denom.clamp_min(1e-8)
        return self.manifold.projx(midpoint_poincare, dim=-1)

    def _einstein_midpoint_tangent(self, points, weights):
        """返回 Einstein midpoint 的原点切空间表示，用于全局读出校正。"""

        if self.einstein_readout_weight <= 0:
            return None
        midpoint = self._einstein_midpoint(points, weights)
        if midpoint is None:
            return None
        return self.manifold.logmap0(midpoint, dim=-1)

    def causal_role_centers(self, points, adjacency):
        """按有向图的 source/sink/hub/global 角色读取四个双曲中心。"""

        if points.ndim != 3 or adjacency is None or adjacency.ndim not in (2, 3):
            return None, None, None
        batch_size, n_nodes, _ = points.shape
        graph = torch.nan_to_num(
            adjacency.to(device=points.device, dtype=points.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).abs()
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(batch_size, -1, -1)
        if graph.shape[0] != batch_size or graph.shape[1:] != (n_nodes, n_nodes):
            return None, None, None
        eye = torch.eye(n_nodes, device=points.device, dtype=points.dtype).unsqueeze(0)
        graph = graph * (1.0 - eye)
        out_degree = graph.sum(dim=-1)
        in_degree = graph.sum(dim=-2)

        def normalized_role_weight(score):
            centered = score - score.mean(dim=-1, keepdim=True)
            scaled = centered / score.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return torch.softmax(scaled / self.causal_role_temperature, dim=-1)

        role_weights = torch.stack(
            [
                torch.full_like(out_degree, 1.0 / max(n_nodes, 1)),
                normalized_role_weight(out_degree - in_degree),
                normalized_role_weight(in_degree - out_degree),
                normalized_role_weight(out_degree + in_degree),
            ],
            dim=1,
        )
        role_points = []
        for role_idx in range(role_weights.shape[1]):
            role_points.append(self._einstein_midpoint(points, role_weights[:, role_idx, :]))
        role_poincare = torch.stack(role_points, dim=1)
        role_tangent = self.manifold.logmap0(role_poincare, dim=-1)
        return role_poincare, role_tangent, role_weights

    def _causal_subnetwork_readout(self, source, adjacency):
        """按因果图强度自动抽取若干 ROI 子网络，并返回子网络级读出。"""

        batch_size, n_nodes, hidden_dim = source.shape
        if adjacency is None or adjacency.ndim not in (2, 3):
            return None, None, None
        graph = torch.nan_to_num(
            adjacency.to(device=source.device, dtype=source.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).abs()
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(batch_size, -1, -1)
        if graph.shape[0] != batch_size or graph.shape[1] != n_nodes or graph.shape[2] != n_nodes:
            return None, None, None
        eye = torch.eye(n_nodes, device=source.device, dtype=source.dtype).unsqueeze(0)
        graph = graph * (1.0 - eye)
        graph_strength = graph.sum(dim=-1) + graph.sum(dim=-2)
        if torch.all(graph_strength <= 0):
            return None, None, None

        subnetwork_count = min(self.causal_subnetwork_count, n_nodes)
        topk = min(self.causal_subnetwork_topk, n_nodes)
        seed_scores, seed_indices = torch.topk(graph_strength, k=subnetwork_count, dim=-1)
        subnet_contexts = []
        subnet_weights = []
        subnet_node_weights = []
        for group_idx in range(subnetwork_count):
            seeds = seed_indices[:, group_idx]
            seed_rows = graph.gather(
                1,
                seeds.view(batch_size, 1, 1).expand(-1, 1, n_nodes),
            ).squeeze(1)
            seed_cols = graph.gather(
                2,
                seeds.view(batch_size, 1, 1).expand(-1, n_nodes, 1),
            ).squeeze(2)
            membership_score = seed_rows + seed_cols + 0.25 * graph_strength
            _, roi_indices = torch.topk(membership_score, k=topk, dim=-1)
            roi_mask = torch.zeros_like(membership_score)
            roi_mask.scatter_(1, roi_indices, 1.0)
            roi_logits = membership_score.masked_fill(roi_mask <= 0, -1e4)
            roi_weights = torch.softmax(roi_logits, dim=-1)
            context = torch.einsum("bn,bnd->bd", roi_weights, source)
            subnet_contexts.append(context)
            subnet_weights.append(seed_scores[:, group_idx])
            subnet_node_weights.append(roi_weights)
        subnet_context = torch.stack(subnet_contexts, dim=1)
        subnet_weight = torch.stack(subnet_weights, dim=1)
        subnet_attention = torch.softmax(subnet_weight, dim=-1)
        subnet_node_weight = torch.stack(subnet_node_weights, dim=1)
        node_importance = torch.einsum("bk,bkn->bn", subnet_attention, subnet_node_weight)
        node_importance = node_importance / node_importance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        subnet_mean = torch.einsum("bn,bnd->bd", node_importance, source)
        return subnet_mean, node_importance, subnet_attention

    def forward(self, x, node_weights=None, reference=None, sample_adjacency=None):
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
        if self.readout_mode in ("causal_weighted_mean_std", "graph_weighted_mean_std"):
            # 模块2/分类图只作为连续软权重参与 mean/std 统计，不做 top-k 子网硬选择。
            weighted_mean = torch.einsum("bn,bnd->bd", node_weights, readout_source)
            weighted_centered = readout_source - weighted_mean.unsqueeze(1)
            weighted_var = torch.einsum(
                "bn,bnd->bd",
                node_weights,
                weighted_centered.pow(2),
            )
            readout_mean = weighted_mean
            std_pool = weighted_var.clamp_min(1e-8).sqrt()
        use_network_masks = (
            n_nodes == self.aal116_network_masks.shape[1]
            and (
                self.use_brain_network_prior
                or self.readout_mode in (
                    "network",
                    "network_stats",
                    "aal_network",
                    "network_gated_node_stats",
                    "fc_gated_node_stats",
                )
            )
        )
        if use_network_masks:
            raw_masks = self.aal116_network_masks.to(device=x.device, dtype=x.dtype)
            masks = raw_masks / raw_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
            network_summary = torch.einsum("gn,bnd->bgd", masks, tangent)
            network_attention = self.network_prior_weights.to(device=x.device, dtype=x.dtype)
            network_attention = network_attention / network_attention.sum().clamp_min(1e-8)
            if self.use_brain_network_prior:
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
            else:
                network_attention = tangent.new_full(
                    (masks.shape[0],), 1.0 / max(masks.shape[0], 1)
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
            elif (
                self.readout_mode in ("network_gated_node_stats", "fc_gated_node_stats")
                and sample_adjacency is not None
                and sample_adjacency.ndim == 3
                and sample_adjacency.shape[-1] == masks.shape[-1]
                and self.network_gate_strength > 0
            ):
                # 用样本 FC 的网络级连接强度调节 ROI 池化权重。
                # 这里 FC 只影响 readout 权重，不直接平移双曲坐标，避免污染 HPEC 的角度/半径结构。
                fc_adj = torch.nan_to_num(sample_adjacency.to(device=x.device, dtype=x.dtype), nan=0.0)
                fc_adj = fc_adj.abs()
                eye = torch.eye(n_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
                fc_adj = fc_adj * (1.0 - eye)
                network_fc = torch.einsum("gn,bnm,hm->bgh", masks, fc_adj, masks)
                network_strength = network_fc.mean(dim=-1)
                network_strength = network_strength / network_strength.mean(dim=-1, keepdim=True).clamp_min(1e-8)
                sample_network_gate = 1.0 + self.network_gate_strength * (network_strength - 1.0)
                sample_network_gate = sample_network_gate.clamp(0.5, 1.5)
                membership = raw_masks.sum(dim=0, keepdim=True).clamp_min(1.0)
                sample_node_gate = torch.einsum("bg,gn->bn", sample_network_gate, raw_masks)
                sample_node_gate = sample_node_gate / membership
                sample_node_gate = sample_node_gate / sample_node_gate.mean(dim=-1, keepdim=True).clamp_min(1e-8)

                gated_tangent = tangent * sample_node_gate.unsqueeze(-1)
                gated_source = readout_source * sample_node_gate.unsqueeze(-1)
                plain_mean = gated_tangent.mean(dim=1)
                centered = gated_source - gated_source.mean(dim=1, keepdim=True)
                std_pool = centered.pow(2).mean(dim=1).clamp_min(1e-8).sqrt()
                max_pool = gated_source.max(dim=1).values
                gated_weights = node_weights * sample_node_gate
                gated_weights = gated_weights / gated_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                weighted_mean = torch.einsum("bn,bnd->bd", gated_weights, gated_tangent)
                readout_mean = (
                    (1.0 - self.graph_readout_alpha) * plain_mean
                    + self.graph_readout_alpha * weighted_mean
                )
                node_weights = gated_weights
                network_attention = sample_network_gate / sample_network_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            network_summary = tangent.new_zeros(batch_size, 1, hidden_dim)
            network_attention = tangent.new_ones(1)

        if self.readout_mode in ("causal_attention", "causal_attn", "graph_attention"):
            query = F.normalize(self.causal_query.to(device=x.device, dtype=x.dtype), p=2, dim=-1)
            node_repr = F.normalize(readout_source, p=2, dim=-1)
            attention_logits = torch.einsum("hd,bnd->bhn", query, node_repr)
            if node_weights is not None and self.causal_attention_graph_weight > 0:
                graph_prior = node_weights.clamp_min(1e-8).log().unsqueeze(1)
                attention_logits = attention_logits + self.causal_attention_graph_weight * graph_prior
            head_attention = torch.softmax(attention_logits, dim=-1)
            head_context = torch.einsum("bhn,bnd->bhd", head_attention, readout_source)
            attention_tangent = self.causal_attention_proj(head_context.flatten(start_dim=1))
            readout_mean = 0.5 * readout_mean + 0.5 * attention_tangent
            node_weights = head_attention.mean(dim=1)
            node_weights = node_weights / node_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.readout_mode in ("causal_subnetwork", "causal_subnet", "graph_subnetwork"):
            # 因果子网络读出：用模块2/分类图的强边自动形成多个 ROI 子网络，
            # 让 z_global 表示“若干因果功能团”的组合，而不是简单全脑平均。
            subnet_stats = self._causal_subnetwork_readout(
                readout_source,
                sample_adjacency,
            )
            if subnet_stats[0] is not None:
                subnet_mean, subnet_node_weights, subnet_attention = subnet_stats
                node_weights = (
                    (1.0 - self.causal_subnetwork_weight) * node_weights
                    + self.causal_subnetwork_weight * subnet_node_weights
                )
                node_weights = node_weights / node_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                readout_mean = (
                    (1.0 - self.causal_subnetwork_weight) * readout_mean
                    + self.causal_subnetwork_weight * subnet_mean
                )
                network_summary = subnet_mean.unsqueeze(1)
                network_attention = subnet_attention

        if self.readout_mode in ("mean_std", "causal_weighted_mean_std", "graph_weighted_mean_std"):
            # fMRI 小样本场景中 max pooling 容易放大单 ROI 噪声；mean+std 保留整体激活和离散度。
            pooled_stats = torch.cat([readout_mean, std_pool], dim=-1)
        else:
            pooled_stats = torch.cat([readout_mean, std_pool, max_pool], dim=-1)
        stat_tangent = self.fusion(pooled_stats)
        z_tangent = stat_tangent + readout_mean
        einstein_tangent = self._einstein_midpoint_tangent(x, node_weights)
        if einstein_tangent is not None:
            # 只校正全局方向，不替代 mean/std 统计，避免小样本下新增高自由度读出。
            z_tangent = (
                (1.0 - self.einstein_readout_weight) * z_tangent
                + self.einstein_readout_weight * einstein_tangent
            )
        if self.output_radius > 0:
            norm = z_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.clamp(self.output_radius / norm, max=1.0)
            z_tangent = z_tangent * scale
            if self.radius_head is not None:
                # HPEC 需要全局点不要长期贴近原点。半径由统计特征预测，方向仍来自图读出。
                min_radius = min(max(self.radius_min_ratio * self.output_radius, 0.0), self.output_radius)
                radius_gate = torch.sigmoid(self.radius_head(pooled_stats))
                target_radius = min_radius + (self.output_radius - min_radius) * radius_gate
                current_norm = z_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                z_tangent = z_tangent / current_norm * torch.maximum(current_norm, target_radius)
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
        readout_mode="mean_std",
        use_radius_head=False,
        radius_min_ratio=0.25,
        causal_attention_heads=4,
        causal_attention_graph_weight=0.5,
        network_gate_strength=0.35,
        use_graph_degree_encoding=False,
        graph_degree_encoding_weight=0.1,
        causal_subnetwork_count=4,
        causal_subnetwork_topk=12,
        causal_subnetwork_weight=0.5,
        einstein_readout_weight=0.0,
        use_causal_role_readout=False,
        causal_role_temperature=1.0,
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
        self.readout_mode = str(readout_mode or "mean_std").lower()
        self.use_graph_degree_encoding = bool(use_graph_degree_encoding)
        self.use_causal_role_readout = bool(use_causal_role_readout)
        self.graph_degree_encoding_weight = max(float(graph_degree_encoding_weight), 0.0)
        self.manifold = geoopt.PoincareBall(c=float(curvature))
        self.backclip = Backclip(radius=backclip_radius)
        self.residual_projection = (
            nn.Identity() if self.input_dim == self.hidden_dim else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.graph_degree_projection = (
            nn.Sequential(
                nn.LayerNorm(2),
                nn.Linear(2, self.input_dim),
                nn.Tanh(),
            )
            if self.use_graph_degree_encoding and self.graph_degree_encoding_weight > 0
            else None
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
            use_radius_head=use_radius_head,
            radius_min_ratio=radius_min_ratio,
            causal_attention_heads=causal_attention_heads,
            causal_attention_graph_weight=causal_attention_graph_weight,
            network_gate_strength=network_gate_strength,
            causal_subnetwork_count=causal_subnetwork_count,
            causal_subnetwork_topk=causal_subnetwork_topk,
            causal_subnetwork_weight=causal_subnetwork_weight,
            einstein_readout_weight=einstein_readout_weight,
            causal_role_temperature=causal_role_temperature,
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

    def _graph_degree_encoding(self, adjacency, is_sample_correlation, batch_size, n_nodes, device, dtype):
        if self.graph_degree_projection is None:
            return None
        graph = self._prepare_adjacency_values(adjacency, is_sample_correlation=is_sample_correlation)
        graph = graph.to(device=device, dtype=dtype)
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(batch_size, -1, -1)
        if graph.shape[1] != n_nodes or graph.shape[2] != n_nodes:
            return None
        eye = torch.eye(n_nodes, device=device, dtype=dtype).unsqueeze(0)
        graph = graph * (1.0 - eye)
        out_degree = graph.sum(dim=-1)
        in_degree = graph.sum(dim=-2)
        degree = torch.stack([out_degree, in_degree], dim=-1)
        degree = torch.log1p(degree.clamp_min(0.0))
        degree = degree - degree.mean(dim=1, keepdim=True)
        degree = degree / degree.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return self.graph_degree_projection(degree)

    def forward(self, cycle_features, adjacency, is_sample_correlation=False, sample_correlation=None):
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
        degree_encoding = self._graph_degree_encoding(
            adjacency,
            is_sample_correlation=is_sample_correlation,
            batch_size=cycle_features.shape[0],
            n_nodes=cycle_features.shape[1],
            device=cycle_features.device,
            dtype=cycle_features.dtype,
        )
        if degree_encoding is not None:
            c_clipped = self.backclip(c_clipped + self.graph_degree_encoding_weight * degree_encoding)
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

        readout_adjacency = (
            adjacency
            if self.readout_mode in ("causal_subnetwork", "causal_subnet", "graph_subnetwork")
            else (sample_correlation if sample_correlation is not None else normalized_adjacency)
        )
        z_global, z_tangent, node_attention, network_summary, network_attention = self.readout(
            h_gcn,
            node_weights=node_weights,
            reference=h0_residual if self.residual_alpha > 0 else h0,
            sample_adjacency=readout_adjacency,
        )
        causal_role_poincare = None
        causal_role_tangent = None
        causal_role_weights = None
        if self.use_causal_role_readout:
            causal_role_poincare, causal_role_tangent, causal_role_weights = (
                self.readout.causal_role_centers(h_gcn, adjacency)
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
            causal_role_poincare=causal_role_poincare,
            causal_role_tangent=causal_role_tangent,
            causal_role_weights=causal_role_weights,
        )
