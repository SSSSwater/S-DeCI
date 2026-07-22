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
    graph_only_logits: torch.Tensor | None = None
    fc_only_logits: torch.Tensor | None = None


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


class DirectedEuclideanGraphConvolution(nn.Module):
    """分别编码因果图的入边、出边及二者差异。"""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(int(in_dim) * 3)
        self.linear = nn.Linear(int(in_dim) * 3, int(out_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x, incoming_adjacency, outgoing_adjacency):
        if incoming_adjacency.ndim == 2:
            incoming = torch.einsum("ij,bjd->bid", incoming_adjacency.T, x)
            outgoing = torch.einsum("ij,bjd->bid", outgoing_adjacency, x)
        elif incoming_adjacency.ndim == 3:
            incoming = torch.einsum(
                "bij,bjd->bid", incoming_adjacency.transpose(1, 2), x
            )
            outgoing = torch.einsum("bij,bjd->bid", outgoing_adjacency, x)
        else:
            raise ValueError(
                "Expected directed adjacency with shape [N, N] or [B, N, N], "
                f"got {tuple(incoming_adjacency.shape)}."
            )
        directional = torch.cat(
            [incoming, outgoing, outgoing - incoming],
            dim=-1,
        )
        return self.dropout(F.gelu(self.linear(self.norm(directional))))


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
        use_graph_stats=False,
        graph_stats_mode="basic",
        graph_stats_input="normalized",
        readout_mode="mean",
        input_residual_weight=0.0,
        external_feature_dim=0,
        edge_readout_topk=0,
        directional_propagation=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = max(int(num_layers), 1)
        self.add_self_loop = bool(add_self_loop)
        self.adjacency_normalization = adjacency_normalization
        self.sample_correlation_mode = sample_correlation_mode
        self.use_graph_stats = bool(use_graph_stats)
        self.graph_stats_mode = str(graph_stats_mode or "basic").lower()
        if self.graph_stats_mode not in ("basic", "causal"):
            raise ValueError(
                f"Unsupported graph_stats_mode={self.graph_stats_mode!r}. "
                "Use 'basic' or 'causal'."
            )
        self.graph_stats_input = str(graph_stats_input or "normalized").lower()
        if self.graph_stats_input not in ("normalized", "raw"):
            raise ValueError(
                f"Unsupported graph_stats_input={self.graph_stats_input!r}. "
                "Use 'normalized' or 'raw'."
            )
        self.input_residual_weight = float(input_residual_weight)
        # 外部特征维度（如 FC 生物标志分支的 embedding）；0 表示不融合，行为与旧版完全一致。
        self.external_feature_dim = max(int(external_feature_dim), 0)
        self.edge_readout_topk = max(int(edge_readout_topk), 0)
        self.directional_propagation = bool(directional_propagation)
        self.readout_mode = str(readout_mode or "mean").lower()
        if self.readout_mode not in ("mean", "attention", "mean_max", "mean_std"):
            raise ValueError(
                f"Unsupported gcn fallback readout_mode={self.readout_mode!r}. "
                "Use 'mean', 'attention', 'mean_max' or 'mean_std'."
            )

        layers = []
        for idx in range(self.num_layers):
            in_dim = self.input_dim if idx == 0 else self.hidden_dim
            layer_cls = (
                DirectedEuclideanGraphConvolution
                if self.directional_propagation
                else EuclideanGraphConvolution
            )
            layers.append(layer_cls(in_dim, self.hidden_dim, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        if self.input_dim == self.hidden_dim:
            self.input_residual_projection = nn.Identity()
        else:
            self.input_residual_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_residual_norm = nn.LayerNorm(self.hidden_dim)
        graph_stats_dim = 16 if self.use_graph_stats and self.graph_stats_mode == "causal" else 4 if self.use_graph_stats else 0
        readout_dim = (
            self.hidden_dim * 2
            if self.readout_mode in ("mean_max", "mean_std")
            else self.hidden_dim
        )
        edge_readout_dim = self.hidden_dim if self.edge_readout_topk > 0 else 0
        graph_feature_dim = readout_dim + graph_stats_dim + edge_readout_dim
        classifier_input_dim = graph_feature_dim + self.external_feature_dim
        if self.readout_mode == "attention":
            self.node_attention = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.node_attention = None
        if self.use_graph_stats:
            self.graph_stats_norm = nn.LayerNorm(graph_stats_dim)
            self.classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Linear(classifier_input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.graph_stats_norm = None
            self.classifier = nn.Linear(classifier_input_dim, self.out_dim)
        if self.edge_readout_topk > 0:
            edge_pair_dim = self.hidden_dim * 4 + 1
            self.edge_readout_projection = nn.Sequential(
                nn.LayerNorm(edge_pair_dim),
                nn.Linear(edge_pair_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.edge_readout_projection = None
        # 分支诊断复用同一个已训练分类器做输入置零反事实，不再维护未训练的随机头。
        self.graph_only_classifier = None
        self.fc_only_classifier = None

    def _graph_stats(self, graph_adjacency, batch_size):
        adjacency = graph_adjacency
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        in_strength = adjacency.sum(dim=-2)
        out_strength = adjacency.sum(dim=-1)
        basic_stats = [
            in_strength.mean(dim=-1),
            in_strength.std(dim=-1, unbiased=False),
            out_strength.mean(dim=-1),
            out_strength.std(dim=-1, unbiased=False),
        ]
        if self.graph_stats_mode == "basic":
            stats = torch.stack(basic_stats, dim=-1)
            return self.graph_stats_norm(stats) if self.graph_stats_norm is not None else stats

        # causal 模式把有向图的结构差异显式送入分类头：
        # 1) 入/出度分布，2) 出入度不平衡，3) 非对称边强度，4) 最强因果边强度。
        n_nodes = adjacency.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        graph = adjacency * off_diag
        asym = (graph - graph.transpose(-1, -2)).abs()
        balance = out_strength - in_strength
        flat_edges = graph.reshape(graph.shape[0], -1)
        k = min(max(int(round(n_nodes * 0.1)), 1), flat_edges.shape[-1])
        topk = torch.topk(flat_edges, k=k, dim=-1).values
        density_threshold = flat_edges.mean(dim=-1, keepdim=True) + flat_edges.std(dim=-1, keepdim=True, unbiased=False)
        density = (flat_edges > density_threshold).to(flat_edges.dtype).mean(dim=-1)
        stats = torch.stack(
            [
                *basic_stats,
                balance.mean(dim=-1),
                balance.std(dim=-1, unbiased=False),
                balance.abs().mean(dim=-1),
                asym.mean(dim=(-2, -1)),
                asym.std(dim=(-2, -1), unbiased=False),
                graph.mean(dim=(-2, -1)),
                graph.std(dim=(-2, -1), unbiased=False),
                graph.max(dim=-1).values.mean(dim=-1),
                graph.max(dim=-2).values.mean(dim=-1),
                topk.mean(dim=-1),
                topk.std(dim=-1, unbiased=False),
                density,
            ],
            dim=-1,
        )
        return self.graph_stats_norm(stats) if self.graph_stats_norm is not None else stats

    def _readout(self, h):
        if self.readout_mode == "attention":
            score = self.node_attention(h).squeeze(-1)
            weight = torch.softmax(score, dim=1).unsqueeze(-1)
            return (h * weight).sum(dim=1)
        if self.readout_mode == "mean_max":
            return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        if self.readout_mode == "mean_std":
            return torch.cat(
                [h.mean(dim=1), h.std(dim=1, unbiased=False)],
                dim=-1,
            )
        return h.mean(dim=1)

    def _edge_readout(self, h, graph_adjacency):
        if self.edge_readout_projection is None or self.edge_readout_topk <= 0:
            return None
        adjacency = graph_adjacency
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(h.shape[0], -1, -1)
        adjacency = adjacency.to(device=h.device, dtype=h.dtype)
        n_nodes = adjacency.shape[-1]
        off_diag = 1.0 - torch.eye(n_nodes, device=h.device, dtype=h.dtype).unsqueeze(0)
        edge_score = (adjacency * off_diag).clamp_min(0.0)
        flat_score = edge_score.reshape(edge_score.shape[0], -1)
        k = min(self.edge_readout_topk, flat_score.shape[-1])
        if k <= 0:
            return None
        values, indices = torch.topk(flat_score, k=k, dim=-1)
        src = torch.div(indices, n_nodes, rounding_mode="floor")
        dst = indices.remainder(n_nodes)
        batch_index = torch.arange(h.shape[0], device=h.device).unsqueeze(-1)
        src_h = h[batch_index, src]
        dst_h = h[batch_index, dst]
        edge_features = torch.cat(
            [
                src_h,
                dst_h,
                (src_h - dst_h).abs(),
                src_h * dst_h,
                values.unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_repr = self.edge_readout_projection(edge_features)
        weights = torch.softmax(values, dim=-1).unsqueeze(-1)
        return (edge_repr * weights).sum(dim=1)

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

    def forward(self, node_features, adjacency, is_sample_correlation=False, external_features=None):
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

        adjacency = adjacency.to(device=node_features.device, dtype=node_features.dtype)
        graph_stats_adjacency = self._prepare_adjacency_values(
            adjacency,
            is_sample_correlation=is_sample_correlation,
        )
        normalized_adjacency = self._normalize_adjacency(
            adjacency,
            is_sample_correlation=is_sample_correlation,
        )
        outgoing_adjacency = None
        if self.directional_propagation:
            # A[i,j] 表示 i -> j；这里单独构造按源节点归一化的出边图。
            outgoing_adjacency = self._normalize_adjacency(
                adjacency.transpose(-1, -2),
                is_sample_correlation=is_sample_correlation,
            ).transpose(-1, -2)
        h = node_features
        for layer in self.layers:
            if self.directional_propagation:
                h = layer(h, normalized_adjacency, outgoing_adjacency)
            else:
                h = layer(h, normalized_adjacency)
        if self.input_residual_weight > 0:
            residual = self.input_residual_projection(node_features)
            h = self.input_residual_norm(h + self.input_residual_weight * residual)
        readout = self._readout(h)
        edge_readout = self._edge_readout(h, graph_stats_adjacency)
        if edge_readout is not None:
            readout = torch.cat([readout, edge_readout], dim=-1)
        if self.use_graph_stats:
            stats_source = (
                graph_stats_adjacency
                if self.graph_stats_input == "raw"
                else normalized_adjacency
            )
            graph_stats = self._graph_stats(stats_source, node_features.shape[0])
            readout = torch.cat([readout, graph_stats], dim=-1)
        graph_only_readout = readout
        fc_only_logits = None
        if self.external_feature_dim > 0:
            # 外部特征（如 FC 生物标志 embedding）在 readout 层与图特征融合，再过分类头。
            # 该 batch 缺失外部特征时补零，保证 classifier 输入维度恒定、不影响其它数据集路径。
            if external_features is None:
                external_features = torch.zeros(
                    node_features.shape[0],
                    self.external_feature_dim,
                    device=readout.device,
                    dtype=readout.dtype,
                )
            else:
                external_features = external_features.to(device=readout.device, dtype=readout.dtype)
                if external_features.ndim != 2 or external_features.shape[-1] != self.external_feature_dim:
                    raise ValueError(
                        f"Expected external_features shape [B, {self.external_feature_dim}], "
                        f"got {tuple(external_features.shape)}."
                    )
            readout = torch.cat([readout, external_features], dim=-1)
        logits = self.classifier(readout)
        if self.external_feature_dim > 0:
            # 反事实诊断不参与梯度，并临时关闭 classifier dropout，避免污染主训练 RNG。
            classifier_was_training = self.classifier.training
            self.classifier.eval()
            graph_only_input = torch.cat(
                [graph_only_readout, torch.zeros_like(external_features)],
                dim=-1,
            )
            fc_only_input = torch.cat(
                [torch.zeros_like(graph_only_readout), external_features],
                dim=-1,
            )
            graph_only_logits = self.classifier(graph_only_input)
            fc_only_logits = self.classifier(fc_only_input)
            if classifier_was_training:
                self.classifier.train()
        else:
            graph_only_logits = logits.detach()
        return GCNFallbackOutput(
            normalized_adjacency=normalized_adjacency,
            h_gcn=h,
            readout=readout,
            logits=logits,
            graph_only_logits=graph_only_logits,
            fc_only_logits=fc_only_logits,
        )
