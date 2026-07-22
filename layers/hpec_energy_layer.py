from dataclasses import dataclass

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HPECOutput:
    prototypes: torch.Tensor
    angle_matrix: torch.Tensor
    aperture: torch.Tensor
    energy_per_proto: torch.Tensor
    prototype_similarity: torch.Tensor
    prototype_distance_logits: torch.Tensor
    energy_matrix: torch.Tensor
    prediction: torch.Tensor
    probability: torch.Tensor
    prototype_assignment: torch.Tensor


def _as_label_index(labels):
    if labels.ndim > 1:
        if labels.shape[-1] == 1:
            labels = labels.reshape(-1)
        else:
            labels = torch.argmax(labels, dim=-1)
    return labels.long()


def hpec_aperture(prototypes, cone_k=0.1, eps=1e-7):
    """计算 HPEC cone aperture psi，返回形状 [K]。"""

    proto_norm = torch.linalg.norm(prototypes, dim=-1).clamp_min(eps)
    asin_input = cone_k * (1.0 - proto_norm.pow(2)) / proto_norm
    asin_input = asin_input.clamp(min=-1.0 + eps, max=1.0 - eps)
    return torch.asin(asin_input)


def hpec_angle(prototypes, points, eps=1e-7):
    """按 HPEC 参考公式计算 prototype 到样本点的共形角度 Xi。"""

    if points.ndim != 2:
        raise ValueError(f"Expected points with shape [B, D], got {tuple(points.shape)}.")
    if prototypes.ndim not in (2, 3):
        raise ValueError(
            f"Expected prototypes with shape [K, D] or [C, K, D], got {tuple(prototypes.shape)}."
        )
    if points.shape[-1] != prototypes.shape[-1]:
        raise ValueError(
            f"Prototype dim {prototypes.shape[-1]} does not match point dim {points.shape[-1]}."
        )

    original_shape = prototypes.shape[:-1]
    prototypes_flat = prototypes.reshape(-1, prototypes.shape[-1])
    point_expanded = points[:, None, :]
    proto_expanded = prototypes_flat[None, :, :]
    dot = (point_expanded * proto_expanded).sum(dim=-1)
    point_norm_sq = points.pow(2).sum(dim=-1, keepdim=True)
    proto_norm = torch.linalg.norm(prototypes_flat, dim=-1).clamp_min(eps)
    proto_norm_sq = proto_norm.pow(2)[None, :]

    numerator = dot * (1.0 + proto_norm_sq) - proto_norm_sq * (1.0 + point_norm_sq)
    distance = torch.linalg.norm(point_expanded - proto_expanded, dim=-1).clamp_min(eps)
    sqrt_term = (1.0 + proto_norm_sq * point_norm_sq - 2.0 * dot).clamp_min(eps).sqrt()
    denominator = proto_norm[None, :] * distance * sqrt_term
    cos_angle = (numerator / denominator.clamp_min(eps)).clamp(min=-1.0 + eps, max=1.0 - eps)
    angle = torch.acos(cos_angle)
    return angle.reshape(points.shape[0], *original_shape)


def hpec_energy_loss(energy_matrix, labels, margin=1.0):
    """HPEC energy loss：真实类能量低，非真实类能量至少高于 margin。"""

    label_index = _as_label_index(labels)
    if energy_matrix.ndim != 2:
        raise ValueError(f"Expected energy_matrix [B, K], got {tuple(energy_matrix.shape)}.")
    if label_index.shape[0] != energy_matrix.shape[0]:
        raise ValueError("Label batch size does not match energy matrix batch size.")

    batch_index = torch.arange(energy_matrix.shape[0], device=energy_matrix.device)
    positive = energy_matrix[batch_index, label_index]
    mask = torch.ones_like(energy_matrix, dtype=torch.bool)
    mask[batch_index, label_index] = False
    negative = energy_matrix[mask].reshape(energy_matrix.shape[0], -1)
    negative = F.relu(float(margin) - negative).sum(dim=1)
    return (positive + negative).mean()


def hpec_energy_ce_loss(energy_matrix, labels, margin=0.0):
    """把负能量作为 logits，并可对真实类别施加能量间隔。"""

    label_index = _as_label_index(labels)
    logits = -energy_matrix
    margin = max(float(margin), 0.0)
    if margin > 0:
        logits = logits.clone()
        batch_index = torch.arange(logits.shape[0], device=logits.device)
        logits[batch_index, label_index] = logits[batch_index, label_index] - margin
    return F.cross_entropy(logits, label_index)


def poincare_distance_matrix(manifold, points, prototypes):
    """计算样本点到每个 prototype 的 Poincare distance，返回 [B, C, K]。"""

    points_expanded = points[:, None, None, :]
    prototypes_expanded = prototypes[None, :, :, :]
    return manifold.dist(points_expanded, prototypes_expanded, dim=-1)


def busemann_score_matrix(manifold, points, prototypes, eps=1e-7):
    """计算 ideal prototype 的 Busemann score；分数越大表示样本越接近该边界原型。"""

    if points.ndim != 2:
        raise ValueError(f"Expected points with shape [B, D], got {tuple(points.shape)}.")
    if prototypes.ndim not in (2, 3):
        raise ValueError(
            f"Expected prototypes with shape [K, D] or [C, K, D], got {tuple(prototypes.shape)}."
        )
    if points.shape[-1] != prototypes.shape[-1]:
        raise ValueError(
            f"Prototype dim {prototypes.shape[-1]} does not match point dim {points.shape[-1]}."
        )

    prototype_tangent = manifold.logmap0(prototypes, dim=-1)
    prototype_direction = F.normalize(prototype_tangent, p=2, dim=-1)
    point_norm_sq = points.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps).unsqueeze(-1)
    point_expanded = points[:, None, None, :]
    direction_expanded = prototype_direction[None, :, :, :]
    diff_sq = (point_expanded - direction_expanded).pow(2).sum(dim=-1).clamp_min(eps)
    numerator = (1.0 - point_norm_sq).clamp_min(eps)
    return torch.log(numerator / diff_sq)


def predict_hpec(energy_matrix):
    prediction = torch.argmin(energy_matrix, dim=-1)
    probability = torch.softmax(-energy_matrix, dim=-1)
    return prediction, probability


def balanced_sinkhorn_assignment(scores, epsilon=0.05, iterations=3, eps=1e-8):
    """用 Sinkhorn-Knopp 得到 batch 内尽量均衡的 prototype 分配矩阵。"""

    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N, K], got {tuple(scores.shape)}.")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return torch.zeros_like(scores)
    temperature = max(float(epsilon), eps)
    q = torch.exp((scores - scores.max()).detach() / temperature).clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    row_target = torch.full(
        (scores.shape[0],),
        1.0 / max(scores.shape[0], 1),
        device=scores.device,
        dtype=scores.dtype,
    )
    col_target = torch.full(
        (scores.shape[1],),
        1.0 / max(scores.shape[1], 1),
        device=scores.device,
        dtype=scores.dtype,
    )
    for _ in range(max(int(iterations), 1)):
        q = q * (row_target / q.sum(dim=1).clamp_min(eps))[:, None]
        q = q * (col_target / q.sum(dim=0).clamp_min(eps))[None, :]
    q = q * scores.shape[0]
    return q / q.sum(dim=1, keepdim=True).clamp_min(eps)


def _assignment_entropy(assignment, eps=1e-8):
    if assignment.numel() == 0:
        return assignment.sum() * 0.0
    prob = assignment.clamp_min(eps)
    entropy = -(prob * prob.log()).sum(dim=-1)
    normalizer = torch.log(
        torch.tensor(float(assignment.shape[-1]), device=assignment.device, dtype=assignment.dtype)
    )
    return entropy.mean() / normalizer.clamp_min(eps)


class HPECPrototypeEnergy(nn.Module):
    """模块 4 HPEC 原型能量层。

    原型先在欧氏单位球面上做 separation 初始化，再缩放到指定半径并投影进
    Poincare Ball。默认原型固定，便于先验证 HPEC loss 的分类效果。
    """

    def __init__(
        self,
        num_classes,
        embedding_dim,
        manifold=None,
        prototype_radius=0.3,
        cone_k=0.1,
        margin=1.0,
        prototypes_per_class=1,
        proto_temperature=0.2,
        trainable_prototypes=False,
        init_steps=500,
        distance_weight=0.0,
        energy_scale=1.0,
        energy_mode="cone",
        loss_mode="margin",
        energy_ce_margin=0.0,
        busemann_temperature=1.0,
        busemann_point_radius=0.0,
        busemann_radius_gate_weight=0.0,
        busemann_radius_gate_center=0.3,
        busemann_class_bias_weight=0.0,
        data_init=False,
        use_sinkhorn_ema=False,
        prototype_update_mode="reliable_tp_ema",
        reliable_confidence_threshold=0.70,
        reliable_view_consistency_threshold=0.55,
        reliable_min_samples=2,
        reliable_weight_floor=0.05,
        epoch_frechet_steps=3,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=3,
        ema_alpha=0.9,
        ema_anchor_weight=0.1,
        intra_class_max_cos=0.35,
        prototype_min_radius_ratio=0.6,
        prototype_max_radius_ratio=1.4,
        prototype_parameterization="poincare_point",
        eps=1e-7,
        seed=2024,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.prototypes_per_class = max(int(prototypes_per_class), 1)
        self.prototype_radius = float(prototype_radius)
        self.cone_k = float(cone_k)
        self.margin = float(margin)
        self.proto_temperature = float(proto_temperature)
        self.distance_weight = float(distance_weight)
        self.energy_scale = float(energy_scale)
        self.energy_mode = str(energy_mode or "cone").lower()
        self.loss_mode = str(loss_mode or "margin").lower()
        self.energy_ce_margin = max(float(energy_ce_margin), 0.0)
        self.busemann_temperature = float(busemann_temperature)
        self.busemann_point_radius = max(float(busemann_point_radius), 0.0)
        self.busemann_radius_gate_weight = max(float(busemann_radius_gate_weight), 0.0)
        self.busemann_radius_gate_center = max(float(busemann_radius_gate_center), 0.0)
        self.busemann_class_bias_weight = max(float(busemann_class_bias_weight), 0.0)
        self.data_init = bool(data_init)
        self.use_sinkhorn_ema = bool(use_sinkhorn_ema)
        self.prototype_update_mode = str(prototype_update_mode or "reliable_tp_ema").lower()
        if self.prototype_update_mode not in (
            "reliable_tp_ema",
            "epoch_reliable_frechet_ema",
            "sinkhorn_ema",
            "none",
        ):
            raise ValueError(
                "prototype_update_mode must be 'reliable_tp_ema', "
                "'epoch_reliable_frechet_ema', 'sinkhorn_ema' or 'none'."
            )
        self.reliable_confidence_threshold = min(
            max(float(reliable_confidence_threshold), 0.0), 1.0
        )
        self.reliable_view_consistency_threshold = min(
            max(float(reliable_view_consistency_threshold), -1.0), 1.0
        )
        self.reliable_min_samples = max(int(reliable_min_samples), 1)
        self.reliable_weight_floor = min(max(float(reliable_weight_floor), 0.0), 1.0)
        self.epoch_frechet_steps = max(int(epoch_frechet_steps), 1)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.ema_alpha = float(ema_alpha)
        self.ema_anchor_weight = float(ema_anchor_weight)
        self.intra_class_max_cos = float(intra_class_max_cos)
        self.prototype_min_radius_ratio = max(float(prototype_min_radius_ratio), 0.0)
        self.prototype_max_radius_ratio = max(
            float(prototype_max_radius_ratio),
            self.prototype_min_radius_ratio + 1e-6,
        )
        self.prototype_parameterization = str(prototype_parameterization or "poincare_point").lower()
        if self.prototype_parameterization not in ("poincare_point", "tangent_direction"):
            raise ValueError(
                f"Unsupported prototype_parameterization={self.prototype_parameterization!r}. "
                "Use 'poincare_point' or 'tangent_direction'."
            )
        self.eps = float(eps)
        self.manifold = manifold or geoopt.PoincareBall(c=1.0)
        self._data_initialized = False

        prototypes_tangent = self._init_hyperspherical_prototypes(
            init_steps=max(int(init_steps), 0),
            seed=int(seed),
        )
        prototypes_tangent = prototypes_tangent.reshape(
            self.num_classes,
            self.prototypes_per_class,
            self.embedding_dim,
        )
        prototypes_tangent = prototypes_tangent * self.prototype_radius
        self.register_buffer("prototype_anchor_tangent", prototypes_tangent.clone())
        prototypes = self.manifold.expmap0(prototypes_tangent, dim=-1, project=True)
        prototypes = self.manifold.projx(prototypes, dim=-1)
        prototype_requires_grad = bool(trainable_prototypes) and self.prototype_update_mode == "sinkhorn_ema"
        self.prototypes = nn.Parameter(prototypes, requires_grad=prototype_requires_grad)
        self.prototype_tangent_direction = nn.Parameter(
            F.normalize(prototypes_tangent.clone(), p=2, dim=-1),
            requires_grad=prototype_requires_grad,
        )
        self.busemann_class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.latest_sinkhorn_stats = {}
        self.latest_prototype_update_stats = {}
        self._epoch_prototype_queue = []

    def _direction_prototypes(self, dtype=None):
        direction = F.normalize(self.prototype_tangent_direction, p=2, dim=-1)
        if dtype is not None:
            direction = direction.to(dtype=dtype)
        radius = torch.as_tensor(
            self.prototype_radius,
            device=direction.device,
            dtype=direction.dtype,
        )
        tangent = direction * radius
        prototypes = self.manifold.expmap0(tangent, dim=-1, project=True)
        return self.manifold.projx(prototypes, dim=-1)

    def _current_prototypes(self, dtype=None):
        if self.prototype_parameterization == "tangent_direction":
            return self._direction_prototypes(dtype=dtype)
        prototypes = self.manifold.projx(self.prototypes, dim=-1)
        if dtype is not None:
            prototypes = prototypes.to(dtype=dtype)
        return prototypes

    def _prototype_radius_bounds(self, tangent):
        lower = max(self.prototype_radius * self.prototype_min_radius_ratio, self.eps)
        upper = max(self.prototype_radius * self.prototype_max_radius_ratio, lower + self.eps)
        lower_tensor = torch.as_tensor(lower, device=tangent.device, dtype=tangent.dtype)
        upper_tensor = torch.as_tensor(upper, device=tangent.device, dtype=tangent.dtype)
        return lower_tensor, upper_tensor

    def _project_tangent_to_radius_shell(self, tangent):
        """把 prototype 限制在固定半径壳层，避免 HPEC 原型塌到 Poincare 原点。"""

        anchor = self.prototype_anchor_tangent.to(device=tangent.device, dtype=tangent.dtype)
        norm = tangent.norm(dim=-1, keepdim=True)
        anchor_direction = F.normalize(anchor, p=2, dim=-1)
        direction = torch.where(norm > self.eps, tangent / norm.clamp_min(self.eps), anchor_direction)
        lower, upper = self._prototype_radius_bounds(tangent)
        shell_norm = torch.minimum(torch.maximum(norm, lower), upper)
        return direction * shell_norm

    def project_prototypes_to_radius_shell(self):
        """原地投影 trainable prototype，保持半径可解释且方向仍可学习。"""

        if self.prototype_parameterization == "tangent_direction":
            with torch.no_grad():
                self.prototype_tangent_direction.data.copy_(
                    F.normalize(self.prototype_tangent_direction.data, p=2, dim=-1)
                )
            return
        with torch.no_grad():
            prototypes = self.manifold.projx(self.prototypes.detach(), dim=-1)
            tangent = self.manifold.logmap0(prototypes, dim=-1)
            tangent = self._project_tangent_to_radius_shell(tangent)
            projected = self.manifold.expmap0(tangent.to(dtype=self.prototypes.dtype), dim=-1, project=True)
            self.prototypes.data.copy_(self.manifold.projx(projected, dim=-1))

    def maybe_initialize_from_batch(self, z_global, labels):
        """用训练 batch 的类中心 warm-start prototype。

        HPEC 参考实现使用 hyperspherical prototype 作为先验；在 fMRI 小样本场景中，
        随机球面方向可能和 HGCN 早期 embedding 偏差较大。因此这里参考
        Prototypical Networks 的“类别中心”思想，只在首次训练 batch 上做一次
        数据驱动初始化：每类取 logmap0(z) 的均值方向，再保留少量球面初始化偏移。
        """

        if self._data_initialized or not self.data_init:
            return
        if self.prototype_update_mode in (
            "sinkhorn_ema",
            "reliable_tp_ema",
            "epoch_reliable_frechet_ema",
        ) and self.prototypes_per_class > 1:
            # 多原型 EMA 依赖初始原型的方向差异；首个 batch 的类中心 warm-start
            # 容易把同类多个 prototype 先压到一处，削弱 Sinkhorn 的均衡分配效果。
            self._data_initialized = True
            return
        label_index = _as_label_index(labels)
        if z_global.shape[0] != label_index.shape[0]:
            return

        with torch.no_grad():
            z_tangent = self.manifold.logmap0(self.manifold.projx(z_global.detach(), dim=-1), dim=-1)
            current_prototypes = self._current_prototypes(dtype=z_global.dtype).detach()
            current_tangent = self.manifold.logmap0(self.manifold.projx(current_prototypes, dim=-1), dim=-1)
            init_tangent = current_tangent.clone()
            for class_idx in range(self.num_classes):
                class_mask = label_index == class_idx
                if not torch.any(class_mask):
                    continue
                class_center = z_tangent[class_mask].mean(dim=0, keepdim=True)
                center_norm = class_center.norm(dim=-1, keepdim=True)
                if torch.all(center_norm <= self.eps):
                    continue
                class_direction = F.normalize(class_center, p=2, dim=-1)
                offsets = current_tangent[class_idx]
                offsets = offsets - offsets.mean(dim=0, keepdim=True)
                offset_norm = offsets.norm(dim=-1, keepdim=True)
                offsets = torch.where(
                    offset_norm > self.eps,
                    offsets / offset_norm.clamp_min(self.eps),
                    offsets,
                )
                # 以类别中心为主、球面偏移为辅，避免每类多个 prototype 完全重合。
                mixed_direction = F.normalize(class_direction + 0.15 * offsets, p=2, dim=-1)
                # 半径跟随当前 embedding 的类中心范数，而不是固定在较远的 prototype_radius。
                # 这样 prototype 初始位置更贴近 HGCN 实际输出，后续 HPEC energy 才有可解释的吸引方向。
                lower, upper = self._prototype_radius_bounds(center_norm)
                data_radius = torch.minimum(torch.maximum(center_norm, lower), upper)
                init_tangent[class_idx] = mixed_direction * data_radius

            init_tangent = self._project_tangent_to_radius_shell(init_tangent)
            if self.prototype_parameterization == "tangent_direction":
                self.prototype_tangent_direction.data.copy_(
                    F.normalize(init_tangent, p=2, dim=-1).to(self.prototype_tangent_direction.dtype)
                )
            else:
                initialized = self.manifold.expmap0(init_tangent.to(device=z_global.device, dtype=z_global.dtype), dim=-1, project=True)
                self.prototypes.data.copy_(self.manifold.projx(initialized, dim=-1).to(self.prototypes.dtype))
            self._data_initialized = True

    def _copy_prototypes_from_tangent(self, tangent):
        """将切空间 prototype 写回当前参数化，并保持 Poincare 有效区域。"""

        tangent = self._project_tangent_to_radius_shell(tangent)
        if self.prototype_parameterization == "tangent_direction":
            self.prototype_tangent_direction.data.copy_(
                F.normalize(tangent, p=2, dim=-1).to(self.prototype_tangent_direction.dtype)
            )
            return
        updated = self.manifold.expmap0(
            tangent.to(device=self.prototypes.device, dtype=self.prototypes.dtype),
            dim=-1,
            project=True,
        )
        self.prototypes.data.copy_(self.manifold.projx(updated, dim=-1).to(self.prototypes.dtype))

    def update_prototypes_with_reliable_tp_ema(
        self,
        z_global,
        labels,
        logits,
        companion_z_global=None,
        energy_per_proto=None,
    ):
        """用可靠 TP 训练样本独立 EMA 移动多 prototype，不参与 autograd。"""

        if self.prototype_update_mode != "reliable_tp_ema":
            return {}
        label_index = _as_label_index(labels)
        if z_global.shape[0] != label_index.shape[0] or logits.shape[0] != label_index.shape[0]:
            return {}

        with torch.no_grad():
            z_global = self.manifold.projx(z_global.detach(), dim=-1)
            z_tangent = self.manifold.logmap0(z_global, dim=-1)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                positive_probability = torch.sigmoid(logits.reshape(-1))
                probability = torch.stack([1.0 - positive_probability, positive_probability], dim=-1)
            else:
                probability = torch.softmax(logits.detach(), dim=-1)
            prediction = probability.argmax(dim=-1)
            confidence = probability.gather(1, label_index[:, None]).squeeze(1)
            reliable = (prediction == label_index) & (confidence >= self.reliable_confidence_threshold)

            if companion_z_global is None:
                consistency = torch.ones_like(confidence)
            else:
                companion = self.manifold.projx(companion_z_global.detach(), dim=-1)
                companion_tangent = self.manifold.logmap0(companion, dim=-1)
                consistency = (
                    1.0
                    + F.cosine_similarity(z_tangent, companion_tangent, dim=-1, eps=self.eps)
                ) / 2.0
                reliable = reliable & (consistency >= self.reliable_view_consistency_threshold)

            current_prototypes = self._current_prototypes(dtype=z_global.dtype).detach()
            prototype_tangent = self.manifold.logmap0(
                self.manifold.projx(current_prototypes, dim=-1), dim=-1
            )
            if energy_per_proto is None:
                energy_per_proto = self(z_global).energy_per_proto.detach()
            else:
                energy_per_proto = energy_per_proto.detach()
            if energy_per_proto.shape[:2] != (z_global.shape[0], self.num_classes):
                raise ValueError("energy_per_proto must have shape [B, classes, prototypes_per_class].")

            new_tangent = prototype_tangent.clone()
            anchor = self.prototype_anchor_tangent.to(device=z_global.device, dtype=z_global.dtype)
            alpha = min(max(float(self.ema_alpha), 0.0), 0.9999)
            anchor_weight = min(max(float(self.ema_anchor_weight), 0.0), 1.0)
            assignment_counts = torch.zeros(
                self.num_classes,
                self.prototypes_per_class,
                device=z_global.device,
                dtype=z_global.dtype,
            )
            updated_mask = torch.zeros_like(assignment_counts, dtype=torch.bool)
            movement = torch.zeros_like(assignment_counts)
            class_reliable_counts = torch.zeros(self.num_classes, device=z_global.device, dtype=z_global.dtype)

            for class_idx in range(self.num_classes):
                class_mask = reliable & (label_index == class_idx)
                class_reliable_counts[class_idx] = class_mask.sum()
                if not torch.any(class_mask):
                    continue
                class_energy = energy_per_proto[class_mask, class_idx, :]
                winners = class_energy.argmin(dim=-1)
                for prototype_idx in range(self.prototypes_per_class):
                    winner_mask = winners == prototype_idx
                    count = winner_mask.sum()
                    assignment_counts[class_idx, prototype_idx] = count
                    if int(count.item()) < self.reliable_min_samples:
                        continue
                    selected_indices = torch.where(class_mask)[0][winner_mask]
                    weights = consistency[selected_indices].clamp_min(self.eps)
                    center = (weights[:, None] * z_tangent[selected_indices]).sum(dim=0)
                    center = center / weights.sum().clamp_min(self.eps)
                    center_norm = center.norm().clamp_min(self.eps)
                    lower, upper = self._prototype_radius_bounds(center.view(1, -1))
                    center = center / center_norm * center_norm.clamp(min=lower.item(), max=upper.item())
                    target = (1.0 - anchor_weight) * center + anchor_weight * anchor[class_idx, prototype_idx]
                    candidate = alpha * prototype_tangent[class_idx, prototype_idx] + (1.0 - alpha) * target
                    new_tangent[class_idx, prototype_idx] = candidate
                    updated_mask[class_idx, prototype_idx] = True
                    movement[class_idx, prototype_idx] = (candidate - prototype_tangent[class_idx, prototype_idx]).norm()

            self._copy_prototypes_from_tangent(new_tangent)
            total_assignments = assignment_counts.sum()
            assignment_prob = assignment_counts / total_assignments.clamp_min(self.eps)
            assignment_entropy = -(
                assignment_prob[assignment_counts > 0]
                * assignment_prob[assignment_counts > 0].clamp_min(self.eps).log()
            ).sum()
            normalizer = torch.log(
                torch.as_tensor(
                    float(max(self.num_classes * self.prototypes_per_class, 2)),
                    device=z_global.device,
                    dtype=z_global.dtype,
                )
            )
            zero = z_global.sum() * 0.0
            stats = {
                "hpec_reliable_tp_ratio": reliable.to(z_global.dtype).mean(),
                "hpec_reliable_confidence_mean": confidence.mean(),
                "hpec_reliable_view_consistency_mean": consistency.mean(),
                "hpec_reliable_class_count_min": class_reliable_counts.min(),
                "hpec_reliable_class_count_max": class_reliable_counts.max(),
                "hpec_reliable_assignment_entropy": assignment_entropy / normalizer.clamp_min(self.eps),
                "hpec_reliable_assignment_count_min": assignment_counts.min(),
                "hpec_reliable_assignment_count_max": assignment_counts.max(),
                "hpec_reliable_updated_prototype_count": updated_mask.to(z_global.dtype).sum(),
                "hpec_reliable_unupdated_prototype_count": (~updated_mask).to(z_global.dtype).sum(),
                "hpec_reliable_ema_displacement_mean": movement[updated_mask].mean()
                if torch.any(updated_mask)
                else zero,
            }
            self.latest_prototype_update_stats = stats
            return stats

    def clear_epoch_prototype_queue(self):
        """清空 epoch 级 prototype 样本缓存，不改变 prototype 本身。"""

        self._epoch_prototype_queue = []

    def queue_epoch_prototype_samples(
        self,
        z_global,
        labels,
        logits,
        companion_z_global=None,
    ):
        """缓存训练样本，供 epoch 结束后独立更新 prototype。

        可靠度使用连续权重而不是 TP 硬门控，避免早期误分类样本永远无法修正原型。
        缓存张量转移到 CPU，既不保留 autograd 图，也不长期占用显存。
        """

        if self.prototype_update_mode != "epoch_reliable_frechet_ema":
            return {}
        label_index = _as_label_index(labels)
        if z_global.shape[0] != label_index.shape[0] or logits.shape[0] != label_index.shape[0]:
            return {}

        with torch.no_grad():
            points = self.manifold.projx(z_global.detach(), dim=-1)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                positive_probability = torch.sigmoid(logits.detach().reshape(-1))
                probability = torch.stack([1.0 - positive_probability, positive_probability], dim=-1)
            else:
                probability = torch.softmax(logits.detach(), dim=-1)
            confidence = probability.gather(1, label_index[:, None]).squeeze(1)

            if companion_z_global is None:
                consistency = torch.ones_like(confidence)
            else:
                companion = self.manifold.projx(companion_z_global.detach(), dim=-1)
                tangent = self.manifold.logmap0(points, dim=-1)
                companion_tangent = self.manifold.logmap0(companion, dim=-1)
                consistency = (
                    1.0
                    + F.cosine_similarity(tangent, companion_tangent, dim=-1, eps=self.eps)
                ) / 2.0

            threshold_scale = max(1.0 - self.reliable_confidence_threshold, 0.05)
            confidence_gate = torch.sigmoid(
                (confidence - self.reliable_confidence_threshold) / threshold_scale
            )
            reliability = self.reliable_weight_floor + (
                1.0 - self.reliable_weight_floor
            ) * confidence_gate * consistency.clamp(0.0, 1.0)
            self._epoch_prototype_queue.append(
                (
                    points.to(device="cpu", dtype=torch.float32),
                    label_index.detach().to(device="cpu"),
                    reliability.to(device="cpu", dtype=torch.float32),
                    confidence.to(device="cpu", dtype=torch.float32),
                    consistency.to(device="cpu", dtype=torch.float32),
                )
            )
            queued = sum(item[0].shape[0] for item in self._epoch_prototype_queue)
            return {
                "hpec_epoch_queue_size": torch.as_tensor(
                    float(queued), device=z_global.device, dtype=z_global.dtype
                ),
                "hpec_epoch_reliability_mean": reliability.mean(),
            }

    def _weighted_frechet_mean(self, points, weights, initial):
        """在 Poincare Ball 上执行少量 Karcher 迭代。"""

        mean = self.manifold.projx(initial.detach(), dim=-1)
        weights = weights.to(device=points.device, dtype=points.dtype).clamp_min(self.eps)
        weights = weights / weights.sum().clamp_min(self.eps)
        for _ in range(self.epoch_frechet_steps):
            base = mean.unsqueeze(0).expand_as(points)
            tangent = self.manifold.logmap(base, points, dim=-1)
            update = (weights[:, None] * tangent).sum(dim=0)
            mean = self.manifold.expmap(mean, update, dim=-1, project=True)
            mean = self.manifold.projx(mean, dim=-1)
        return mean

    def finalize_epoch_prototype_update(self):
        """用整个训练 epoch 的样本执行流形 soft-assignment EMA。"""

        if self.prototype_update_mode != "epoch_reliable_frechet_ema":
            return {}
        if not self._epoch_prototype_queue:
            return {}

        with torch.no_grad():
            device = self.prototypes.device
            dtype = self.prototypes.dtype
            points = torch.cat([item[0] for item in self._epoch_prototype_queue], dim=0).to(
                device=device, dtype=dtype
            )
            labels = torch.cat([item[1] for item in self._epoch_prototype_queue], dim=0).to(device)
            reliability = torch.cat([item[2] for item in self._epoch_prototype_queue], dim=0).to(
                device=device, dtype=dtype
            )
            confidence = torch.cat([item[3] for item in self._epoch_prototype_queue], dim=0).to(
                device=device, dtype=dtype
            )
            consistency = torch.cat([item[4] for item in self._epoch_prototype_queue], dim=0).to(
                device=device, dtype=dtype
            )
            self.clear_epoch_prototype_queue()

            current = self._current_prototypes(dtype=dtype).detach()
            updated = current.clone()
            anchor = self.manifold.expmap0(
                self.prototype_anchor_tangent.to(device=device, dtype=dtype),
                dim=-1,
                project=True,
            )
            alpha = min(max(float(self.ema_alpha), 0.0), 0.9999)
            anchor_weight = min(max(float(self.ema_anchor_weight), 0.0), 1.0)
            occupancy = torch.zeros(
                self.num_classes,
                self.prototypes_per_class,
                device=device,
                dtype=dtype,
            )
            movement = torch.zeros_like(occupancy)
            updated_mask = torch.zeros_like(occupancy, dtype=torch.bool)

            all_energy = self(points).energy_per_proto.detach()
            temperature = max(float(self.proto_temperature), 1e-4)
            for class_idx in range(self.num_classes):
                class_mask = labels == class_idx
                if int(class_mask.sum().item()) < self.reliable_min_samples:
                    continue
                class_points = points[class_mask]
                class_reliability = reliability[class_mask]
                assignment = torch.softmax(
                    -all_energy[class_mask, class_idx, :] / temperature,
                    dim=-1,
                )
                weighted_assignment = assignment * class_reliability[:, None]
                occupancy[class_idx] = weighted_assignment.sum(dim=0)

                for prototype_idx in range(self.prototypes_per_class):
                    weights = weighted_assignment[:, prototype_idx]
                    if weights.sum() < self.eps:
                        continue
                    target = self._weighted_frechet_mean(
                        class_points,
                        weights,
                        current[class_idx, prototype_idx],
                    )
                    if anchor_weight > 0:
                        to_anchor = self.manifold.logmap(
                            target,
                            anchor[class_idx, prototype_idx],
                            dim=-1,
                        )
                        target = self.manifold.expmap(
                            target,
                            anchor_weight * to_anchor,
                            dim=-1,
                            project=True,
                        )
                    direction = self.manifold.logmap(
                        current[class_idx, prototype_idx],
                        target,
                        dim=-1,
                    )
                    candidate = self.manifold.expmap(
                        current[class_idx, prototype_idx],
                        (1.0 - alpha) * direction,
                        dim=-1,
                        project=True,
                    )
                    updated[class_idx, prototype_idx] = self.manifold.projx(candidate, dim=-1)
                    movement[class_idx, prototype_idx] = self.manifold.dist(
                        current[class_idx, prototype_idx],
                        updated[class_idx, prototype_idx],
                        dim=-1,
                    )
                    updated_mask[class_idx, prototype_idx] = True

            updated_tangent = self.manifold.logmap0(updated, dim=-1)
            self._copy_prototypes_from_tangent(updated_tangent)
            occupancy_prob = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            occupancy_entropy = -(
                occupancy_prob.clamp_min(self.eps) * occupancy_prob.clamp_min(self.eps).log()
            ).sum(dim=-1)
            entropy_norm = torch.log(
                torch.as_tensor(
                    float(max(self.prototypes_per_class, 2)),
                    device=device,
                    dtype=dtype,
                )
            )
            stats = {
                "hpec_epoch_sample_count": torch.as_tensor(float(points.shape[0]), device=device, dtype=dtype),
                "hpec_epoch_confidence_mean": confidence.mean(),
                "hpec_epoch_consistency_mean": consistency.mean(),
                "hpec_epoch_reliability_mean": reliability.mean(),
                "hpec_epoch_occupancy_min": occupancy.min(),
                "hpec_epoch_occupancy_max": occupancy.max(),
                "hpec_epoch_occupancy_entropy": (occupancy_entropy / entropy_norm.clamp_min(self.eps)).mean(),
                "hpec_epoch_updated_prototype_count": updated_mask.to(dtype).sum(),
                "hpec_epoch_prototype_movement_mean": movement[updated_mask].mean()
                if torch.any(updated_mask)
                else movement.sum() * 0.0,
                "hpec_epoch_prototype_movement_max": movement.max(),
            }
            self.latest_prototype_update_stats = stats
            return stats

    def update_prototypes_with_sinkhorn_ema(self, z_global, labels):
        """按论文式多原型思路，用 Sinkhorn 均衡分配和 EMA 更新每类 prototype。"""

        if not self.use_sinkhorn_ema or self.prototypes_per_class <= 1:
            return {}
        label_index = _as_label_index(labels)
        if z_global.shape[0] != label_index.shape[0]:
            return {}

        with torch.no_grad():
            z_global = self.manifold.projx(z_global.detach(), dim=-1)
            z_tangent = self.manifold.logmap0(z_global, dim=-1)
            current_prototypes = self._current_prototypes(dtype=z_global.dtype).detach()
            proto_tangent = self.manifold.logmap0(self.manifold.projx(current_prototypes, dim=-1), dim=-1)
            z_norm = F.normalize(z_tangent, p=2, dim=-1)
            proto_norm = F.normalize(proto_tangent, p=2, dim=-1)
            anchor_tangent = self.prototype_anchor_tangent.to(device=z_global.device, dtype=z_global.dtype)
            new_tangent = proto_tangent.clone()
            entropy_values = []
            usage_values = []
            alpha = min(max(float(self.ema_alpha), 0.0), 0.9999)
            anchor_weight = min(max(float(self.ema_anchor_weight), 0.0), 1.0)

            for class_idx in range(self.num_classes):
                class_mask = label_index == class_idx
                if not torch.any(class_mask):
                    continue
                scores = torch.matmul(z_norm[class_mask], proto_norm[class_idx].T)
                assignment = balanced_sinkhorn_assignment(
                    scores,
                    epsilon=self.sinkhorn_epsilon,
                    iterations=self.sinkhorn_iters,
                    eps=self.eps,
                )
                weight_sum = assignment.sum(dim=0).clamp_min(self.eps)
                assigned_center = torch.matmul(assignment.T, z_tangent[class_mask]) / weight_sum[:, None]
                center_norm = assigned_center.norm(dim=-1, keepdim=True)
                fallback = proto_tangent[class_idx]
                assigned_center = torch.where(center_norm > self.eps, assigned_center, fallback)
                assigned_direction = F.normalize(assigned_center, p=2, dim=-1)
                lower, upper = self._prototype_radius_bounds(center_norm)
                assigned_radius = torch.minimum(torch.maximum(center_norm, lower), upper)
                assigned_target = assigned_direction * assigned_radius
                if anchor_weight > 0:
                    anchor_direction = F.normalize(anchor_tangent[class_idx], p=2, dim=-1)
                    anchor_target = anchor_direction * assigned_radius
                    assigned_target = (1.0 - anchor_weight) * assigned_target + anchor_weight * anchor_target
                new_tangent[class_idx] = alpha * proto_tangent[class_idx] + (1.0 - alpha) * assigned_target
                entropy_values.append(_assignment_entropy(assignment))
                usage_values.append(weight_sum)

            if self.prototypes_per_class > 1:
                proto_direction = F.normalize(new_tangent, p=2, dim=-1)
                for class_idx in range(self.num_classes):
                    class_direction = proto_direction[class_idx]
                    class_cosine = torch.matmul(class_direction, class_direction.T)
                    eye = torch.eye(
                        self.prototypes_per_class,
                        device=class_cosine.device,
                        dtype=torch.bool,
                    )
                    repel = F.relu(class_cosine - float(self.intra_class_max_cos)).masked_fill(eye, 0.0)
                    if torch.any(repel > 0):
                        push = torch.matmul(repel, class_direction)
                        new_tangent[class_idx] = new_tangent[class_idx] - 0.12 * push
                if anchor_weight > 0:
                    new_tangent = (
                        (1.0 - 0.12 * anchor_weight) * new_tangent
                        + (0.12 * anchor_weight) * anchor_tangent
                    )

            self._copy_prototypes_from_tangent(new_tangent)
            if usage_values:
                usage = torch.cat(usage_values)
                stats = {
                    "hpec_sinkhorn_assignment_entropy": torch.stack(entropy_values).mean(),
                    "hpec_sinkhorn_usage_min": usage.min(),
                    "hpec_sinkhorn_usage_max": usage.max(),
                }
            else:
                zero = self.prototypes.sum() * 0.0
                stats = {
                    "hpec_sinkhorn_assignment_entropy": zero,
                    "hpec_sinkhorn_usage_min": zero,
                    "hpec_sinkhorn_usage_max": zero,
                }
            self.latest_sinkhorn_stats = stats
            return stats

    def _init_hyperspherical_prototypes(self, init_steps, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        mapping = torch.randn(
            self.num_classes * self.prototypes_per_class,
            self.embedding_dim,
            generator=generator,
            dtype=torch.float32,
        )
        mapping = F.normalize(mapping, p=2, dim=-1)
        if init_steps <= 0 or mapping.shape[0] <= 1:
            return mapping

        mapping = mapping.requires_grad_(True)
        optimizer = torch.optim.SGD([mapping], lr=0.1, momentum=0.9, weight_decay=1e-4)
        eye = torch.eye(mapping.shape[0], dtype=mapping.dtype)
        for _ in range(init_steps):
            with torch.no_grad():
                mapping.copy_(F.normalize(mapping, p=2, dim=-1))
            optimizer.zero_grad()
            cosine = mapping @ mapping.t()
            # 最大化类别之间的角间隔：最相近的非自身 prototype 越小越好。
            separation_loss = (cosine - 2.0 * eye).max(dim=1)[0].mean()
            separation_loss.backward()
            optimizer.step()
        with torch.no_grad():
            mapping.copy_(F.normalize(mapping, p=2, dim=-1))
        return mapping.detach()

    def forward(self, z_global):
        self.project_prototypes_to_radius_shell()
        prototypes = self._current_prototypes(dtype=z_global.dtype)
        z_global = self.manifold.projx(z_global, dim=-1)
        angle_matrix = hpec_angle(prototypes, z_global, eps=self.eps)
        aperture = hpec_aperture(prototypes, cone_k=self.cone_k, eps=self.eps)
        cone_violation = F.relu(angle_matrix - aperture[None, :, :])
        # HPEC 原式以 cone violation 为核心；Poincare distance 提供原型分类常用的径向可分信号。
        distance_matrix = poincare_distance_matrix(self.manifold, z_global, prototypes)
        energy_per_proto = cone_violation + self.distance_weight * distance_matrix
        temperature = max(self.proto_temperature, 1e-6)
        # 用 softmin 权重做加权平均，保持类别 energy 非负；直接 -tau*logsumexp 会在多 prototype
        # 同时接近 0 时得到负值，进而破坏 HPEC margin loss 的语义。
        softmin_weight = torch.softmax(-energy_per_proto / temperature, dim=-1)
        energy_matrix = (softmin_weight * energy_per_proto).sum(dim=-1)
        if self.energy_scale != 1.0:
            energy_matrix = energy_matrix * self.energy_scale
        distance_logits = -temperature * torch.logsumexp(
            -distance_matrix / temperature,
            dim=-1,
        )
        distance_logits = distance_logits - distance_logits.mean(dim=-1, keepdim=True)
        z_tangent = self.manifold.logmap0(z_global, dim=-1)
        prototype_tangent = self.manifold.logmap0(prototypes, dim=-1)
        prototype_similarity = torch.einsum(
            "bd,ckd->bck",
            F.normalize(z_tangent, p=2, dim=-1),
            F.normalize(prototype_tangent, p=2, dim=-1),
        )
        if self.energy_mode == "busemann":
            busemann_points = z_global
            if self.busemann_point_radius > 0:
                # Busemann 原型本质上描述“朝向哪个理想边界方向”。小样本 fMRI 中，样本半径
                # 过早变成置信度容易过拟合训练集；可选固定半径只保留方向证据。
                z_tangent_for_score = self.manifold.logmap0(z_global, dim=-1)
                z_direction = F.normalize(z_tangent_for_score, p=2, dim=-1)
                target_radius = torch.as_tensor(
                    self.busemann_point_radius,
                    device=z_direction.device,
                    dtype=z_direction.dtype,
                )
                score_tangent = z_direction * target_radius
                busemann_points = self.manifold.expmap0(score_tangent, dim=-1, project=True)
                busemann_points = self.manifold.projx(busemann_points, dim=-1)
            busemann_scores = busemann_score_matrix(self.manifold, busemann_points, prototypes, eps=self.eps)
            if self.busemann_radius_gate_weight > 0:
                # Busemann 主要表达类别方向；样本半径保留为低自由度置信门控，而不是替代方向证据。
                # 这样避免固定半径丢失层级深度信息，也避免新增额外损失项。
                z_radius = z_global.norm(dim=-1, keepdim=True).unsqueeze(-1)
                gate = 1.0 + self.busemann_radius_gate_weight * (
                    z_radius - self.busemann_radius_gate_center
                )
                gate = gate.clamp(0.5, 1.5)
                busemann_scores = busemann_scores * gate
            if self.busemann_class_bias_weight > 0:
                # Busemann score 描述样本朝向每类理想边界方向的程度；每类一个可学习偏置
                # 对应 horosphere/分类边界的平移，用于小样本类别不均衡下的能量校准。
                class_bias = (
                    self.busemann_class_bias.to(device=busemann_scores.device, dtype=busemann_scores.dtype)
                    .view(1, self.num_classes, 1)
                )
                busemann_scores = busemann_scores + self.busemann_class_bias_weight * class_bias
            temperature = max(self.busemann_temperature, 1e-6)
            energy_per_proto = -busemann_scores
            energy_matrix = -temperature * torch.logsumexp(busemann_scores / temperature, dim=-1)
            if self.energy_scale != 1.0:
                energy_matrix = energy_matrix * self.energy_scale
            prototype_similarity = busemann_scores
        elif self.energy_mode != "cone":
            raise ValueError(
                f"Unsupported energy_mode={self.energy_mode!r}. Use 'cone' or 'busemann'."
            )
        prediction, probability = predict_hpec(energy_matrix)
        prototype_assignment = torch.argmin(energy_per_proto.reshape(energy_per_proto.shape[0], -1), dim=-1)
        return HPECOutput(
            prototypes=prototypes,
            angle_matrix=angle_matrix,
            aperture=aperture,
            energy_per_proto=energy_per_proto,
            prototype_similarity=prototype_similarity,
            prototype_distance_logits=distance_logits,
            energy_matrix=energy_matrix,
            prediction=prediction,
            probability=probability,
            prototype_assignment=prototype_assignment,
        )

    def loss(self, energy_matrix, labels):
        if self.loss_mode in ("ce", "cross_entropy", "energy_ce"):
            return hpec_energy_ce_loss(
                energy_matrix,
                labels,
                margin=self.energy_ce_margin,
            )
        if self.loss_mode != "margin":
            raise ValueError(
                f"Unsupported hpec loss_mode={self.loss_mode!r}. Use 'margin' or 'energy_ce'."
            )
        return hpec_energy_loss(energy_matrix, labels, margin=self.margin)
