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


def hpec_energy_ce_loss(energy_matrix, labels):
    """直接把负能量作为分类 logits，优化 HPEC 原型能量的类别边界。"""

    label_index = _as_label_index(labels)
    return F.cross_entropy(-energy_matrix, label_index)


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


def hpec_multi_prototype_losses(
    z_global,
    prototypes,
    labels,
    manifold,
    temperature=0.2,
    prototype_radius_target=0.3,
    use_sinkhorn=False,
    sinkhorn_epsilon=0.05,
    sinkhorn_iters=3,
    prototype_anchor_tangent=None,
    sample_margin=0.2,
    intra_class_max_cos=0.35,
    inter_class_max_cos=0.0,
):
    """计算多 prototype 相关损失，返回未加权的 L_mle/L_pcl/L_pal。"""

    label_index = _as_label_index(labels)
    tau = max(float(temperature), 1e-6)
    z_tangent = manifold.logmap0(z_global, dim=-1)
    prototype_tangent = manifold.logmap0(prototypes, dim=-1)
    z_norm = F.normalize(z_tangent, p=2, dim=-1)
    prototype_norm = F.normalize(prototype_tangent, p=2, dim=-1)

    similarity = torch.einsum("bd,ckd->bck", z_norm, prototype_norm)
    batch_index = torch.arange(z_norm.shape[0], device=z_norm.device)
    true_prototypes = prototype_norm[label_index]
    true_similarity = similarity[batch_index, label_index]
    if use_sinkhorn and prototype_norm.shape[1] > 1:
        true_assignment = torch.zeros_like(true_similarity)
        usage_values = []
        entropy_values = []
        for class_idx in range(prototype_norm.shape[0]):
            class_mask = label_index == class_idx
            if not torch.any(class_mask):
                continue
            class_assignment = balanced_sinkhorn_assignment(
                similarity[class_mask, class_idx],
                epsilon=sinkhorn_epsilon,
                iterations=sinkhorn_iters,
                eps=1e-8,
            )
            true_assignment[class_mask] = class_assignment
            usage_values.append(class_assignment.sum(dim=0))
            entropy_values.append(_assignment_entropy(class_assignment))
        positive_logit = torch.logsumexp(
            true_assignment.clamp_min(1e-8).log() + true_similarity / tau,
            dim=-1,
        )
        all_logits = torch.logsumexp((similarity / tau).reshape(similarity.shape[0], -1), dim=-1)
        mle_loss = -(positive_logit - all_logits).mean()
        assigned_prototype = torch.einsum("bk,bkd->bd", true_assignment, true_prototypes)
        pal_loss = (z_norm - assigned_prototype).pow(2).sum(dim=-1).mean()
        if usage_values:
            usage = torch.cat(usage_values)
            assignment_usage_min = usage.min()
            assignment_usage_max = usage.max()
            sinkhorn_assignment_entropy = torch.stack(entropy_values).mean()
        else:
            assignment_usage_min = true_similarity.sum() * 0.0
            assignment_usage_max = true_similarity.sum() * 0.0
            sinkhorn_assignment_entropy = true_similarity.sum() * 0.0
    else:
        class_logits = torch.logsumexp(similarity / tau, dim=-1)
        mle_loss = F.cross_entropy(class_logits, label_index)
        best_index = torch.argmax(true_similarity, dim=-1)
        best_prototype = true_prototypes[batch_index, best_index]
        pal_loss = (z_norm - best_prototype).pow(2).sum(dim=-1).mean()
        assignment_usage_min = true_similarity.sum() * 0.0
        assignment_usage_max = true_similarity.sum() * 0.0
        sinkhorn_assignment_entropy = true_similarity.sum() * 0.0

    flat_prototypes = prototype_norm.reshape(-1, prototype_norm.shape[-1])
    num_classes, prototypes_per_class = prototype_norm.shape[:2]
    proto_labels = torch.arange(num_classes, device=z_norm.device).repeat_interleave(prototypes_per_class)
    proto_cosine = torch.matmul(flat_prototypes, flat_prototypes.T)
    proto_similarity = proto_cosine / tau
    proto_eye = torch.eye(proto_similarity.shape[0], device=z_norm.device, dtype=torch.bool)
    positive_mask = proto_labels[:, None].eq(proto_labels[None, :]) & (~proto_eye)
    negative_mask = ~proto_labels[:, None].eq(proto_labels[None, :])
    true_soft = torch.logsumexp(true_similarity / tau, dim=-1) * tau
    false_mask = F.one_hot(label_index, num_classes=num_classes).bool().unsqueeze(-1)
    false_similarity = similarity.masked_fill(false_mask, -1e9)
    hardest_false = false_similarity.reshape(false_similarity.shape[0], -1).max(dim=-1)[0]
    pcl_loss = F.relu(hardest_false - true_soft + float(sample_margin)).mean()

    prototype_radius = torch.linalg.norm(prototypes, dim=-1)
    radius_target = float(prototype_radius_target)
    radius_loss = F.relu(prototype_radius - radius_target).pow(2).mean()
    if prototypes_per_class > 1:
        same_class_similarity = torch.einsum("ckd,cld->ckl", prototype_norm, prototype_norm)
        same_eye = torch.eye(prototypes_per_class, device=z_norm.device, dtype=torch.bool).unsqueeze(0)
        same_offdiag = same_class_similarity.masked_select(~same_eye.expand_as(same_class_similarity))
        diversity_loss = F.relu(same_offdiag - float(intra_class_max_cos)).pow(2).mean()
        intra_orthogonal_loss = same_offdiag.pow(2).mean()
        same_class_cos_max = same_offdiag.max() if same_offdiag.numel() else flat_prototypes.sum() * 0.0
    else:
        diversity_loss = flat_prototypes.sum() * 0.0
        intra_orthogonal_loss = flat_prototypes.sum() * 0.0
        same_class_cos_max = flat_prototypes.sum() * 0.0

    hsic_loss = prototype_hsic_loss(prototype_tangent)
    offdiag_mask = ~proto_eye
    offdiag_cos = torch.abs(proto_cosine)[offdiag_mask]
    prototype_cos_abs_mean = offdiag_cos.mean() if offdiag_cos.numel() else flat_prototypes.sum() * 0.0
    prototype_cos_abs_max = offdiag_cos.max() if offdiag_cos.numel() else flat_prototypes.sum() * 0.0

    inter_values = proto_cosine[negative_mask]
    # 异类 prototype 不必完全正交，但余弦相似超过 margin 时会侵蚀类别边界。
    inter_margin_loss = (
        F.relu(inter_values - float(inter_class_max_cos)).pow(2).mean()
        if inter_values.numel()
        else flat_prototypes.sum() * 0.0
    )
    class_direction = F.normalize(prototype_norm.mean(dim=1), p=2, dim=-1)
    if class_direction.shape[0] > 1:
        class_cosine = torch.matmul(class_direction, class_direction.T)
        class_eye = torch.eye(class_cosine.shape[0], device=z_norm.device, dtype=torch.bool)
        class_center_margin_loss = F.relu(
            class_cosine.masked_select(~class_eye) - float(inter_class_max_cos)
        ).pow(2).mean()
    else:
        class_center_margin_loss = flat_prototypes.sum() * 0.0
    if prototype_anchor_tangent is not None:
        anchor = prototype_anchor_tangent.to(device=prototype_tangent.device, dtype=prototype_tangent.dtype)
        anchor = anchor.reshape_as(prototype_tangent)
        anchor_direction = F.normalize(anchor, p=2, dim=-1)
        anchor_loss = (prototype_norm - anchor_direction).pow(2).sum(dim=-1).mean()
    else:
        anchor_loss = flat_prototypes.sum() * 0.0

    return {
        "hpec_mle_loss": mle_loss,
        "hpec_pcl_loss": pcl_loss,
        "hpec_pal_loss": pal_loss,
        "hpec_radius_loss": radius_loss,
        "hpec_diversity_loss": diversity_loss,
        "hpec_hsic_loss": hsic_loss,
        "hpec_intra_orthogonal_loss": intra_orthogonal_loss,
        "hpec_inter_margin_loss": inter_margin_loss,
        "hpec_class_center_margin_loss": class_center_margin_loss,
        "hpec_anchor_loss": anchor_loss,
        "prototype_cos_abs_mean": prototype_cos_abs_mean,
        "prototype_cos_abs_max": prototype_cos_abs_max,
        "prototype_same_class_cos_max": same_class_cos_max,
        "hpec_sinkhorn_assignment_entropy": sinkhorn_assignment_entropy,
        "hpec_sinkhorn_usage_min": assignment_usage_min,
        "hpec_sinkhorn_usage_max": assignment_usage_max,
    }


def prototype_hsic_loss(prototype_tangent, eps=1e-8):
    """HSIC-style prototype decorrelation in tangent space.

    We treat each prototype as one observation and each tangent dimension as a
    variable.  Linear-kernel HSIC on the centered prototype matrix penalizes
    shared second-order dependence between prototype directions.  Minimizing it
    keeps prototypes spread instead of collapsing to one neighborhood.
    """

    flat = prototype_tangent.reshape(-1, prototype_tangent.shape[-1])
    if flat.shape[0] <= 1:
        return flat.sum() * 0.0
    flat = flat - flat.mean(dim=0, keepdim=True)
    flat = F.normalize(flat, p=2, dim=-1)
    kernel = torch.matmul(flat, flat.T)
    n = kernel.shape[0]
    center = torch.eye(n, device=flat.device, dtype=flat.dtype) - torch.full(
        (n, n),
        1.0 / n,
        device=flat.device,
        dtype=flat.dtype,
    )
    centered_kernel = center @ kernel @ center
    offdiag = centered_kernel - torch.diag_embed(torch.diagonal(centered_kernel))
    centered_hsic = offdiag.pow(2).sum() / max(float((n - 1) ** 2), eps)
    gram = torch.matmul(flat, flat.T)
    gram_offdiag = gram - torch.diag_embed(torch.diagonal(gram))
    orthogonal_penalty = gram_offdiag.pow(2).sum() / max(float(n * (n - 1)), eps)
    return centered_hsic + orthogonal_penalty


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
        busemann_temperature=1.0,
        data_init=False,
        prototype_radius_reg_target=None,
        use_sinkhorn_ema=False,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=3,
        ema_alpha=0.9,
        ema_anchor_weight=0.1,
        sample_margin=0.2,
        intra_class_max_cos=0.35,
        inter_class_max_cos=0.0,
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
        self.busemann_temperature = float(busemann_temperature)
        self.data_init = bool(data_init)
        self.use_sinkhorn_ema = bool(use_sinkhorn_ema)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.ema_alpha = float(ema_alpha)
        self.ema_anchor_weight = float(ema_anchor_weight)
        self.sample_margin = float(sample_margin)
        self.intra_class_max_cos = float(intra_class_max_cos)
        self.inter_class_max_cos = float(inter_class_max_cos)
        self.prototype_radius_reg_target = float(
            prototype_radius if prototype_radius_reg_target is None else prototype_radius_reg_target
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
        self.prototypes = nn.Parameter(prototypes, requires_grad=bool(trainable_prototypes))
        self.latest_sinkhorn_stats = {}

    def maybe_initialize_from_batch(self, z_global, labels):
        """用训练 batch 的类中心 warm-start prototype。

        HPEC 参考实现使用 hyperspherical prototype 作为先验；在 fMRI 小样本场景中，
        随机球面方向可能和 HGCN 早期 embedding 偏差较大。因此这里参考
        Prototypical Networks 的“类别中心”思想，只在首次训练 batch 上做一次
        数据驱动初始化：每类取 logmap0(z) 的均值方向，再保留少量球面初始化偏移。
        """

        if self._data_initialized or not self.data_init:
            return
        if self.use_sinkhorn_ema and self.prototypes_per_class > 1:
            # 多原型 EMA 依赖初始原型的方向差异；首个 batch 的类中心 warm-start
            # 容易把同类多个 prototype 先压到一处，削弱 Sinkhorn 的均衡分配效果。
            self._data_initialized = True
            return
        label_index = _as_label_index(labels)
        if z_global.shape[0] != label_index.shape[0]:
            return

        with torch.no_grad():
            z_tangent = self.manifold.logmap0(self.manifold.projx(z_global.detach(), dim=-1), dim=-1)
            current_tangent = self.manifold.logmap0(self.manifold.projx(self.prototypes.detach(), dim=-1), dim=-1)
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
                data_radius = center_norm.clamp(
                    min=min(self.prototype_radius, 0.05),
                    max=max(self.prototype_radius, 0.05),
                )
                init_tangent[class_idx] = mixed_direction * data_radius

            initialized = self.manifold.expmap0(init_tangent.to(device=z_global.device, dtype=z_global.dtype), dim=-1, project=True)
            self.prototypes.data.copy_(self.manifold.projx(initialized, dim=-1).to(self.prototypes.dtype))
            self._data_initialized = True

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
            proto_tangent = self.manifold.logmap0(self.manifold.projx(self.prototypes.detach(), dim=-1), dim=-1)
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
                assigned_radius = center_norm.clamp(
                    min=min(self.prototype_radius, 0.05),
                    max=max(self.prototype_radius, 0.05),
                )
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

            updated = self.manifold.expmap0(new_tangent.to(device=z_global.device, dtype=z_global.dtype), dim=-1, project=True)
            self.prototypes.data.copy_(self.manifold.projx(updated, dim=-1).to(self.prototypes.dtype))
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
        with torch.no_grad():
            self.prototypes.data.copy_(self.manifold.projx(self.prototypes.data, dim=-1))
        prototypes = self.manifold.projx(self.prototypes.to(dtype=z_global.dtype), dim=-1)
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
        z_tangent = self.manifold.logmap0(z_global, dim=-1)
        prototype_tangent = self.manifold.logmap0(prototypes, dim=-1)
        prototype_similarity = torch.einsum(
            "bd,ckd->bck",
            F.normalize(z_tangent, p=2, dim=-1),
            F.normalize(prototype_tangent, p=2, dim=-1),
        )
        if self.energy_mode == "busemann":
            busemann_scores = busemann_score_matrix(self.manifold, z_global, prototypes, eps=self.eps)
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
            energy_matrix=energy_matrix,
            prediction=prediction,
            probability=probability,
            prototype_assignment=prototype_assignment,
        )

    def loss(self, energy_matrix, labels):
        if self.loss_mode in ("ce", "cross_entropy", "energy_ce"):
            return hpec_energy_ce_loss(energy_matrix, labels)
        if self.loss_mode != "margin":
            raise ValueError(
                f"Unsupported hpec loss_mode={self.loss_mode!r}. Use 'margin' or 'energy_ce'."
            )
        return hpec_energy_loss(energy_matrix, labels, margin=self.margin)

    def prototype_losses(self, z_global, labels):
        return hpec_multi_prototype_losses(
            z_global=z_global,
            prototypes=self.manifold.projx(self.prototypes.to(dtype=z_global.dtype), dim=-1),
            labels=labels,
            manifold=self.manifold,
            temperature=self.proto_temperature,
            prototype_radius_target=self.prototype_radius_reg_target,
            use_sinkhorn=self.use_sinkhorn_ema,
            sinkhorn_epsilon=self.sinkhorn_epsilon,
            sinkhorn_iters=self.sinkhorn_iters,
            prototype_anchor_tangent=self.prototype_anchor_tangent,
            sample_margin=self.sample_margin,
            intra_class_max_cos=self.intra_class_max_cos,
            inter_class_max_cos=self.inter_class_max_cos,
        )
