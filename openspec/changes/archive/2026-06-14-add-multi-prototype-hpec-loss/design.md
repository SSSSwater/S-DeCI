## Context

当前 `S-DeCI` 已有模块 1-4：模块 1 生成 Cycle/seasonal feature，模块 2 学习因果图，模块 3 通过 HGCN 得到 `z_global`，模块 4 使用 HPEC prototype energy 做分类。现有 HPEC 实现是每个类别一个 prototype，适合先验证 HPEC 能量分类，但不足以表达同一诊断类别内部的多种连接模式。

用户提供的论文笔记强调“每类多个 prototype”可以保留类内多样性，并通过 `L_mle`、`L_pcl`、`L_pal` 等损失约束样本与 prototype 分布的匹配关系。本变更在现有 `S-DeCI` 上扩展模块 4，不直接修改原始 `docs/新模块设计.md`，而是新增一份实现说明文档记录落地设计。

## Goals / Non-Goals

**Goals:**

- 将 HPEC prototype 从每类一个扩展为每类 `K` 个，默认 `K` 可配置。
- 支持从多个 prototype 聚合类别级 energy，用于保持现有 HPEC 分类接口。
- 新增 prototype-related loss：`L_mle`、`L_pcl`、`L_pal`，并能按权重加入总 loss。
- 在中间量 heatmap 和最终 t-SNE 中展示多 prototype。
- 保持 `hpec_prototypes_per_class=1` 时尽量接近当前单 prototype 行为。

**Non-Goals:**

- 本变更不实现论文中的 site-independence HSIC loss；MDD/ABIDE 等数据当前未统一暴露 site label，强行加入会扩大范围。
- 本变更不改变模块 1、模块 2、模块 3 的核心结构。
- 本变更不修改原始 `docs/新模块设计.md`，只新增实现说明文档。

## Decisions

### 1. Prototype 张量形状使用 `[C, K, D]`

模块 4 内部 prototype 改为 `prototypes: [num_classes, prototypes_per_class, embedding_dim]`。当 `K=1` 时可自然退化为当前单 prototype 情况。

选择原因：类别和类内 prototype 的语义清楚，可直接按 label 选择同类 prototype，也方便 t-SNE 标出 `class/prototype index`。

备选方案：把所有 prototype flatten 为 `[C*K, D]` 并额外维护 label 映射。该方案实现分类聚合时更绕，因此不作为主方案。

### 2. 类别 energy 使用多 prototype 聚合

模块 4 对每个样本计算 `energy_per_proto: [B, C, K]`。类别级 energy 使用 soft-min 聚合：

```text
class_energy[b, c] = -tau_energy * logsumexp(-energy_per_proto[b, c, :] / tau_energy)
```

当 `tau_energy` 较小时近似选择最匹配 prototype；当 `K=1` 时等价于单 prototype energy。

选择原因：`min` 不够平滑，`mean` 会被不相关 prototype 稀释；soft-min 能保留“最相关 prototype 解释样本”的语义，并保持可微。

### 3. Prototype loss 在切空间中计算

HPEC energy 仍在 Poincare Ball 中计算；`L_mle`、`L_pcl`、`L_pal` 使用 `logmap0(z_global)` 与 `logmap0(prototypes)` 后的切空间向量计算，并进行 `L2 normalize`。

选择原因：论文笔记中的 prototype learning 以高维球面/余弦相似度为主；切空间更便于实现相似度、距离和 t-SNE 对照，同时不破坏 HPEC energy 的双曲分类路径。

### 4. `L_mle` 使用同类/异类多 prototype 概率匹配

`L_mle` 使用样本与所有 prototype 的 cosine similarity，按温度 `hpec_proto_temperature` 缩放。基础形式为：

```text
score[b, c, k] = cos(z_b, p_c,k) / tau
L_mle = CE(logsumexp(score over k), label)
```

如果实现 Sinkhorn 软分配，则对同类 prototype 的聚合可乘以 Sinkhorn assignment 权重；否则先使用 logsumexp 作为稳定可微近似。

选择原因：先保证可训练与可解释，避免因 batch 内某类别样本过少导致 Sinkhorn 不稳定。Sinkhorn 可作为实现增强项，但不得阻塞基础功能。

### 5. `L_pcl` 和 `L_pal` 的角色分开

- `L_pcl`：在 prototype 空间中约束同类 prototype 与异类 prototype 的相似度关系。实现时 MUST 避免 prototype collapse；若采用“同类拉近、异类推开”的形式，需要保留类内多 prototype 多样性，可通过 margin 或排除过强类内收缩实现。
- `L_pal`：对每个样本找同类中最相似 prototype，最小化样本与该 prototype 的切空间距离。

选择原因：`L_mle` 提供概率分类约束，`L_pal` 直接增强样本到同类 prototype 的实例级对齐，`L_pcl` 负责 prototype 之间的结构边界。

### 6. 训练循环通过模型缓存读取新增 loss

`S-DeCI.compute_primary_loss(labels)` 继续返回主 HPEC/classification loss。新增 prototype loss 通过模型缓存或 `get_aux_losses()` 暴露，训练循环将其加到总 loss 并打印。

建议总 loss：

```text
Loss_total =
  Loss_HPEC
  + alpha * Loss_Recon
  + lambda * Loss_DAG
  + gamma * L1
  + beta_mle * L_mle
  + beta_pcl * L_pcl
  + beta_pal * L_pal
```

选择原因：不改变 `forward()` 返回值，兼容当前实验管线和测试脚本。

## Risks / Trade-offs

- [Risk] 多 prototype 可能增加过拟合，尤其是小样本数据集上 `K` 太大时。  
  Mitigation: 默认 `K` 使用保守值；暴露 `hpec_prototypes_per_class` 和 loss 权重，允许快速退回 `K=1`。

- [Risk] `L_pcl` 若实现为过强的类内拉近，会抵消“类内多样性”的目的。  
  Mitigation: 设计中要求避免 prototype collapse，可采用 margin、温度或较小权重，并在 t-SNE 中观察 prototype 分布。

- [Risk] Sinkhorn 软分配在 batch 内某类别样本少于 prototype 数时不稳定。  
  Mitigation: 基础实现可使用 logsumexp 聚合；Sinkhorn 作为可选增强，不作为唯一训练路径。

- [Risk] prototype t-SNE 点数增加后图例拥挤。  
  Mitigation: prototype 使用星形 marker，label 使用 `prototype c-k` 或简化图例；必要时只标类别不标全部编号。

## Migration Plan

1. 扩展 HPEC 层，支持 `[C, K, D]` prototype、类别 energy 聚合和新增 prototype loss。
2. 扩展 `S-DeCI` 初始化、loss 缓存、可视化缓存和预测接口。
3. 扩展训练入口参数和测试脚本参数。
4. 更新最终 t-SNE，使多个 prototype 与 train/test embedding 一起降维显示。
5. 新增 `docs/S-DeCI多原型HPEC实现说明.md`，记录调用方式、loss 构成和回滚方式。

回滚：设置 `hpec_prototypes_per_class=1`，并关闭 `lambda_hpec_mle`、`lambda_hpec_pcl`、`lambda_hpec_pal`；若需要代码回滚，则恢复 HPEC 单 prototype 层与训练入口参数。

## Open Questions

- 默认每类 prototype 数量是否应直接采用论文中的 `K=50`，还是先用较小值如 `K=4/8` 便于小样本 fMRI 训练稳定。
- `L_pcl` 是否采用论文原式，还是加入 diversity margin 防止同类 prototype 完全坍缩。
- 后续是否需要单独新增 site label 读取能力，再实现 HSIC site-independence loss。
