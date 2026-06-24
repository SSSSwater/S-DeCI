## Context

当前 `S-DeCI` 已经具备两部分能力：模块 1 从 fMRI 时间序列中提取 Cycle/seasonal feature，并用 seasonal logits 分类；模块 2 接收聚合后的 Cycle feature `C: [B, N, d_model]`，学习共享因果邻接矩阵 `A_learned: [N, N]`，并提供 reconstruction、DAG、L1 辅助 loss。

`docs/新模块设计.md` 对模块 3 的定位是：对因果特征执行 Backclip 限幅，投影到 Poincare Ball，用模块 2 的因果图执行 HGCN 双曲信息传播，再用可微 Fréchet mean 读取全脑拓扑中心 `z_global`。本次变更要求先不实现模块 4/HPEC 原型分类，而是直接用 `z_global` 作为分类依据。

参考源码主要来自：

- `reference/HPEC-main/`：`geoopt.PoincareBallExact`、Riemannian optimizer、双曲原型相关代码，可作为双曲流形接口使用方式参考。
- `reference/Differentiable-Frechet-Mean-master/`：Fréchet mean 与图聚合思路，可作为 readout 实现参考。
- 现有 `layers/causal_graph_learner.py`：模块 2 输出 `A_learned` 与方向性诊断量，是模块 3 的图拓扑输入。

## Goals / Non-Goals

**Goals:**

- 在 `S-DeCI` 中新增模块 3，使用模块 1 的 Cycle feature 和模块 2 的 `A_learned` 生成双曲全脑中心点 `z_global`。
- 将 `z_global` 维度设为超参数，默认 `128`，并直接作为当前阶段分类依据。
- 在 `layers/` 下新增可复用 HGCN/双曲 readout 层；模块 3 的装配、缓存和分类头保留在 `models/S_DeCI.py`。
- 让分类 loss 通过模块 3 自然回传到模块 2 因果图，严格遵守 `docs/新模块设计.md` 中的联合损失构成。
- 复用或迁移 reference 中必要 HGCN / Differentiable-Frechet-Mean 逻辑，正式训练不得依赖 `reference/` 路径。
- 为新增逻辑添加中文注释，并扩展训练中间量可视化。
- 保持 `models/DeCI.py` 不受影响。

**Non-Goals:**

- 不实现模块 4、HPEC 原型、角度能量损失或原型分类。
- 不修改 `docs/新模块设计.md` 原文；若需要记录本次落地方案，应新建补充文档。
- 不把真实因果矩阵作为训练监督。
- 不把 HGCN 层放入 `models/`，避免模型文件承担过多底层数学实现。
- 不强制所有模型使用 `geoopt` 优化器；仅在模块 3 需要时提供兼容路径。

## Decisions

### 1. HGCN 基础层放在 `layers/`，模块 3 装配放在 `S-DeCI`

新增文件建议为 `layers/hyperbolic_gcn_layer.py` 或相近名称，包含：

- `Backclip`
- `HyperbolicGraphConvolution`
- `FrechetReadout`
- `Module3HGCNReadout` 或组合层

`models/S_DeCI.py` 负责：

- 根据配置初始化模块 3。
- 将模块 2 输出的 `A_learned` 和 Cycle feature 输入模块 3。
- 缓存 `c_clipped`、`h0`、`h_gcn`、`z_global` 等中间量。
- 使用 `z_global` 的切空间表示或等价可分类表示接分类头。

选择原因：`layers/` 更适合放数学层和可复用图卷积逻辑；`S-DeCI` 保留数据流编排，便于后续模块 4 接入。

备选方案：全部写进 `models/S_DeCI.py`。该方案实现快，但会让模型文件过长，后续难以单独测试 HGCN 层。

### 2. 模块 3 默认启用时，分类依据切换到 `z_global`

新增配置建议：

- `use_hgcn_module3`: 是否启用模块 3。
- `hgcn_hidden_dim`: `z_global` 维度，默认 `128`。
- `hgcn_layers`: HGCN 层数，默认先用 `1`。
- `hgcn_curvature`: Poincare Ball 曲率，默认 `1.0`。
- `hgcn_backclip_radius`: Backclip 半径或阈值。
当 `use_hgcn_module3=1` 时，`S-DeCI.forward()` 仍只返回分类输出，但分类输出来自 `z_global` 分类头，而不是原 seasonal logits。原 seasonal logits 可保留为 fallback 或诊断量。

选择原因：用户明确要求“最终输出得到的 128 维双曲中心点先直接作为分类依据，不设置模块 4”。这意味着模块 3 不是旁路诊断，而是当前阶段主分类路径。

备选方案：把 `z_global` 和 seasonal logits 融合分类。该方案可能提升稳定性，但会混淆模块 3 效果，不符合当前实验目标。

### 3. HGCN 输入使用模块 1 Cycle feature，图拓扑使用模块 2 `A_learned`

数据流：

1. `x_enc: [B, T, N]`
2. 模块 1 输出聚合后的 `C: [B, N, d_model]`
3. 模块 2 输出 `A_learned: [N, N]`、`C_hat: [B, N, d_model]`
4. 模块 3 使用 `C` 和 `A_learned`
5. 输出 `H_gcn: [B, N, hgcn_hidden_dim]`
6. Fréchet readout 输出 `z_global: [B, hgcn_hidden_dim]`
7. 分类头输出 `[B, 1]` 或 `[B, classes]`

默认使用 `C` 而不是 `C_hat` 作为 HGCN 节点特征。`C_hat` 保留为模块 2 重构诊断量。

选择原因：`C` 是模块 1 的真实特征，`C_hat` 是模块 2 重构结果，直接用 `C` 能减少早期模块 2 重构误差对分类的额外干扰。

备选方案：使用 `C_hat` 或 `torch.cat([C, C_hat], dim=-1)`。这可作为后续实验配置，但本次先保持单一路径。

### 4. 模块 2 的因果图必须接收模块 3 分类梯度

模块 3 使用 `A_learned` 聚合邻居特征，因此 `z_global` 分类 loss 必须通过 HGCN 回传到 `A_learned` 和模块 2 参数。实现中不再设计“阻断分类 loss 到模块 2 因果图”的开关，避免偏离 `docs/新模块设计.md` 中模块 2 与模块 3 联合优化的目标。

当前阶段没有模块 4/HPEC，因此联合损失落地为：

```text
Loss_total =
    Loss_cls(z_global, label)
  + alpha * Loss_Recon(C, C_hat)
  + lambda * Loss_DAG(A_learned)
  + gamma * L1(A_learned)
```

其中 `Loss_cls` 使用 `z_global` 的分类头输出，`Loss_Recon`、`Loss_DAG` 和 `L1` 沿用模块 2 当前实现，训练 loss 不使用真实因果矩阵。

选择原因：用户明确要求去掉阻断配置，并严格遵守新模块设计文档中的损失函数构成。模块 3 的目的就是让分类信号告诉因果图哪些连接对判别有用。

### 5. 使用 `geoopt` 实现 Poincare Ball 与 Mobius 运算

模块 3 优先使用 `geoopt.PoincareBall` 或 `geoopt.PoincareBallExact`：

- Backclip 后执行 `expmap0` 得到 `h0`。
- 用 `mobius_matvec` 将特征从 `d_model` 映射到 `hgcn_hidden_dim`。
- 使用 `A_learned` 做邻居权重，在切空间或通过 Mobius addition 完成聚合。

对于邻居聚合，设计优先级：

1. 若 `geoopt` 接口和张量形状允许，严格使用 `mobius_add` / `mobius_matvec` 构造逐邻居聚合。
2. 若 116 节点逐边 Mobius 聚合过慢，则使用稳定的 tangent-space 聚合：`logmap0 -> A 聚合 -> expmap0`，并在代码注释中说明这是工程化近似，保持可微且便于训练。

选择原因：文档要求使用 Geoopt 和 Mobius 运算，同时 fMRI 116 节点批训练需要性能可控。

### 6. Fréchet readout 迁移到项目内，不直接依赖 `reference/`

读取 `reference/Differentiable-Frechet-Mean-master` 后，将必要的 Fréchet mean/readout 逻辑迁移或重写到 `layers/` 中。正式训练不得 import `reference/...`。

实现策略：

- 首选：实现可微 Fréchet mean 迭代器，对每个样本的 `[N, hgcn_hidden_dim]` 双曲节点表示求中心点。
- 稳定 fallback：使用 tangent-space weighted mean，即 `logmap0(h_gcn).mean(dim=1)` 后再 `expmap0`，并暴露为 `frechet_readout_method`，用于低预算训练和调试。

选择原因：reference 中的代码是研究实现，路径和依赖结构不适合作为正式训练直接依赖；迁移后更容易测试和维护。

### 7. 分类头使用 `logmap0(z_global)` 的欧氏切空间表示

虽然 `z_global` 是双曲点，但现有分类 loss 和评估流程都是欧氏输出。分类头建议使用：

```python
z_tangent = manifold.logmap0(z_global)
logits = classifier(z_tangent)
```

二分类仍输出 sigmoid 后 `[B, 1]`，多分类输出 `[B, classes]` logits。

选择原因：直接对双曲点坐标做线性分类可行但几何语义弱；使用 `logmap0` 是常见的双曲模型 readout 方式，兼容现有 MSE/CE 训练。

### 8. 可视化扩展但默认关闭

`S-DeCI.visualize_causal_intermediates()` 可扩展为同时展示模块 2 和模块 3 中间量，或新增 `visualize_module3_intermediates()`。

建议展示：

- `Cycle/seasonal feature`
- `A_learned`
- `A_learned - A_learned.T`
- `C_clipped`
- `H0/Poincare projection`
- `H_gcn`
- `z_global`
- `logmap0(z_global)`

3D 张量继续只显示 Batch0，并由 `visualize_tensors` 显示 shape 和 Batch 提示。默认 `visualize_causal=0`，训练结束后仅在显式开启时保存。

### 9. 新增实现说明文档，不修改原始 docs

根据项目规则，`docs/新模块设计.md` 保持初始参考不修改。若实现中需要记录“模块 3 当前落地版本”，新建例如 `docs/S-DeCI模块3实现说明.md`，说明：

- 与原始设计一致的部分。
- 当前阶段暂不实现模块 4。
- Fréchet readout 的实现或 fallback。
- 训练和可视化入口。

## Risks / Trade-offs

- [Risk] `geoopt` 未安装或版本不兼容 → 在实现前检查 `.venv`，必要时更新 `requirements.txt`，并在测试中覆盖 import。
- [Risk] 逐边 Mobius 聚合在 116 节点上过慢 → 提供 tangent-space 聚合 fallback，并用低预算训练测试验证速度。
- [Risk] Fréchet mean 迭代数过多导致训练慢或梯度不稳定 → 将迭代次数设为超参数，默认使用小步数；必要时使用 tangent mean readout。
- [Risk] 分类 loss 直接优化 `A_learned` 可能破坏因果稀疏性 → 严格保留模块 2 的 DAG/L1 正则项，并通过 `lambda_causal_dag`、`lambda_causal_l1` 权重控制结构约束强度。
- [Risk] `z_global` 分类替换 seasonal logits 后早期效果下降 → 保留配置回退到 seasonal 分类路径，确保训练能跑通并便于 ablation。
- [Risk] 双曲点接近边界造成数值不稳定 → Backclip、`projx` 和 eps clamp 必须在 HGCN 层中使用。

## Migration Plan

1. 检查 `.venv` 是否已有 `geoopt`；若缺失，补充依赖并验证 import。
2. 新增 `layers/` 下 HGCN/双曲 readout 文件，迁移或重写 Backclip、Poincare projection、HGCN、Fréchet readout。
3. 修改 `models/S_DeCI.py`，新增模块 3 初始化、forward 数据流、分类头、缓存和中文注释。
4. 修改训练参数入口：`run_cv.py`、`test_training_smoke.py`、`test_matai_small_sample.py`。
5. 扩展可视化输出，确保显式开启时保存模块 3 中间量，默认不保存。
6. 新建实现说明文档，例如 `docs/S-DeCI模块3实现说明.md`，不修改原始参考文档。
7. 增加验证：forward shape、`z_global` shape、分类 loss 能回传到模块 2 因果图参数、低预算训练、可视化图片。
8. 回滚时设置 `use_hgcn_module3=0` 切回 seasonal logits 分类；必要时删除新增 HGCN 层和参数，不影响 `DeCI`。

## Open Questions

- Fréchet readout 是否必须完全复刻 reference 的 autograd function，还是允许 tangent mean 作为默认工程实现？建议 specs 要求支持可微 readout，并在任务中先实现稳定版本，再按 reference 补充严格版本。
- 当前 `z_global` 分类头使用 `logmap0(z_global)` 后线性分类；是否需要同时缓存原始双曲点供后续模块 4 使用？建议缓存两者。
