## Why

当前 `S-DeCI` 已经完成模块 1 的 Cycle/seasonal 特征提取和模块 2 的因果图学习，但分类输出仍然只依赖 Cycle/seasonal logits，尚未利用学习到的因果拓扑进行图信息传播。按照 `docs/新模块设计.md`，下一步需要接入模块 3：将模块 2 学到的因果图用于 HGCN，并读取 128 维（可配置）的双曲中心点作为当前阶段的分类依据。

## What Changes

- 在 `layers/` 中新增 HGCN 相关层，例如 Backclip、Poincaré Ball 投影、Mobius 图卷积、Fréchet 均值 readout 等可复用组件，严格参考 `docs/新模块设计.md` 与 `reference/` 下 HGCN、Differentiable-Frechet-Mean、HPEC 相关实现。
- 在 `S-DeCI` 中新增模块 3 组装逻辑：使用模块 1 的 Cycle/seasonal feature 和模块 2 学到的因果邻接矩阵 `A_learned`，得到全脑双曲中心点 `z_global`。
- 将 `z_global` 的维度设为超参数，默认使用 `128` 维，并在当前阶段直接以 `z_global` 作为分类依据，不新增模块 4、HPEC 原型角度损失或能量分类器。
- 允许分类 loss 经模块 3 自然反向传播到模块 2 的因果图学习参数，严格遵守 `docs/新模块设计.md` 中的联合损失构成。
- 为模块 3 新增中间量缓存与可视化，包括 clipped Cycle feature、Poincaré Ball 投影、HGCN 节点表示、`A_learned`、方向差分矩阵和 `z_global` 等。
- 在新增的 `S-DeCI` 模块 3 接入逻辑、HGCN 层和训练中间量缓存处添加中文注释，必要英文关键词如 `HGCN`、`Poincare Ball`、`Mobius`、`Frechet mean`、`z_global` 保留。
- 更新训练入口和测试脚本参数，使模块 3 可开关、双曲中心维度可配置，并能在低预算测试中跑通。
- 不修改 `docs/` 下原始设计文档；若需要记录本次实现后的项目说明，应在 `docs/` 下新建补充文档。
- 回滚方案：关闭 `use_hgcn_module3` 或将 `S-DeCI` 分类头切回原 Cycle/seasonal logits 分支；若需要彻底回滚，可移除新增 HGCN 层、`S-DeCI` 模块 3 初始化与训练参数，不影响原 `DeCI`。

## Capabilities

### New Capabilities

- `module3-hgcn-readout`: 定义模块 3 的 HGCN 双曲图卷积、Backclip、Poincaré Ball 投影、Fréchet 均值读取、`z_global` 输出和训练诊断要求。

### Modified Capabilities

- `s-deci-model`: `S-DeCI` 将从“仅使用 Cycle/seasonal logits 分类”扩展为可配置地使用模块 3 输出的双曲中心点 `z_global` 作为分类依据，并缓存模块 3 中间量。
- `module2-causal-learning`: 模块 2 学到的 `A_learned` 将被模块 3 作为图卷积拓扑使用，并需要接收来自模块 3 分类目标的梯度；总损失由 `z_global` 分类损失、reconstruction、DAG 和 L1 sparsity 构成。
- `tensor-visualization-helper`: S-DeCI 可视化要求扩展到模块 3 的双曲投影、HGCN 节点表示、`z_global` 和因果图传播相关中间量。

## Impact

- 影响代码范围：
  - `models/S_DeCI.py`
  - `layers/` 下新增 HGCN/双曲图卷积相关文件
  - `layers/causal_graph_learner.py`
  - `exp/exp_classification_CV.py`
  - `run_cv.py`
  - `test_training_smoke.py`
  - `test_matai_small_sample.py`
  - `utils/tensor_visualization.py`（仅在现有 helper 能力不足时修改）
- 可能新增依赖：
  - `geoopt`，用于 Poincaré Ball、Mobius 运算和必要的黎曼优化支持。
  - 若直接复用 `reference/Differentiable-Frechet-Mean` 的实现，需要将必要代码迁移到项目内，避免正式训练依赖 `reference/` 目录。
- 行为影响：
  - 默认是否启用模块 3 需要在设计阶段明确；若启用，`S-DeCI` 的主分类依据会从 seasonal logits 切换到 `z_global` 分类头。
  - 模块 2 的因果图将不再只是诊断量，也会作为模块 3 图传播输入参与分类路径。
  - 验证和测试需要覆盖 forward shape、分类 loss 反传、模块 2/3 联合训练、可视化输出和低预算训练跑通。
