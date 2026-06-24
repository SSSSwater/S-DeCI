## Why

当前 `S-DeCI` 已实现新设计中的模块 1，并且模块 2 已在 `module_2_test/` 中独立验证了从 Cycle-like 特征学习因果图的能力。现在需要把模块 2 接入 `S-DeCI` 的模块 1 输出，使模型在正式训练路径中学习脑区间因果关系，同时保持当前阶段的分类逻辑仍只使用周期特征，避免提前引入尚未实现的模块 3。

## What Changes

- 修改 `models/S_DeCI.py`，在模块 1 的 DeCI block 输出周期特征后接入模块 2 因果图学习。
- 复用已实现的模块 2 因果学习逻辑，学习全局共享的脑区因果邻接矩阵 `A`，不再只停留在 `module_2_test/` 独立实验目录。
- 模块 2 输入使用模块 1 产生的 Cycle/seasonal 特征，保持形状语义为 `[B, N, d_model]`。
- `S-DeCI` 的最终分类输出仍只使用 Cycle/seasonal logits，不使用因果图卷积或模块 3 分类结果。
- 训练时新增模块 2 相关 loss 输出或模型属性，使后续训练流程能够将 reconstruction、DAG acyclicity、L1 sparsity 等因果学习损失纳入优化。
- 支持可视化关键中间量，包括模块 1 输出的周期特征、模块 2 学到的 `A_learned`、阈值化后的因果矩阵、重构特征和必要的差异/诊断矩阵。
- 可视化复用 `utils.tensor_visualization.visualize_tensors`，默认不改变训练、验证或推理行为，只有显式开启时才保存图片。
- 不实现模块 3，不引入 HGCN，不改变当前“分类只用周期特征”的阶段性约束。
- 不直接修改 `models/DeCI.py` 的主模型逻辑。

## Capabilities

### New Capabilities

无。本次是在既有 `S-DeCI` 与模块 2 能力上做集成，不引入独立新能力域。

### Modified Capabilities

- `s-deci-model`: 扩展 `S-DeCI`，使其在模块 1 输出周期特征后接入模块 2 因果图学习，同时保持分类仍只使用周期特征。
- `module2-causal-learning`: 将模块 2 从独立合成测试能力扩展为可被 `S-DeCI` 正式模型复用的因果图学习组件，并要求其支持真实模块 1 输出作为输入。
- `tensor-visualization-helper`: 增加 `S-DeCI` + 模块 2 集成场景下的中间量可视化调用要求，确保可视化默认不影响训练行为。

## Impact

- 影响文件：
  - `models/S_DeCI.py`
  - `module_2_test/causal_graph_learner.py` 或新建共享模块文件（如果设计阶段确认需要从测试目录迁出）
  - `exp/exp_classification_CV.py` 或训练流程中可读取模型辅助 loss 的位置
  - `utils/tensor_visualization.py` 的调用点
  - 相关 README 或新增说明文档
- 影响训练：
  - `run_cv.py --model S-DeCI` 仍应能跑通。
  - 分类指标仍来自 Cycle/seasonal 分支。
  - 因果学习 loss 需要参与训练总损失，或至少通过清晰的模型输出机制供训练循环纳入。
- 影响输出：
  - 可保存因果矩阵和中间特征 heatmap。
  - 可记录模块 2 的 reconstruction、DAG penalty、L1 sparsity 等诊断指标。
- 不新增外部依赖，继续使用现有 `torch`、`numpy`、`matplotlib`。
- 回滚方式：
  - 撤销 `S_DeCI.py` 中模块 2 接入逻辑。
  - 移除训练流程中对模块 2 loss 的读取和叠加。
  - 删除新增的 `S-DeCI` 因果可视化调用点或配置项。
  - 保留已归档的独立 `module_2_test/`，不影响模块 2 独立验证代码。
