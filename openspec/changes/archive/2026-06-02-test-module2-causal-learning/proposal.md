## Why

`docs/新模块设计.md` 中的模块 2 负责从 Cycle 特征 `C` 中学习脑区间潜在因果邻接矩阵 `A`，这是后续图卷积和联合损失的前置能力。当前需要先将模块 2 独立实现并验证其可学习性，避免在直接衔接 `S-DeCI` 后难以区分问题来自前端特征、因果模块还是训练目标。

## What Changes

- 在根目录新增 `module_2_test/` 文件夹，用于集中放置本次模块 2 测试相关代码。
- 在 `module_2_test/` 下新增独立的模块 2 Python 文件，用于实现可微因果邻接矩阵学习，不放入 `models/`，也不直接接入 `S-DeCI`。
- 模块输入口径按后续应接收的 Cycle 特征 `C` 设计，默认形状为 `[B, N, 64]`。
- 模块内部维护可学习邻接参数 `A`，使用 mask 强制对角线为 `0`，并输出有效邻接矩阵。
- 实现基于 `A` 的特征重构 `C_hat`，用于训练约束。
- 实现 DAG acyclicity penalty，参考 `reference/` 中 Differentiable DAG / NOTEARS 相关源码思想。
- 根据 `docs/新DAG因果方法.md` 新增 Analytic DAG Constraint，使用谱半径缩放后的矩阵逆约束，并在训练脚本中与 NOTEARS 方法对比。
- 在 `module_2_test/` 下新增合成数据生成代码与训练检查脚本，构造带有明确因果关系的 Cycle-like 输入，并训练模块 2 恢复已知因果结构；合成数据必须支持 `n_nodes > 8`，边权重由随机种子控制生成，不使用固定权重。
- 训练 loss 不使用真实因果矩阵或二值结构标签；`A_true` 和 `A_structure_true` 只用于训练完成后的指标、权重差值和可视化对比。
- 训练脚本必须报告重构误差、DAG penalty 和邻接恢复指标，并显式比较学习到的因果矩阵 `A_learned` 与生成训练样本时使用的 `A_true` 是否一致。
- 训练脚本必须支持在同一次运行中分别训练 NOTEARS 与 Analytic DAG 两种方法，并输出 `comparison_summary.json`。
- 训练脚本必须使用已有 `utils.tensor_visualization.visualize_tensors` 将 `A_true`、`A_learned` 和必要的差异矩阵保存为 heatmap，便于人工检查因果矩阵一致性。
- 本次不实现模块 3、模块 4，也不接入 `S-DeCI` 的输出。

## Capabilities

### New Capabilities

- `module2-causal-learning`: 独立的模块 2 因果图学习能力，覆盖可学习邻接矩阵、DAG penalty、Cycle 特征重构、合成因果输入生成和训练有效性检查。

### Modified Capabilities

## Impact

- 影响文件：新增 `module_2_test/`，其中包含模块 2 实现文件、合成数据生成代码、训练检查脚本、可视化输出目录和必要的本地说明文件。
- 影响系统：新增一个可独立训练和验证的因果学习模块，不改变现有 `S-DeCI`、`DeCI`、数据加载、主训练流程或分类模型注册逻辑。
- 依赖：优先使用现有 `.venv` 中的 `torch`、`numpy` 等依赖；如参考源码需要额外库，必须在 design 阶段明确说明并尽量避免引入。
- 风险：合成因果数据的生成逻辑必须足够明确，否则训练指标可能只反映相关性或重构能力，而不能证明学到了预设因果结构。
