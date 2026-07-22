## Why

当前模块 2 的 temporal NTS-NOTEARS 已经从“静态特征重构”改为“历史时间窗预测未来时间点”，方向上是正确的；但它的边学习器仍偏局部线性，表达能力有限，难以区分不同 lag、不同脑区交互模式。引入多头注意力可以让模块 2 在保留时序预测因果学习框架的同时，更灵活地学习 `x_j(t-l) -> x_i(t)` 的候选影响关系。

同时需要明确：attention weight 不能直接等同于因果矩阵。该变更的目标是用多头注意力生成候选跨时序影响，再通过结构门控、时序预测损失、稀疏/平滑约束与可选归因校正，得到稳定的 `A_lag` 因果图。

## What Changes

- 将模块 2 的正式方法扩展为 `Attention-guided Temporal NTS-NOTEARS`，在现有 temporal SEM/NTS-NOTEARS 路径上加入多头 lag-window cross-attention。
- 新增可复用的 attention-guided temporal causal learner，输入仍为模块 1 输出或保留的时间序列 `X: [B, T, N]` 或 `[B, T, N, D]`。
- 使用过去 `lag_order` 个时间窗预测未来时间点，学习 `A_lag: [lag_order, N, N]` 作为跨时间主因果图。
- 使用结构门控 `G_lag` 聚合多头注意力，避免直接把 raw attention map 当作因果矩阵。
- 保留 `A0: [N, N]`，其作用是建模同时间片残余依赖，并承载 DAGMA/NOTEARS 式无环约束；`A0` 不作为模块 3 的主分类图。
- 模块 3 默认继续使用 `A_lag.mean(dim=0)` 作为 learned causal graph，并可按现有 `module2_sample_correlation_blend` 与样本相关矩阵融合。
- 可视化输出需覆盖 `A_lag_mean`、每个 lag 的 `A_lag[k]`、`A0`、最终 `A_cls`，并在标题中标明 attention-guided 路径。
- 训练日志需显示 attention 图诊断量，例如 `attn_entropy`、`gate_mass`、`A_lag_mass`、`A0_mass`、`dag_loss`、`graph_delta`。
- 新建项目修改后的说明文档，不直接修改原始 doc 参考文档。
- 回滚方案：保留当前 `nts_notears`/`temporal_sem` 路径作为默认兼容或显式 fallback；若 attention-guided 方法效果不稳定，可将入口默认切回原 temporal NTS-NOTEARS，并删除/禁用新增 learner 的调用。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `module2-causal-learning`: 模块 2 需要支持 attention-guided temporal NTS-NOTEARS，输出显式 `A_lag`、`A0`、`A_cls` 与相关诊断量。
- `s-deci-model`: S-DeCI 需要能选择新的模块 2 方法，并将 attention-guided `A_lag_mean` 传入模块 3/GCN fallback。
- `training-test-scripts`: 训练入口与根目录测试脚本需要暴露精简后的 attention-guided 模块 2 参数，并支持快速训练检查与日志/可视化输出。

## Impact

- 影响模块范围：
  - `layers/`: 新增或修改模块 2 learner 层，主要涉及 attention-guided temporal causal learner。
  - `models/S_DeCI.py`: 模块 2 初始化、forward、图选择、缓存和可视化诊断需要接入新方法。
  - `run_cv.py`、`test_mdd_best_config.py`、`test_abide_best_config.py` 等训练入口：新增方法选择与少量 attention 参数。
  - `exp/exp_classification_CV.py`: 训练日志和可视化诊断字段可能需要补充。
  - `openspec/specs/`: 更新模块 2、S-DeCI 与训练脚本相关规范。
- 不改变数据集读取格式，不改变模块 3/4 的输入接口。
- 不新增真实因果图监督；合成数据中的真实图仍只能用于训练后评估和可视化。
- 不修改原始 `doc/` 参考文档；如需说明本次新设计，应新建独立说明文档。
