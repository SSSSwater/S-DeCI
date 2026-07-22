## Why

当前 S-DeCI 的模块二虽然借鉴了 NOTEARS / NTS-NOTEARS 和 Differentiable-DAG-Sampling，但实际输入已经是模块一输出的节点特征 `C: [B, N, D]`，并不包含显式 lag 时序结构，因此更接近静态 feature-level DAG learner。该实现容易出现因果矩阵变化小、图结构与分类目标耦合不稳定、全局共享图难以表达不同标签差异等问题。

本变更希望在保留模块设计合理性的前提下，增强模块二的因果学习稳定性与判别性，使其既能作为模块三的有向图输入，又能继续支持与现有相关矩阵 fallback 做对比。

## What Changes

- 将模块二语义明确为 S-DeCI 中的 `feature-level / temporal-aware causal graph learner`，避免将当前 `[B, N, D]` 输入误称为严格 NTS-NOTEARS。
- 新增或完善 `dagma_logdet` 风格的 DAG 约束作为默认推荐方法，用于替代大规模 ROI 图上较不稳定的固定权重 matrix-exponential penalty。
- 保留现有 `nts_notears` 与 `dag_sampling` 方法作为可切换对照路径。
- 为模块二输入增加独立标准化，使 DAG 学习不直接受模块一输出尺度漂移影响。
- 将单一全局共享因果图扩展为“共享图 + 样本残差图”的可选结构，使不同样本/标签可以产生轻量差异化图结构。
- 为 reconstruction、DAG、L1、样本残差图约束增加可调度权重，避免固定小权重导致 DAG 约束无效或图结构早期僵死。
- 增加模块二诊断输出，包括 DAG penalty、L1、图变化幅度、共享图/样本残差图统计和方向性指标。
- 新建修改后说明文档，不直接修改 `docs/` 中作为初始参考的原文档。
- 回滚方案：保留现有 `nts_notears` / `dag_sampling` 参数和旧全局共享图路径，新增参数默认可切回旧行为；如新方法效果不佳，可通过关闭新方法或关闭样本残差图回到当前实现。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `module2-causal-learning`: 修改模块二因果学习的结构、方法切换、损失调度、样本级图表达和诊断可视化要求。
- `s-deci-model`: 更新 S-DeCI 对模块二输出图的调用语义，支持共享图与样本残差图输入模块三/GCN fallback。

## Impact

- 影响范围：
  - `layers/causal_graph_learner.py`
  - `models/S_DeCI.py`
  - `run_cv.py`
  - `exp/exp_classification_CV.py`
  - 相关测试脚本与训练脚本
  - `openspec/specs/module2-causal-learning/spec.md`
  - `openspec/specs/s-deci-model/spec.md`
- 对外参数会新增模块二方法、图残差、损失调度、标准化相关参数；已有参数应保持兼容。
- 不引入强制新数据依赖；若实现 DAGMA log-det 约束，应基于 PyTorch 张量运算完成。
- `docs/` 下初始设计文档不直接修改；新增项目内更新说明文档用于记录修改后的设计。
