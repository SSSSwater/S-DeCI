# s-deci-gcn-fallback Specification

## Purpose
TBD - created by archiving change add-s-deci-module-toggles. Update Purpose after archive.
## Requirements
### Requirement: GCN fallback 分类路径

系统 SHALL 在 `S-DeCI` 的模块 3/4 联合禁用时，提供普通 Euclidean GCN fallback 分类路径，用于替代 HGCN + HPEC。

#### Scenario: 使用模块 2 因果矩阵作为 GCN adjacency
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **AND** `use_causal_module2 == 1`
- **WHEN** `S-DeCI.forward()` 完成模块 1 特征提取和模块 2 因果图学习
- **THEN** GCN fallback MUST 使用模块 2 输出的 `A_learned` 作为 adjacency
- **AND** GCN fallback MUST 使用模块 1 输出的 `[B, N, d_model]` 节点特征作为输入
- **AND** 模型 MUST 输出与现有分类训练流程兼容的 logits 或二分类分数

#### Scenario: 使用样本相关矩阵作为 GCN adjacency
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **AND** `use_causal_module2 == 0`
- **WHEN** `S-DeCI.forward()` 接收到 `correlation_matrix`
- **THEN** GCN fallback MUST 使用 `correlation_matrix` 作为 batch 级 adjacency
- **AND** GCN fallback MUST NOT 初始化或调用模块 2 causal graph learner
- **AND** 总 loss MUST NOT 包含模块 2 reconstruction、DAG 或 L1 auxiliary loss

#### Scenario: 缺少 adjacency 时清晰失败
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **AND** `use_causal_module2 == 0`
- **WHEN** `S-DeCI.forward()` 未接收到 `correlation_matrix`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 说明 GCN fallback 需要样本相关矩阵或启用模块 2

### Requirement: GCN fallback 中间量缓存

系统 SHALL 缓存 GCN fallback 的关键中间量，供训练诊断、heatmap 和 t-SNE 可视化使用。

#### Scenario: 缓存 GCN fallback 表征
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** GCN fallback 完成 forward
- **THEN** 模型 MUST 缓存实际使用的 adjacency
- **AND** 模型 MUST 缓存 GCN hidden、readout feature 和最终分类输出
- **AND** 这些缓存 MUST 不改变 `S-DeCI.forward()` 的主返回值

#### Scenario: 可视化区分 GCN 与 HGCN 路径
- **GIVEN** 显式启用中间量可视化
- **WHEN** 当前 fold 训练结束
- **THEN** 系统 MUST 能保存 GCN fallback 的 adjacency 和 readout 可视化
- **AND** 文件名或标题 MUST 能区分 `gcn_fallback` 与 `hgcn_hpec`

