## ADDED Requirements

### Requirement: S-DeCI 支持模块 1 时间序列输出

`S-DeCI` SHALL 在保留现有节点特征输出的同时，支持向模块 2 提供节点级时间序列输出。

#### Scenario: 模块 1 启用时提供分解后时间序列
- **GIVEN** `use_deci_module1 == 1`
- **AND** `causal_learning_target == "temporal_sem"`
- **WHEN** `S-DeCI.forward()` 执行模块 1
- **THEN** 模型 MUST 缓存模块 1 产生的节点时间序列或可等价表示时间动态的中间量
- **AND** 该时间序列 MUST 用于 temporal SEM 模块 2
- **AND** 模型 MUST 继续提供模块 3/4 所需的 `[B, N, d_model]` 节点特征

#### Scenario: 模块 1 关闭时回退到原始时间序列
- **GIVEN** `use_deci_module1 == 0`
- **AND** `causal_learning_target == "temporal_sem"`
- **WHEN** `S-DeCI.forward()` 接收原始输入
- **THEN** 模块 2 MUST 使用原始节点时间序列或归一化后的原始节点时间序列作为 temporal SEM 输入
- **AND** 模型 MUST 继续使用 raw projection 生成模块 3/4 或 fallback 分类所需节点特征

### Requirement: S-DeCI 可切换静态与时间因果路径

`S-DeCI` SHALL 支持在现有静态特征因果学习与新增 temporal SEM 因果学习之间切换。

#### Scenario: 使用 temporal SEM 路径
- **GIVEN** `use_causal_module2 == 1`
- **AND** `causal_learning_target == "temporal_sem"`
- **WHEN** 模型执行 forward
- **THEN** 模型 MUST 调用 temporal SEM 因果学习器
- **AND** 模块 2 auxiliary loss MUST 来自 temporal SEM prediction loss 与图正则
- **AND** 下游图传播路径 MUST 使用 temporal SEM 输出的有效图

#### Scenario: 回退静态特征路径
- **GIVEN** `use_causal_module2 == 1`
- **AND** `causal_learning_target == "static_feature"`
- **WHEN** 模型执行 forward
- **THEN** 模型 MUST 保持当前静态 `CausalGraphLearner` 行为
- **AND** 训练流程 MUST 继续读取现有 reconstruction、DAG 和 L1 auxiliary loss

### Requirement: S-DeCI 传递 temporal graph 给下游模块

`S-DeCI` SHALL 将 temporal SEM 学到的图转换为模块 3 HGCN 或 GCN fallback 可消费的 adjacency。

#### Scenario: 融合 A0 与 A_lag
- **GIVEN** temporal SEM learner 输出 `A0` 和 `A_lag`
- **WHEN** 下游模块需要 adjacency
- **THEN** `S-DeCI` MUST 生成一个可传播的 `A_effective`
- **AND** `A_effective` MUST 保持 `A[parent, child]` 方向语义
- **AND** 融合方式 MUST 被记录到诊断信息中

#### Scenario: 下游模块支持 batch 图
- **GIVEN** temporal SEM learner 启用样本残差图
- **WHEN** 下游 HGCN 或 GCN fallback 接收 `A_effective`
- **THEN** `A_effective` MAY 为 `[B, N, N]`
- **AND** 下游模块 MUST 正确处理 batch adjacency
