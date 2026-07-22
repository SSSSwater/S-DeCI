## ADDED Requirements

### Requirement: 模块 2 使用模块 1 去噪输出学习因果图

模块 2 SHALL 支持从模块 1 的去噪输出中学习因果图，而不是直接依赖噪声较强的原始输入。

#### Scenario: 接收去噪节点特征

- **GIVEN** 模块 1 已输出去噪节点特征
- **WHEN** 模块 2 执行因果图学习
- **THEN** 模块 2 MUST 使用去噪节点特征或由其构造的去噪时序表示作为主要输入
- **AND** 输入形状 MUST 与模块 2 当前 `[B, N, F]` 语义兼容

#### Scenario: 接收去噪时间序列

- **GIVEN** `causal_learning_target == "temporal_sem"`
- **WHEN** 模块 1 提供去噪后的时间序列表示
- **THEN** 模块 2 MUST 能使用该表示计算预测式 SEM loss
- **AND** temporal SEM loss MUST 能参与 autograd

### Requirement: 模块 2 支持因果图稳定性约束

模块 2 SHALL 支持在训练中约束增强视图或 batch 间的因果图一致性，以降低 ABIDE 短时序上的过拟合。

#### Scenario: 计算因果图稳定性 loss

- **GIVEN** 模块 2 可获得同一样本的原始视图和扰动视图
- **WHEN** `lambda_causal_stability > 0`
- **THEN** 模块 2 MUST 计算两种视图下因果图或因果效应的稳定性 loss
- **AND** 该 loss MUST 能加入总 loss 并反向传播

#### Scenario: 关闭稳定性 loss

- **WHEN** `lambda_causal_stability == 0`
- **THEN** 稳定性 loss MUST 不影响模块 2 训练
- **AND** 模块 2 MUST 保持当前训练行为兼容

### Requirement: 模块 2 限制样本级残差图自由度

模块 2 SHALL 在 ABIDE 完整四模块默认配置中限制 sample graph residual 的幅度与复杂度，使共享因果图承担主要结构表达。

#### Scenario: 低自由度样本残差图

- **GIVEN** `use_sample_graph_residual == 1`
- **WHEN** 模块 2 生成样本级残差图
- **THEN** 样本残差图 MUST 受 `sample_graph_delta_scale` 限幅
- **AND** 样本残差图 SHOULD 支持低秩或低隐藏维参数化
- **AND** 默认 ABIDE 配置 MUST 使用保守的 `sample_graph_delta_scale`

#### Scenario: 样本残差图正则

- **WHEN** `lambda_sample_graph_l1 > 0` 或 `lambda_sample_graph_deviation > 0`
- **THEN** 模块 2 MUST 计算对应正则项
- **AND** 日志 SHOULD 打印这些正则项的未加权值或加权贡献

### Requirement: 模块 2 输出因果图诊断

模块 2 SHALL 输出足够的因果图诊断量，用于判断 ABIDE 训练是否学到稳定结构。

#### Scenario: 输出 graph mass 和方向性

- **WHEN** 模块 2 完成 forward
- **THEN** 系统 MUST 能读取因果图总强度或 graph mass
- **AND** 系统 SHOULD 能读取方向性或反对称诊断

#### Scenario: 输出稳定性诊断

- **WHEN** 启用因果图稳定性约束
- **THEN** 系统 MUST 能读取 stability loss 或等价指标
- **AND** 可视化 SHOULD 能显示训练后因果图与阈值化图
