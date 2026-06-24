## ADDED Requirements

### Requirement: 模块 3 支持因果图 edge dropout

模块 3 SHALL 支持对模块 2 输出的因果 adjacency 进行训练期 edge dropout，以降低 HGCN 对少量训练边的记忆。

#### Scenario: 训练期 edge dropout

- **GIVEN** `causal_edge_dropout > 0`
- **AND** 模型处于 training 模式
- **WHEN** 模块 3 接收模块 2 输出的 adjacency
- **THEN** 模块 3 MUST 随机丢弃部分非对角边
- **AND** 丢弃后的 adjacency MUST 继续经过现有归一化流程
- **AND** 该操作 MUST 不改变 batch 和节点维度

#### Scenario: 推理期不丢边

- **GIVEN** `causal_edge_dropout > 0`
- **AND** 模型处于 evaluation 模式
- **WHEN** 模块 3 接收 adjacency
- **THEN** 模块 3 MUST 使用完整 adjacency
- **AND** MUST NOT 随机丢弃边

### Requirement: 模块 3 提供双曲半径与切空间诊断

模块 3 SHALL 缓存和输出 `z_global` 半径、`z_tangent` 范数等诊断量，用于分析双曲空间是否塌缩或过拟合。

#### Scenario: 缓存半径诊断

- **WHEN** 模块 3 完成 `z_global` readout
- **THEN** 系统 MUST 能读取 `z_global` 的 Poincare 半径分布或等价范数
- **AND** 训练日志 SHOULD 能打印 batch 或 epoch 级统计

#### Scenario: 缓存切空间诊断

- **WHEN** 模块 3 计算 `logmap0(z_global)` 或 `z_tangent`
- **THEN** 系统 MUST 能读取其均值、方差或范数诊断
- **AND** 可视化 SHOULD 能显示这些诊断量与 label 的关系

### Requirement: 模块 3 支持双曲表示正则

模块 3 SHALL 支持可配置的双曲表示正则，以约束 ABIDE 训练中 `z_global` 的半径和分布。

#### Scenario: 计算半径正则

- **WHEN** `lambda_hgcn_radius_reg > 0`
- **THEN** 模块 3 MUST 计算 `z_global` 半径相关正则
- **AND** 该正则 MUST 能加入总 loss 并反向传播

#### Scenario: 关闭半径正则

- **WHEN** `lambda_hgcn_radius_reg == 0`
- **THEN** 半径正则 MUST 不影响训练
- **AND** 模块 3 MUST 保持当前行为兼容
