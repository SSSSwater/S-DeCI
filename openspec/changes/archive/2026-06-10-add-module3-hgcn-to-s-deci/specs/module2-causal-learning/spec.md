## ADDED Requirements

### Requirement: 模块 2 因果图供模块 3 HGCN 使用

模块 2 SHALL 将学习到的 `A_learned` 作为模块 3 HGCN 的图拓扑输入。

#### Scenario: 模块 3 读取 A_learned

- **GIVEN** `S-DeCI` 启用模块 2 和模块 3
- **WHEN** 模块 2 完成因果图学习 forward
- **THEN** 模块 3 MUST 使用模块 2 输出的连续邻接矩阵 `A_learned`
- **AND** `A_learned` 的方向语义 MUST 继续按 `A[parent, child]` 解释

#### Scenario: A_learned 保持可微

- **GIVEN** 模块 3 使用 `A_learned` 执行 HGCN 图传播
- **WHEN** 分类 loss 从 `z_global` 反向传播
- **THEN** `A_learned` MUST 保持在 autograd graph 中
- **AND** 模块 2 的因果图学习参数 MUST 能收到来自分类 loss 的梯度

### Requirement: 模块 2 与模块 3 联合训练不使用真实因果监督

模块 2 SHALL 在模块 3 联合训练时继续保持无监督因果学习约束，不得使用真实因果矩阵参与训练 loss。

#### Scenario: 联合 loss 不包含真实因果矩阵

- **GIVEN** 模块 2 和模块 3 联合训练
- **WHEN** 系统计算 `Loss_total`
- **THEN** loss MUST NOT 使用 `A_true`
- **AND** loss MUST NOT 使用 `A_structure_true`
- **AND** 真实因果矩阵若存在 MUST 仅用于独立实验指标或可视化诊断

#### Scenario: 保留模块 2 结构正则

- **GIVEN** 模块 3 分类 loss 会回传到模块 2 因果图
- **WHEN** 系统计算模块 2 辅助 loss
- **THEN** MUST 继续计算 reconstruction loss
- **AND** MUST 继续计算归一化 DAG acyclicity loss
- **AND** MUST 继续计算归一化 L1 sparsity loss

