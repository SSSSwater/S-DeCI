## ADDED Requirements

### Requirement: 模块 2 支持 temporal SEM 学习模式

模块 2 SHALL 在现有静态特征图学习之外，支持时间序列预测式 SEM 学习模式。

#### Scenario: 选择 temporal SEM 模式
- **GIVEN** 用户配置 `causal_learning_target == "temporal_sem"`
- **WHEN** `S-DeCI` 初始化模块 2
- **THEN** 系统 MUST 初始化 temporal SEM 因果学习器
- **AND** 系统 MUST NOT 使用静态特征重构作为唯一模块 2 训练目标

#### Scenario: 保留静态模式
- **GIVEN** 用户配置 `causal_learning_target == "static_feature"`
- **WHEN** `S-DeCI` 初始化模块 2
- **THEN** 系统 MUST 保留当前静态因果学习器
- **AND** 系统 MUST 继续支持 `nts_notears`、`dagma_logdet` 和 `dag_sampling`

### Requirement: 模块 2 输出 temporal SEM 诊断字段

模块 2 SHALL 在 temporal SEM 模式下输出可被训练循环读取的损失项和图诊断字段。

#### Scenario: 输出损失字典
- **WHEN** temporal SEM 模块完成 forward
- **THEN** 模块 MUST 暴露 temporal prediction loss
- **AND** MUST 暴露 DAGMA loss
- **AND** MUST 暴露 sparsity loss
- **AND** MUST 暴露 sample residual graph 正则
- **AND** 所有训练 loss MUST 能参与 PyTorch autograd

#### Scenario: 输出图诊断字典
- **WHEN** temporal SEM 模块完成 forward
- **THEN** 模块 MUST 暴露 `A0` 图统计
- **AND** MUST 暴露 `A_lag` 图统计
- **AND** MUST 暴露 DAGMA 调度阶段和有效权重
- **AND** MUST 暴露样本 residual graph 幅度统计

### Requirement: 模块 2 支持图稳定性评估

模块 2 SHALL 支持在 fold 结束后计算 learned graph 的稳定性指标。

#### Scenario: 计算 fold graph stability
- **WHEN** 至少一个 fold 完成训练
- **THEN** 系统 MUST 能保存该 fold 的 `A_shared` 或 temporal graph
- **AND** 多 fold 完成后 MUST 能计算 fold 间图相似度
- **AND** 图相似度 MUST 不影响模型训练梯度

#### Scenario: 计算 top-k 边频率
- **WHEN** 多个 fold 或多次 iteration 产生 learned graph
- **THEN** 系统 MUST 能统计 top-k 边出现频率
- **AND** MUST 能保存 edge frequency 矩阵或表格
- **AND** 输出 MUST 区分 `A0` 与 `A_lag`
