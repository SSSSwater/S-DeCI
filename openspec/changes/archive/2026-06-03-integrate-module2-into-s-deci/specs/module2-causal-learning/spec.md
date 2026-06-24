## ADDED Requirements

### Requirement: 模块 2 提供正式可复用因果学习组件

模块 2 SHALL 提供可被正式模型 import 的 PyTorch 因果图学习组件，用于从 Cycle/seasonal feature 中学习共享脑区因果邻接矩阵。

#### Scenario: 正式模型可复用模块 2

- **WHEN** `S-DeCI` 需要接入模块 2
- **THEN** 系统 MUST 提供不依赖 `module_2_test/` 测试目录的模块 2 核心组件
- **AND** 该组件 MUST 能被 `models/S_DeCI.py` 或其依赖层正常 import

#### Scenario: 保留独立测试目录

- **WHEN** 正式可复用组件创建完成
- **THEN** `module_2_test/` MUST 继续保留独立测试、合成数据和训练检查用途
- **AND** 正式模型 MUST NOT 从 `module_2_test/` 直接导入生产训练所需组件

### Requirement: 模块 2 接收 S-DeCI 模块 1 输出

模块 2 SHALL 支持以 `S-DeCI` 模块 1 输出的 Cycle/seasonal feature 作为输入，并学习全局共享的因果图。

#### Scenario: 输入真实 Cycle feature

- **WHEN** 模块 2 接收来自 `S-DeCI` 的 `C`
- **THEN** `C` 的形状 MUST 为 `[B, N, d_model]`
- **AND** 模块 2 MUST 保持 batch、节点和特征维度语义不变

#### Scenario: 输出共享因果矩阵

- **WHEN** 模块 2 完成 forward
- **THEN** MUST 输出形状为 `[N, N]` 的连续邻接矩阵 `A_learned`
- **AND** `A_learned` 的对角线 MUST 为 `0`
- **AND** 邻接方向 MUST 按 `A[parent, child]` 解释

### Requirement: 模块 2 无监督因果学习 loss

模块 2 SHALL 为正式训练提供无监督因果学习 loss，不得依赖真实因果矩阵作为训练监督。

#### Scenario: 计算训练用 loss

- **WHEN** 模块 2 接收 Cycle/seasonal feature 并输出重构结果
- **THEN** MUST 能计算 reconstruction loss
- **AND** MUST 能计算归一化 DAG acyclicity loss
- **AND** MUST 能计算归一化 L1 sparsity loss

#### Scenario: 不泄漏真实因果图

- **WHEN** `S-DeCI` 正式训练模块 2
- **THEN** 模块 2 loss MUST NOT 使用 `A_true`、`A_structure_true` 或任何真实因果矩阵
- **AND** 真实因果矩阵若存在 MUST 仅用于独立实验后的指标或可视化诊断

### Requirement: 模块 2 支持可配置训练权重

模块 2 SHALL 允许训练流程配置 reconstruction、DAG acyclicity 和 L1 sparsity 的 loss 权重。

#### Scenario: 应用 loss 权重

- **WHEN** 训练流程读取模块 2 辅助 loss
- **THEN** 系统 MUST 能分别应用 reconstruction、DAG acyclicity 和 L1 sparsity 的权重
- **AND** 这些权重 MUST 可通过配置参数调整

#### Scenario: 支持关闭模块 2

- **WHEN** 用户通过配置关闭模块 2 或将其总权重设为 `0`
- **THEN** `S-DeCI` MUST 能退化为仅使用模块 1 Cycle/seasonal 分类的训练路径
- **AND** 训练流程 MUST 继续跑通
