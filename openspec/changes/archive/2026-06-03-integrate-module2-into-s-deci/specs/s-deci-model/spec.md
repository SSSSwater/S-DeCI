## ADDED Requirements

### Requirement: S-DeCI 接入模块 2 因果图学习

`S-DeCI` SHALL 在模块 1 产生 Cycle/seasonal feature 后接入模块 2 因果图学习，并且模块 2 输入 MUST 保持 `[B, N, d_model]` 的节点特征语义。

#### Scenario: 使用 Cycle feature 作为模块 2 输入

- **WHEN** `S-DeCI` 执行 forward 并完成 DeCI block 分解
- **THEN** 模型 MUST 从 block 内部或等价路径获得形状为 `[B, N, d_model]` 的 Cycle/seasonal feature
- **AND** 模型 MUST 将该 feature 输入模块 2 因果图学习组件

#### Scenario: 多层 Cycle feature 聚合

- **WHEN** `S-DeCI` 使用多个 DeCI block
- **THEN** 模型 MUST 支持将多个 block 的 Cycle/seasonal feature 聚合为模块 2 输入
- **AND** 默认聚合方式 MUST 与当前 seasonal logits 聚合语义保持一致

### Requirement: S-DeCI 分类仍仅使用 Cycle 分支

`S-DeCI` SHALL 保持最终分类输出只来自 Cycle/seasonal 分类分支，不得在当前阶段使用模块 2 的因果图或模块 3/HGCN 结果参与分类。

#### Scenario: 二分类输出保持兼容

- **WHEN** `configs.classes == 2`
- **THEN** `S-DeCI.forward()` MUST 返回形状为 `[B, 1]` 的分类概率
- **AND** 返回值 MUST 与现有 MSE 二分类训练路径兼容
- **AND** 返回值 MUST NOT 直接使用因果图卷积分类结果

#### Scenario: 多分类输出保持兼容

- **WHEN** `configs.classes > 2`
- **THEN** `S-DeCI.forward()` MUST 返回形状为 `[B, classes]` 的 logits
- **AND** 返回值 MUST 与现有 CE 多分类训练路径兼容
- **AND** 返回值 MUST NOT 直接使用因果图卷积分类结果

### Requirement: S-DeCI 暴露模块 2 辅助损失

`S-DeCI` SHALL 在 forward 后暴露模块 2 的辅助损失和诊断量，使训练流程能够将因果学习 loss 纳入总 loss，同时不改变 `forward()` 的主返回值。

#### Scenario: 暴露因果学习 loss

- **WHEN** `S-DeCI` 开启模块 2 并完成一次 forward
- **THEN** 模型 MUST 能提供 reconstruction loss、DAG acyclicity loss 和 L1 sparsity loss
- **AND** 这些 loss MUST 能参与 PyTorch autograd 反向传播

#### Scenario: forward 返回值不变

- **WHEN** 训练流程调用 `y_hat = model(x_enc)`
- **THEN** `S-DeCI.forward()` MUST 只返回分类输出 `y_hat`
- **AND** 模块 2 的辅助 loss MUST 通过模型属性或方法读取

### Requirement: S-DeCI 关键逻辑中文注释

`S-DeCI` SHALL 在本次新增或改动的关键逻辑处提供简洁中文注释，必要英文关键词可以保留。

#### Scenario: 注释模块 2 接入逻辑

- **WHEN** 开发者查看 `models/S_DeCI.py`
- **THEN** 模块 2 初始化、Cycle feature 聚合、辅助 loss 缓存和可视化触发相关代码 MUST 带有中文注释
- **AND** 注释 MUST 说明当前阶段分类仍只使用 Cycle/seasonal 分支

#### Scenario: 保留必要英文关键词

- **WHEN** 注释中涉及 `Cycle`、`seasonal`、`causal graph`、`DAG` 或 `adjacency`
- **THEN** 注释 MAY 保留这些英文关键词
- **AND** 注释 MUST 便于中文阅读和后续维护
