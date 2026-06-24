## ADDED Requirements

### Requirement: 模块 3 支持 batch 级 adjacency

模块 3 SHALL 同时支持全局 adjacency `[N, N]` 和样本级 adjacency `[B, N, N]`。

#### Scenario: 使用全局 adjacency

- **GIVEN** `adjacency` 的形状为 `[N, N]`
- **WHEN** 模块 3 执行 HGCN forward
- **THEN** 模块 3 MUST 将同一 adjacency 用于 batch 内所有样本
- **AND** 该路径 MUST 与当前模块 2 `A_learned` 输入兼容

#### Scenario: 使用 batch adjacency

- **GIVEN** `cycle_features` 的形状为 `[B, N, D]`
- **AND** `adjacency` 的形状为 `[B, N, N]`
- **WHEN** 模块 3 执行 HGCN forward
- **THEN** 模块 3 MUST 对每个样本使用其对应的 adjacency
- **AND** 输出 `H_gcn` 的形状 MUST 保持 `[B, N, hgcn_hidden_dim]`
- **AND** 输出 `z_global` 的形状 MUST 保持 `[B, hgcn_hidden_dim]`

### Requirement: 模块 3 规范化样本相关矩阵图

模块 3 SHALL 在使用 sample correlation adjacency 时执行数值清理和图归一化。

#### Scenario: 清理相关矩阵

- **GIVEN** 输入 adjacency 来自样本相关系数矩阵
- **WHEN** 模块 3 归一化 adjacency
- **THEN** 模块 3 MUST 将 NaN 或 Inf 替换为有限值
- **AND** MUST 支持按配置处理负相关，至少包括 `abs`、`positive` 和 `raw`
- **AND** 默认模式 MUST 为 `abs`

#### Scenario: 加入 self-loop 并归一化

- **GIVEN** 模块 3 配置 `hgcn_add_self_loop == 1`
- **WHEN** adjacency 进入图传播
- **THEN** 模块 3 MUST 对 `[N, N]` 和 `[B, N, N]` 两种 adjacency 都支持添加 self-loop
- **AND** MUST 对两种 adjacency 都支持 `row`、`sym` 和 `none` 归一化方式

#### Scenario: 拒绝错误形状

- **GIVEN** adjacency 既不是 `[N, N]` 也不是 `[B, N, N]`
- **WHEN** 模块 3 forward 被调用
- **THEN** 模块 3 MUST 以清晰错误失败
