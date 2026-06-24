## ADDED Requirements

### Requirement: S-DeCI 在模块 2 关闭时使用样本相关矩阵

`S-DeCI` SHALL 在 `use_causal_module2=0` 且 `use_hgcn_module3=1` 时，使用输入 batch 对应的样本相关系数矩阵作为模块 3 adjacency。

#### Scenario: 模块 2 关闭且模块 3 开启

- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 接收到 `correlation_matrix`
- **THEN** 模型 MUST 将 `correlation_matrix` 传入模块 3
- **AND** 模型 MUST NOT 初始化或调用模块 2 因果学习器
- **AND** 模型 MUST NOT 计算 reconstruction、DAG 或 L1 auxiliary loss

#### Scenario: 模块 2 关闭但缺少相关矩阵

- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 未接收到 `correlation_matrix`
- **THEN** 模型 MUST 以清晰错误失败
- **AND** 错误信息 MUST 说明模块 3 需要 sample correlation adjacency 或启用模块 2

#### Scenario: 模块 2 开启时保持原行为

- **GIVEN** `use_causal_module2 == 1`
- **WHEN** `S-DeCI.forward()` 执行
- **THEN** 模型 MUST 继续使用模块 2 产生的 `A_learned` 作为模块 3 adjacency
- **AND** 模型 MUST 继续暴露模块 2 auxiliary loss
- **AND** 输入的 `correlation_matrix` MUST NOT 替代 `A_learned`

### Requirement: S-DeCI 缓存相关矩阵回退路径诊断量

`S-DeCI` SHALL 在使用样本相关矩阵作为 adjacency 时缓存该图结构，供可视化和调试使用。

#### Scenario: 缓存 sample correlation adjacency

- **GIVEN** `S-DeCI` 使用样本相关矩阵进入模块 3
- **WHEN** forward 完成
- **THEN** 模型 MUST 能读取本次使用的 sample correlation adjacency
- **AND** 缓存 MUST 不改变 `S-DeCI.forward()` 的主返回值

#### Scenario: 保留模块 4 分类路径

- **GIVEN** `use_hpec_module4 == 1`
- **AND** 模块 2 关闭但模块 3 使用样本相关矩阵成功运行
- **WHEN** 模型计算分类输出
- **THEN** 模块 4 MUST 继续使用模块 3 输出的 `z_global` 计算 HPEC energy
- **AND** `Loss_HPEC` MUST 继续作为 primary loss
