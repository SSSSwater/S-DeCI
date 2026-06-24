## ADDED Requirements

### Requirement: 可视化样本相关矩阵 adjacency

S-DeCI 中间量可视化 SHALL 在模块 2 关闭且模块 3 使用样本相关矩阵时显示该 adjacency。

#### Scenario: 保存 sample correlation heatmap

- **GIVEN** 用户显式开启 `visualize_causal`
- **AND** `S-DeCI` 使用 sample correlation adjacency 进入模块 3
- **WHEN** 每个 fold 训练结束后保存中间量 heatmap
- **THEN** 可视化 MUST 包含 sample correlation adjacency
- **AND** 文件名 MUST 能区分 train/test 和 fold

#### Scenario: 与模块 3 中间量对照

- **GIVEN** sample correlation adjacency 已缓存
- **WHEN** 可视化模块 3 中间量
- **THEN** 系统 MUST 能同时显示 sample correlation adjacency、Module3 normalized adjacency、`H_gcn` 和 `z_global`
- **AND** 三维张量仍 MUST 按现有规则默认显示 Batch0 并提示维度

#### Scenario: 模块 2 开启路径保持原可视化

- **GIVEN** `use_causal_module2 == 1`
- **WHEN** 用户保存 S-DeCI 中间量可视化
- **THEN** 系统 MUST 继续显示 `A_learned`、`A_learned_binary` 和模块 2 reconstruction 相关中间量
- **AND** sample correlation adjacency MUST NOT 替代模块 2 可视化内容
