## ADDED Requirements

### Requirement: S-DeCI 模块 3 中间量可视化

系统 SHALL 支持显式可视化 `S-DeCI` 模块 3 的关键训练中间量。

#### Scenario: 可视化模块 3 双曲中间量

- **GIVEN** 用户显式开启 `S-DeCI` 因果和 HGCN 可视化
- **WHEN** 模型已完成一次包含模块 3 的 forward
- **THEN** 系统 MUST 能调用 `utils.tensor_visualization.visualize_tensors`
- **AND** MUST 能保存 `C_clipped`
- **AND** MUST 能保存 Poincare 投影结果或 `H0`
- **AND** MUST 能保存 `H_gcn`
- **AND** MUST 能保存 `z_global` 或 `logmap0(z_global)`

#### Scenario: 可视化模块 2 与模块 3 衔接

- **GIVEN** 用户显式开启模块 3 可视化
- **WHEN** 系统保存模块 3 heatmap
- **THEN** 输出 MUST 包含 `A_learned`
- **AND** 输出 MUST 包含 `A_learned - A_learned.T`
- **AND** 输出 MUST 能体现模块 2 因果图如何作为模块 3 图传播输入

#### Scenario: 默认不保存模块 3 可视化

- **GIVEN** 用户未显式开启可视化配置
- **WHEN** `S-DeCI` 执行常规训练、验证或推理
- **THEN** 系统 MUST NOT 默认保存模块 3 heatmap 图片
- **AND** 可视化逻辑 MUST NOT 改变 `S-DeCI.forward()` 返回值

### Requirement: 模块 3 可视化标题显示维度信息

模块 3 可视化 SHALL 延续现有 helper 的维度副标题规则，确保 3D 张量只显示 Batch0 或指定 batch。

#### Scenario: 显示模块 3 张量维度

- **GIVEN** 模块 3 中间量为 3D tensor
- **WHEN** 系统生成 heatmap
- **THEN** subplot 标题或副标题 MUST 显示原始 shape
- **AND** MUST 提示当前展示的 batch index

