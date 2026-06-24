## ADDED Requirements

### Requirement: S-DeCI 因果学习中间量可视化

系统 SHALL 支持在 `S-DeCI` 接入模块 2 后显式可视化关键中间量，并且默认不改变训练、验证或推理行为。

#### Scenario: 可视化 Cycle 和因果矩阵

- **WHEN** 用户显式开启 `S-DeCI` 因果学习可视化
- **THEN** 系统 MUST 调用 `utils.tensor_visualization.visualize_tensors`
- **AND** MUST 能保存 Cycle/seasonal feature、重构特征 `C_hat`、重构误差、连续邻接矩阵 `A_learned` 和阈值化邻接矩阵

#### Scenario: 默认不执行可视化

- **WHEN** 用户未显式开启可视化配置或手动调用可视化方法
- **THEN** `S-DeCI` MUST NOT 在常规训练、验证或推理中保存图片
- **AND** 可视化逻辑 MUST NOT 改变模型 forward 返回值

#### Scenario: 三维张量显示 Batch0 提示

- **WHEN** 可视化 Cycle/seasonal feature 或 `C_hat` 等 3D 张量
- **THEN** helper MUST 默认只显示 Batch0
- **AND** subplot 标题或副标题 MUST 显示原始维度并提示当前展示的 batch index

### Requirement: S-DeCI 可视化输出可定位

系统 SHALL 允许用户配置或指定 `S-DeCI` 因果学习可视化输出位置。

#### Scenario: 保存到指定目录

- **WHEN** 用户提供可视化输出目录或保存路径
- **THEN** 系统 MUST 将 heatmap 图片保存到该位置
- **AND** 文件名 MUST 能区分 Cycle feature、重构结果、因果矩阵和差异矩阵

#### Scenario: 训练中限制保存频率

- **WHEN** 用户在训练中启用可视化
- **THEN** 系统 MUST 支持通过配置或手动调用限制保存频率
- **AND** 默认配置 MUST 避免每个 batch 都保存图片
