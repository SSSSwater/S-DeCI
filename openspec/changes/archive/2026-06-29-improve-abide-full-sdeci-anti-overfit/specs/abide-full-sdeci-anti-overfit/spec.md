## ADDED Requirements

### Requirement: ABIDE 默认启用完整 S-DeCI 四模块

系统 SHALL 在 ABIDE-120 的默认测试配置中启用完整的 `S-DeCI` 四模块主路径，而不是将 GCN fallback 作为默认最佳路径。

#### Scenario: 默认四模块路径

- **GIVEN** 用户运行 `test_abide_best_config.py` 且不额外覆盖模块开关
- **WHEN** 脚本构造 `S-DeCI` 实验参数
- **THEN** 默认参数 MUST 启用模块 1、模块 2、模块 3 和模块 4
- **AND** 默认参数 MUST 关闭站点对抗分支
- **AND** 默认参数 MUST 使用 ABIDE-120 输入长度 `seq_len=120`

#### Scenario: GCN fallback 仅作对照

- **WHEN** 用户显式关闭模块 2 或模块 3/4
- **THEN** GCN fallback MUST 仅作为消融或对照路径
- **AND** 它 MUST NOT 覆盖 ABIDE 默认最佳配置

### Requirement: ABIDE 训练入口支持去噪增强

系统 SHALL 为 ABIDE 主训练入口提供模块 1 去噪增强参数，用于降低短时序输入中的噪声记忆。

#### Scenario: 配置随机时间窗

- **WHEN** 用户为 ABIDE 训练入口设置随机裁剪参数
- **THEN** 训练时序输入 MUST 支持随机时间窗裁剪
- **AND** 验证与测试 MUST 保持确定性裁剪

#### Scenario: 配置 temporal dropout

- **WHEN** 用户启用 temporal dropout 或 ROI dropout
- **THEN** 模块 1 MUST 在训练期对输入施加相应扰动
- **AND** 扰动 MUST 仅作用于训练路径

### Requirement: ABIDE 训练入口支持因果稳定化

系统 SHALL 在 ABIDE 默认训练路径中允许模块 2 使用更保守的因果图约束，以优先学习稳定共享因果关系。

#### Scenario: 稳定因果默认

- **WHEN** ABIDE 默认配置启用模块 2
- **THEN** 模块 2 MUST 使用较小的样本残差图自由度
- **AND** 模块 2 MUST 支持稳定性约束权重
- **AND** 模块 2 MUST 继续输出 DAG 和 L1 辅助 loss

#### Scenario: 不使用真实因果监督

- **WHEN** ABIDE 默认路径训练模块 2
- **THEN** 训练 MUST NOT 使用真实因果矩阵作为监督
- **AND** 真实因果矩阵若存在 MUST 仅用于独立诊断或可视化

### Requirement: ABIDE 训练入口支持双曲和原型正则

系统 SHALL 为 ABIDE 的模块 3/4 提供更稳的双曲和原型正则，以减少训练集记忆。

#### Scenario: 双曲半径诊断

- **WHEN** ABIDE 默认配置启用模块 3
- **THEN** 系统 MUST 记录 `z_global` 的半径或范数诊断
- **AND** 系统 SHOULD 支持 edge dropout

#### Scenario: 多 prototype warm-start

- **WHEN** ABIDE 默认配置启用模块 4
- **THEN** prototype MUST 支持训练 fold 内 warm-start
- **AND** prototype loss MUST 支持多 prototype 的类内多样性和类间分离

### Requirement: ABIDE 默认脚本输出更可解释的训练诊断

系统 SHALL 在 ABIDE 默认测试脚本中输出更详细的训练诊断，帮助判断过拟合来源。

#### Scenario: 输出模块路径与损失

- **WHEN** ABIDE 默认脚本运行训练
- **THEN** 日志 MUST 显示当前模块开关
- **AND** 日志 MUST 包含模块 1、模块 2、模块 3/4 和 HPEC 相关 loss

#### Scenario: 输出中间量可视化

- **WHEN** ABIDE 默认脚本显式开启可视化
- **THEN** 系统 MUST 保存 train/test 中间量 heatmap
- **AND** 系统 MUST 保存最终 epoch 的 train/test t-SNE
- **AND** t-SNE MUST 区分 train/test 样式并用颜色区分 label
