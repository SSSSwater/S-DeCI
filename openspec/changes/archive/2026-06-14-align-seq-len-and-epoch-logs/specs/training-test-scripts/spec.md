## MODIFIED Requirements

### Requirement: 训练指标打印

训练流程 SHALL 在按 `print_metric_every` 打印 epoch 结果时，以可读且紧凑的格式展示 loss 与指标。

#### Scenario: 按类别分行打印 epoch 结果

- **GIVEN** `print_metric_every > 0` 或 `print_process` 启用
- **WHEN** 训练流程打印某个 epoch 的训练与验证结果
- **THEN** 输出 MUST 使用分隔线区分不同 epoch
- **AND** 输出 MUST 将 loss 字段放在 `[Loss]` 行
- **AND** 输出 MUST 将训练集指标放在 `[Train Metrics]` 行
- **AND** 输出 MUST 将验证集指标放在 `[Validation Metrics]` 行
- **AND** 输出 SHOULD 避免每个字段单独占一行，以控制日志高度

#### Scenario: 保留关键 loss 字段

- **WHEN** 训练流程打印 `[Loss]` 行
- **THEN** 输出 MUST 包含 `total_loss`、`cls_loss` 和 `val_loss`
- **AND** 若 S-DeCI 模块 2 或模块 4 启用，输出 SHOULD 包含对应 auxiliary loss、HPEC loss 或 prototype loss 字段

#### Scenario: 保留关键 metric 字段

- **WHEN** 训练流程打印 `[Train Metrics]` 和 `[Validation Metrics]`
- **THEN** 输出 MUST 包含 accuracy、precision、recall、macro F1 和 ROC AUC
