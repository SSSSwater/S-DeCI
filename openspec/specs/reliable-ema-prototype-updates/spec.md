## Purpose

定义 HPEC 多 prototype 的可靠 true-positive 独立 EMA 更新机制，替换默认 Sinkhorn 均衡分配并保留其 legacy 对照。

## Requirements

### Requirement: 可靠 TP 独立 prototype 更新

系统 SHALL 支持 `hpec_prototype_update_mode=reliable_tp_ema`，只使用预测正确、高置信度且可选双视图一致的训练样本在无梯度上下文移动 prototype。

#### Scenario: 优化器更新后移动 prototype
- **GIVEN** 当前为训练 batch 且已完成 `optimizer.step()`
- **WHEN** 训练循环调用可靠 prototype 更新接口
- **THEN** 系统 MUST 仅接受预测正确且真实类概率达到 `hpec_reliable_confidence_threshold` 的样本
- **AND** 互补视图启用时 MUST 应用 `hpec_reliable_view_consistency_threshold`
- **AND** MUST 按真实类内最低 HPEC energy prototype 分配样本
- **AND** MUST 用 EMA、初始化 anchor 和 Poincare 半径壳约束更新 prototype
- **AND** MUST NOT 将该移动加入总 loss 或执行第二次 backward

#### Scenario: 验证和测试冻结 prototype
- **GIVEN** 当前不是训练 batch
- **WHEN** 模型执行 forward
- **THEN** 系统 MUST 不更新 prototype

### Requirement: Sinkhorn legacy 与更新诊断

系统 SHALL 保留 `sinkhorn_ema` 和 `none` 模式，并记录可靠 TP 更新覆盖度。

#### Scenario: 选择更新模式
- **WHEN** 用户设置 `hpec_prototype_update_mode`
- **THEN** 系统 MUST 支持 `reliable_tp_ema`、`sinkhorn_ema` 与 `none`
- **AND** `reliable_tp_ema` MUST 不调用 Sinkhorn
- **AND** 系统 MUST 记录可靠 TP 比例、每 prototype 更新数、未更新数、EMA 位移和 assignment entropy
