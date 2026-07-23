## ADDED Requirements

### Requirement: 可靠 TP 多原型 EMA 更新

系统 SHALL 提供不依赖总 loss 的可靠 true-positive 样本 EMA 更新，用于移动 HPEC 的每类多个 prototype。

#### Scenario: 筛选可靠训练样本
- **GIVEN** `hpec_prototype_update_mode == "reliable_tp_ema"`
- **AND** 当前为训练 batch 且已完成标准视图 forward
- **WHEN** 系统筛选 prototype 更新样本
- **THEN** 样本 MUST 同时满足训练标签为目标类、最终标准 logits 预测正确和真实类概率不低于 `hpec_reliable_confidence_threshold`
- **AND** 更新 MUST 不依赖互补视图、companion embedding 或视图一致性阈值
- **AND** 验证、测试和推理样本 MUST 不参与筛选或更新

#### Scenario: 可靠样本独立移动 prototype
- **GIVEN** 一个类-原型对拥有不少于 `hpec_reliable_min_samples` 的可靠样本
- **WHEN** 训练 batch 完成 `optimizer.step()` 后更新 prototype
- **THEN** 系统 MUST 在 `torch.no_grad()` 或等价无梯度上下文中，根据同类最小 energy prototype 分配计算可靠样本的切空间中心
- **AND** MUST 将该中心与初始化锚点混合后按 `hpec_reliable_ema_alpha` 执行 EMA
- **AND** MUST 在映回 Poincare Ball 前执行半径壳投影或等价有效区域约束
- **AND** 更新 MUST 不依赖额外 prototype loss，也不得执行第二次 backward

#### Scenario: 可靠样本不足时保持 prototype
- **GIVEN** 一个类-原型对的可靠样本数小于 `hpec_reliable_min_samples`
- **WHEN** 系统执行 prototype 更新
- **THEN** 对应 prototype MUST 保持原值
- **AND** 系统 MUST 记录该 prototype 未更新的原因和次数

### Requirement: Sinkhorn legacy 回退

系统 SHALL 保留 Sinkhorn EMA 作为明确 legacy 对照，而默认可靠 TP EMA 路径不得调用 Sinkhorn 分配。

#### Scenario: 切换更新模式
- **GIVEN** 用户设置 `hpec_prototype_update_mode`
- **WHEN** 模型初始化
- **THEN** 系统 MUST 接受 `reliable_tp_ema`、`sinkhorn_ema` 和 `none`
- **AND** `reliable_tp_ema` MUST 不调用 `balanced_sinkhorn_assignment`
- **AND** `sinkhorn_ema` MUST 保持旧有均衡分配行为以支持消融
- **AND** `none` MUST 冻结 prototype 的数据驱动更新

### Requirement: 原型更新质量诊断

系统 SHALL 输出可靠 TP EMA 的更新质量，便于发现类别或 prototype 坍缩。

#### Scenario: 输出更新统计
- **WHEN** 训练 batch 结束
- **THEN** 系统 MUST 记录可靠 TP 比例、每类可靠样本数、每个 prototype 分配数、每个 prototype 的 EMA 位移和未更新计数
- **AND** MUST 记录同类 prototype assignment entropy 或等价覆盖度指标
- **AND** 这些量 MUST 写入 TensorBoard，且不参与默认训练 loss
