## Purpose

定义 S-DeCI 训练期因果显著性 ROI 遮挡互补学习，提升小样本 fMRI 图表征对少数 ROI 的鲁棒性，同时不污染模块 2 的时序因果发现。

## Requirements

### Requirement: 共享因果图的互补视图

系统 SHALL 在训练期可选地从标准模块 3 节点表征和 `A_cls[parent, child]` 构造 ROI 遮挡互补视图。

#### Scenario: 遮挡只发生在模块 3 输入
- **GIVEN** `use_causal_complementary_learning == 1`
- **WHEN** 模型执行训练 forward
- **THEN** 系统 MUST 先以完整时间序列得到标准 `A_cls`
- **AND** MUST 仅遮挡模块 3 输入节点特征
- **AND** MUST 用共享模块 3/4 参数和同一 `A_cls` 计算互补表征
- **AND** MUST NOT 为互补视图重新运行模块 2 或学习第二张因果图

### Requirement: 动态显著性遮挡与双曲一致性

系统 SHALL 以有向信息流和节点语义活跃度融合的 detach 显著性执行 warm-up 后的动态 top-k 遮挡，并可选计算 Poincare 一致性损失。

#### Scenario: 训练期渐进遮挡
- **GIVEN** 当前 epoch 超过 `causal_complementary_warmup_epochs`
- **WHEN** 系统选择互补 ROI mask
- **THEN** 遮挡比例 MUST 不超过 `causal_complementary_max_mask_ratio`
- **AND** MUST 在 `causal_complementary_ramp_epochs` 内渐进增长
- **AND** 只有 `causal_complementary_view_loss_weight > 0` 时，标准与互补图点 Poincare 距离才可加入总 loss

#### Scenario: 评估期保持标准预测
- **GIVEN** 模型处于验证、测试或推理模式
- **WHEN** 未显式收集可视化
- **THEN** 系统 MUST 跳过互补分支
- **AND** 指标 MUST 只使用标准视图 logits
