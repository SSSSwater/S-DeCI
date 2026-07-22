## ADDED Requirements

### Requirement: 因果显著性互补视图

系统 SHALL 在显式启用时从标准模块 3 输出和模块 2 分类图构造训练期互补视图；该视图不得改变模块 2 的时序因果学习输入或产生第二张因果图。

#### Scenario: 构造标准和互补视图
- **GIVEN** `use_causal_complementary_learning == 1`
- **AND** 模块 2 输出 `A_cls[parent, child]`，节点特征 `C` 为 `[B, N, D]`
- **WHEN** `S-DeCI` 在训练模式执行 forward
- **THEN** 系统 MUST 先用完整 `C` 和 `A_cls` 计算标准模块 3/4 输出
- **AND** MUST 从有向信息流和标准节点双曲表征构造每样本 ROI 显著性
- **AND** MUST 仅遮挡送入共享模块 3 的节点特征，得到 `[B, N, D]` 的互补特征
- **AND** MUST 使用同一份 `A_cls` 和共享模块 3/4 参数计算互补输出
- **AND** MUST NOT 重新调用模块 2 或用遮挡特征重学因果图

#### Scenario: 根据有向图和语义活跃度计算显著性
- **GIVEN** 标准节点双曲表示为 `H_gcn`，`A_cls` 为 `[N, N]` 或 `[B, N, N]`
- **WHEN** 系统计算节点显著性
- **THEN** 拓扑项 MUST 同时考虑节点的出边和入边绝对强度
- **AND** 语义项 MUST 来自标准视图节点切空间表征的活跃度或等价任务相关量
- **AND** 系统 MUST 用 `causal_salience_topology_weight` 融合两项
- **AND** mask 选择分数 MUST 停止梯度，避免模型通过操纵显著性绕过遮挡

#### Scenario: 关闭互补视图保持兼容
- **GIVEN** `use_causal_complementary_learning == 0`
- **WHEN** 模型训练或评估
- **THEN** 系统 MUST 只计算标准模块 3/4 分支
- **AND** MUST 不产生互补分支计算开销
- **AND** MUST 将互补 loss 与诊断以零值或明确未启用状态暴露给训练日志

### Requirement: 动态 ROI 遮挡日程

系统 SHALL 使用可配置的动态 Gumbel-top-k 或等价随机化 top-k 机制遮挡高显著 ROI，并在 warm-up 后渐进增大遮挡比例。

#### Scenario: 训练期渐进遮挡
- **GIVEN** 互补视图已启用
- **AND** 当前 epoch 大于 `causal_complementary_warmup_epochs`
- **WHEN** 系统采样遮挡 ROI
- **THEN** 系统 MUST 使用 `causal_complementary_max_mask_ratio` 限定最大遮挡比例
- **AND** MUST 在 `causal_complementary_ramp_epochs` 内渐进增加当前比例
- **AND** 每个样本 MUST 遮挡由其显著性分布和随机扰动共同确定的 ROI
- **AND** 模块 2 的输入时间序列与 `A_cls` MUST 保持完整、不被遮挡

#### Scenario: warm-up 期间不遮挡
- **GIVEN** 当前 epoch 不大于 `causal_complementary_warmup_epochs`
- **WHEN** 模型训练
- **THEN** 互补分支 MUST 不施加 ROI 遮挡
- **AND** 系统 MUST 记录当前 mask ratio 为零

### Requirement: 双曲视图一致性

系统 SHALL 支持将标准与互补图级 Poincare 表征之间的距离作为独立可选损失，不增加默认额外分类监督。

#### Scenario: 计算 Poincare 一致性损失
- **GIVEN** 互补视图已启用
- **AND** `causal_complementary_view_loss_weight > 0`
- **WHEN** 训练流程聚合总 loss
- **THEN** 系统 MUST 计算标准 `z_global` 与互补 `z_global` 的 Poincare distance 平方均值
- **AND** MUST 仅以该权重将其加入总 loss
- **AND** MUST 记录未加权和加权后的 loss

#### Scenario: 一致性损失独立关闭
- **GIVEN** `causal_complementary_view_loss_weight == 0`
- **WHEN** 互补视图启用
- **THEN** 系统 MAY 仍计算互补输出和 prototype 更新一致性权重
- **AND** MUST 不将视图距离加入总 loss

### Requirement: BrainCL 对齐监督可独立消融

系统 SHALL 提供默认关闭的双向 InfoNCE 与遮挡分支 HPEC energy CE，使互补分支能够在显式配置时参与共享模块 3/4 的梯度优化，同时不改变已验证默认路线。

#### Scenario: 启用双向实例对比
- **GIVEN** `causal_complementary_instance_loss_weight > 0`
- **AND** 标准与互补图表示的 batch 大小至少为 2
- **WHEN** 训练流程计算互补监督
- **THEN** 系统 MUST 在同一受试者的标准/互补表示之间构造正对
- **AND** MUST 将 batch 内其他受试者作为负对计算双向 InfoNCE
- **AND** MUST 使用 `causal_complementary_instance_temperature` 控制相似度温度
- **AND** MUST 记录未加权和加权后的 InfoNCE

#### Scenario: 启用遮挡分支类别监督
- **GIVEN** `causal_complementary_masked_ce_weight > 0`
- **WHEN** 共享模块 4 得到互补视图 HPEC energy
- **THEN** 系统 MUST 以负 energy 作为类别 logits 计算训练标签交叉熵
- **AND** MUST 不在验证或测试阶段使用真实标签参与预测
- **AND** MUST 记录未加权和加权后的遮挡分支 CE

#### Scenario: 论文对齐监督默认关闭
- **GIVEN** 双向 InfoNCE 和遮挡分支 CE 权重均为 0
- **WHEN** 模型训练
- **THEN** 两项监督 MUST 对总 loss 零贡献
- **AND** 现有可靠 TP EMA 默认路线 MUST 保持兼容

### Requirement: 互补学习可诊断

系统 SHALL 为互补学习输出 TensorBoard 和最终可视化诊断。

#### Scenario: 记录互补诊断
- **WHEN** 互补视图在训练中执行
- **THEN** 系统 MUST 记录当前 mask ratio、拓扑显著性熵、语义显著性熵和标准/互补 Poincare 距离
- **AND** 最后 epoch 的可视化 MUST 能区分标准和互补表征
- **AND** 测试标签仅可用于可视化颜色，不得参与 mask、图学习或预测
