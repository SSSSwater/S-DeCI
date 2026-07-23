## MODIFIED Requirements

### Requirement: 模块 4 直接使用 Poincare 图级表示

模块 4 SHALL 直接接收模块 3 输出的 Poincare `z_global`，并使用同一 Poincare 几何中的 HPEC 多原型能量生成双曲分类证据。

#### Scenario: HPEC 计算不经过跨流形 bridge
- **GIVEN** 模块 3 已输出 `z_global`，形状为 `[B,H]`
- **WHEN** 模块 4 计算类别能量
- **THEN** `z_global` 与类别 prototype MUST 位于同一 Poincare Ball
- **AND** 系统 MUST NOT 把 Lorentz-to-Poincare bridge、MAC 或 HBR 作为正式路径必经步骤
- **AND** 系统 MUST 缓存类别能量、prototype 匹配、Poincare 半径和最终双曲证据用于诊断

### Requirement: 双曲原型证据与欧氏结构证据共同决策

模块 4 SHALL 将 HPEC energy 转换为双曲分类证据，并与欧氏局部结构分类证据在 logits 层进行 dual-view evidence fusion。

#### Scenario: 默认融合预测
- **GIVEN** HPEC energy 为 `E`，欧氏分支 logits 为 `L_base`
- **WHEN** 系统生成最终预测
- **THEN** 双曲证据 MUST 由 `-E` 或其校准残差得到
- **AND** 最终 logits MUST 同时包含欧氏局部结构证据和双曲层级原型证据
- **AND** 预测类别、概率和测试指标 MUST 使用最终融合 logits
- **AND** 系统 MUST NOT 通过降低模块 3/4 到近零权重来伪造启用状态

### Requirement: LP 专用半径机制退出正式损失

正式模块 4 loss SHALL 不包含只服务于已退役 LP bridge 的 MAC/HBR 项。

#### Scenario: 计算正式总损失
- **WHEN** 模型使用当前 HGCN-HPEC 主路线训练
- **THEN** 总损失 MUST NOT 包含 `lp_hbr_weighted_loss`
- **AND** 训练入口 MUST NOT 要求 `mac_min_radius`、`mac_max_radius`、`hbr_safe_radius` 或 `hbr_loss_weight`
- **AND** Poincare 半径统计 MAY 作为无梯度诊断保留
