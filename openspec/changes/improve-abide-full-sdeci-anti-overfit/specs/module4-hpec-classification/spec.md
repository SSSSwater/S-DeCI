## ADDED Requirements

### Requirement: HPEC prototype 支持训练 fold 内 warm-start

模块 4 SHALL 支持仅使用训练 fold 表示进行 prototype warm-start，避免随机初始化导致早期训练不稳定。

#### Scenario: 使用训练表示初始化 prototype

- **GIVEN** `hpec_data_init == 1`
- **AND** 当前数据来自训练 fold
- **WHEN** 模块 4 执行 warm-start
- **THEN** prototype MUST 只使用训练 fold 的 `z_global` 或 `z_tangent` 表示初始化
- **AND** prototype 初始化 MUST 使用训练标签
- **AND** prototype 初始化 MUST NOT 使用验证或测试样本

#### Scenario: 延迟或分阶段初始化

- **WHEN** 模块 4 配置 warm-start 延迟步数或初始化步数
- **THEN** 系统 MUST 在达到条件后更新 prototype 初始化
- **AND** 更新后 prototype MUST 位于 Poincare Ball 有效区域内

### Requirement: HPEC energy 支持 ABIDE margin 与半径约束

模块 4 SHALL 支持用于 ABIDE 抗过拟合的 energy margin 和 prototype 半径约束。

#### Scenario: 类间 margin

- **WHEN** `hpec_margin > 0`
- **THEN** HPEC loss MUST 约束真实类别 energy 低于异类 energy 至少一个 margin
- **AND** margin 约束 MUST 能参与 autograd

#### Scenario: prototype 半径约束

- **WHEN** `lambda_hpec_radius_reg > 0`
- **THEN** 模块 4 MUST 计算 prototype 半径正则
- **AND** 正则 MUST 限制 prototype 远离数值不稳定边界

### Requirement: HPEC 诊断输出包含样本-原型匹配

模块 4 SHALL 输出样本与 prototype 的匹配诊断，帮助分析训练集记忆与测试集不可分问题。

#### Scenario: 输出 prototype assignment

- **WHEN** 模块 4 完成 forward
- **THEN** 系统 MUST 能读取每个样本最匹配的 prototype id
- **AND** 系统 SHOULD 能区分真实类别内最匹配 prototype 与全局最匹配 prototype

#### Scenario: 可视化 prototype energy

- **WHEN** 用户显式开启可视化
- **THEN** 系统 SHOULD 保存 prototype-level energy heatmap
- **AND** 标题 MUST 能标明 batch label 与 prediction 来源
