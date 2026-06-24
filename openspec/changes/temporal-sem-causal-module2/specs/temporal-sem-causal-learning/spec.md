## ADDED Requirements

### Requirement: 模块 1 提供时间序列因果输入

系统 SHALL 支持从 S-DeCI 模块 1 获取节点级时间序列输出，用于模块 2 的 temporal SEM 因果学习。

#### Scenario: 模块 1 输出时间序列
- **GIVEN** `causal_learning_target == "temporal_sem"`
- **WHEN** `S-DeCI.forward()` 完成模块 1 前端处理
- **THEN** 系统 MUST 缓存可供模块 2 使用的节点时间序列
- **AND** 时间序列 MUST 保留 batch、time、node 语义
- **AND** 系统 MUST 同时保留模块 3/4 所需的节点特征路径

#### Scenario: 模块 1 禁用时使用原始时间序列
- **GIVEN** `use_deci_module1 == 0`
- **AND** `causal_learning_target == "temporal_sem"`
- **WHEN** `S-DeCI.forward()` 接收 `[B, T, N]` 原始输入
- **THEN** 模块 2 temporal SEM MUST 使用归一化后的原始节点时间序列或其轻量投影
- **AND** 系统 MUST NOT 强制依赖 DeCI block 的中间输出

### Requirement: 时间序列预测式 SEM 因果学习器

系统 SHALL 提供可复用的 temporal SEM 因果学习器，用历史节点时间序列预测当前或下一步节点时间序列。

#### Scenario: 接收时间序列输入
- **GIVEN** 输入时间序列形状为 `[B, T, N]` 或 `[B, T, N, D]`
- **WHEN** temporal SEM learner 执行 forward
- **THEN** 系统 MUST 按配置的 `lag_order` 构造历史窗口
- **AND** MUST 输出预测结果 `X_hat`
- **AND** `X_hat` MUST 与预测目标时间片形状兼容

#### Scenario: 输出 contemporaneous 与 lagged 图
- **WHEN** temporal SEM learner 完成 forward
- **THEN** 系统 MUST 输出 `A0`
- **AND** `A0` MUST 形状为 `[N, N]`
- **AND** 系统 MUST 输出 `A_lag`
- **AND** `A_lag` MUST 形状为 `[L, N, N]`
- **AND** 所有图的对角线 MUST 为 0

### Requirement: 预测式 SEM 损失

系统 SHALL 使用预测式 SEM 损失训练 temporal SEM 因果图，不得以静态特征重构作为唯一目标。

#### Scenario: 计算 temporal SEM loss
- **WHEN** temporal SEM learner 完成一次 forward
- **THEN** 系统 MUST 计算 temporal prediction loss
- **AND** MUST 计算 `A0` 的 DAGMA DAG loss
- **AND** MUST 计算 `A0` 与 `A_lag` 的 sparsity loss
- **AND** MUST 计算 lag graph smoothness 或相邻 lag 稳定性 loss

#### Scenario: 不使用真实因果矩阵训练
- **WHEN** 系统训练 temporal SEM learner
- **THEN** loss MUST NOT 使用 `A_true`
- **AND** loss MUST NOT 使用 `A_structure_true`
- **AND** 若存在真实图，真实图 MUST 仅用于合成实验后的诊断和可视化

### Requirement: DAGMA 完整调度

系统 SHALL 对 `A0` 的 DAGMA log-det 约束使用阶段式调度，而不是固定权重 penalty。

#### Scenario: warmup 阶段
- **GIVEN** 当前 epoch 位于 DAGMA warmup 阶段
- **WHEN** 系统计算 temporal SEM auxiliary loss
- **THEN** prediction loss MUST 占主导
- **AND** DAGMA loss 权重 MUST 低于最终权重

#### Scenario: barrier/refine 阶段
- **GIVEN** 当前 epoch 进入 DAGMA barrier 或 refine 阶段
- **WHEN** 系统计算 temporal SEM auxiliary loss
- **THEN** DAGMA log-det loss 权重 MUST 按调度增强
- **AND** 稀疏 loss MUST 按调度增强
- **AND** 系统 MUST 记录当前阶段名称和各项有效权重

### Requirement: 样本残差图受限

系统 SHALL 仅允许样本级图以低秩、小幅度 residual graph 形式修正共享图。

#### Scenario: 生成低秩残差图
- **GIVEN** `use_sample_graph_residual == 1`
- **WHEN** temporal SEM learner 生成样本级图
- **THEN** 系统 MUST 生成形状为 `[B, N, N]` 的 `A_delta`
- **AND** `A_delta` MUST 使用低秩或等价受限参数化
- **AND** `A_delta` 的对角线 MUST 为 0
- **AND** `A_delta` MUST 经过幅度裁剪或缩放

#### Scenario: 关闭样本残差图
- **GIVEN** `use_sample_graph_residual == 0`
- **WHEN** temporal SEM learner 输出有效图
- **THEN** `A_effective` MUST 等于共享图或共享图融合结果
- **AND** 系统 MUST NOT 生成自由样本级完整邻接矩阵

### Requirement: 图诊断与稳定性输出

系统 SHALL 在训练、验证和测试后输出 temporal SEM 图学习诊断，作为 acc/AUC 之外的必要观察内容。

#### Scenario: 输出图学习指标
- **WHEN** 一个 epoch 完成
- **THEN** 日志 MUST 包含 temporal prediction loss
- **AND** MUST 包含 DAGMA penalty 或 DAGMA stage
- **AND** MUST 包含 `A0` 与 `A_lag` 的 sparsity
- **AND** MUST 包含图方向性或非对称度
- **AND** MUST 包含样本 residual graph 幅度

#### Scenario: 输出 fold 级稳定性
- **WHEN** 一个 fold 训练完成
- **THEN** 系统 MUST 保存或打印 learned graph 的 fold stability 指标
- **AND** MUST 保存 top-k edge frequency 或等价边稳定性统计
- **AND** MUST 记录 `A0` 与 `A_lag` 的主要边列表或 heatmap

### Requirement: 可视化 temporal SEM 中间量

系统 SHALL 使用现有 tensor visualization 工具可视化 temporal SEM 的关键中间量。

#### Scenario: 保存图可视化
- **WHEN** temporal SEM 训练完成一个 fold
- **THEN** 系统 MUST 保存 `A0` heatmap
- **AND** MUST 保存 `A_lag` heatmap 或 lag 聚合视图
- **AND** MUST 保存 `A_effective` heatmap
- **AND** 若启用样本残差图，MUST 保存 `A_delta` 示例 heatmap

#### Scenario: 保存预测可视化
- **WHEN** temporal SEM 训练完成一个 fold
- **THEN** 系统 SHOULD 保存预测目标与 `X_hat` 的对比图
- **AND** 可视化标题 MUST 标明张量维度和数据来源
