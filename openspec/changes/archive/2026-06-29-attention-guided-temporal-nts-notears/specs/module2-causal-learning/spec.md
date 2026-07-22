## ADDED Requirements

### Requirement: 模块 2 支持 Attention-guided Temporal NTS-NOTEARS

模块 2 SHALL 支持一种 attention-guided temporal NTS-NOTEARS 方法，用于从时间序列历史窗口预测未来时间点，并输出显式跨时间因果图 `A_lag`。

#### Scenario: 使用历史窗口预测未来时间点

- **GIVEN** 模块 2 选择 attention-guided temporal NTS-NOTEARS 方法
- **WHEN** 输入时间序列形状为 `[B, T, N]` 或 `[B, T, N, D]`
- **THEN** 系统 MUST 使用 `x_{t-1}, ..., x_{t-L}` 预测 `x_t`
- **AND** 系统 MUST 输出预测值 `x_hat` 与目标 `target`
- **AND** `x_hat` 与 `target` 的形状 MUST 一致

#### Scenario: 输出跨时间主因果图

- **GIVEN** 模块 2 完成一次 forward
- **WHEN** 系统读取 attention-guided learner 的输出
- **THEN** 系统 MUST 输出 `A_lag: [lag_order, N, N]`
- **AND** `A_lag[l, parent, child]` MUST 表示过去第 `l+1` 个 lag 的 parent 脑区对未来 child 脑区的影响
- **AND** `A_lag` 的对角线 MUST 为 `0`

### Requirement: Attention 权重不得直接作为因果矩阵

模块 2 SHALL 将多头 attention 视为动态候选影响，而不是直接将 raw attention map 作为最终因果矩阵。

#### Scenario: attention 公式遵守跨时间方向语义

- **GIVEN** learner 正在使用多头 lag-window attention 预测 `x_child(t)`
- **WHEN** 系统计算第 `h` 个 head、第 `l` 个 lag、从 parent 到 child 的注意力
- **THEN** 系统 MUST 使用目标 child 节点历史摘要构造 `q_{h,child}(t)`
- **AND** 系统 MUST 使用 parent 节点在 `t-l-1` 或等价第 `l+1` 个 lag 的历史值构造 `k_{h,l,parent}(t)` 与 `v_{h,l,parent}(t)`
- **AND** 系统 MUST 按 `score_{h,l,parent->child}(t) = <q_{h,child}(t), k_{h,l,parent}(t)> / sqrt(d_h)` 计算 attention score
- **AND** attention 方向 MUST 表示 `parent` 的过去影响 `child` 的未来

#### Scenario: 使用结构门控聚合 attention

- **GIVEN** learner 已计算多头 lag-window attention
- **WHEN** 系统构造 `A_lag`
- **THEN** 系统 MUST 使用可学习结构门控 `G_lag` 调制 attention
- **AND** 系统 MUST 对 batch、time 和 head 维度聚合后得到稳定的 `A_lag`
- **AND** 模块 3 MUST NOT 直接读取 raw attention map 作为 adjacency

#### Scenario: A_lag 由 attention 与 gate 聚合得到

- **GIVEN** 系统已得到 `attn_{h,l,parent->child}(t)` 与 `gate_{h,l,parent->child}`
- **WHEN** 系统生成最终跨时间因果图
- **THEN** 系统 MUST 先计算 `edge_{h,l,parent->child}(t) = attn_{h,l,parent->child}(t) * gate_{h,l,parent->child}`
- **AND** 系统 MUST 通过 `A_lag[l,parent,child] = mean_{batch,time,head}(edge_{h,l,parent->child}(t))` 聚合得到 `A_lag`
- **AND** `A_lag` MUST 维持 `A[parent, child]` 的方向语义

#### Scenario: 暴露 attention 诊断而非替代图

- **WHEN** 训练日志或可视化读取 attention-guided 模块 2 诊断量
- **THEN** 系统 SHOULD 输出 attention entropy、gate mass 或等价诊断量
- **AND** 这些诊断量 MUST 与最终 `A_lag` 区分命名

### Requirement: A0 只表示同时间片残余依赖

模块 2 SHALL 保留 `A0: [N, N]` 作为同时间片残余依赖图，用于吸收历史窗口无法解释的同步残差，并承载 DAGMA/NOTEARS 式无环约束。

#### Scenario: A0 不作为主分类图

- **GIVEN** attention-guided 模块 2 输出 `A0` 和 `A_lag`
- **WHEN** 模块 3 或 GCN fallback 需要 learned causal graph
- **THEN** 系统 MUST 默认使用 `A_lag.mean(dim=0)`
- **AND** 系统 MUST NOT 默认使用 `A0` 作为模块 3 adjacency

#### Scenario: DAG 约束仅作用于 A0

- **GIVEN** 系统计算 attention-guided 模块 2 auxiliary loss
- **WHEN** 系统计算 DAGMA 或 NOTEARS 风格无环约束
- **THEN** DAG loss MUST 作用于 `A0`
- **AND** DAG loss MUST NOT 直接作用于 `A_lag`

#### Scenario: A0 与 A_lag 共同参与预测

- **GIVEN** learner 已通过 `A_lag` 从历史窗口得到初步预测
- **WHEN** 系统启用 `A0` 残余建模
- **THEN** `A0` MAY 对预测残差进行同时间片修正
- **AND** 该修正 MUST NOT 读取真实未来目标作为输入

### Requirement: 模块 2 损失保持简洁

attention-guided 模块 2 SHALL 使用时序预测损失、稀疏损失、lag 平滑损失和 `A0` 无环损失组成 auxiliary loss。

#### Scenario: 计算 attention-guided auxiliary loss

- **GIVEN** attention-guided 模块 2 已输出 `x_hat`、`target`、`A_lag` 和 `A0`
- **WHEN** 系统计算模块 2 auxiliary loss
- **THEN** loss MUST 包含 `temporal_pred_loss`
- **AND** loss MUST 包含 `temporal_sparse_loss`
- **AND** loss MUST 包含 `temporal_smooth_loss`
- **AND** loss MUST 包含作用于 `A0` 的 `causal_dag_loss`
- **AND** loss MUST NOT 使用真实因果矩阵作为监督

### Requirement: attention-guided 因果图可视化

模块 2 SHALL 支持 attention-guided 因果学习过程的关键图可视化。

#### Scenario: 保存关键因果图

- **WHEN** 当前 fold 训练结束且启用中间量可视化
- **THEN** 系统 MUST 保存 `A_lag_mean`
- **AND** 系统 MUST 保存每个 lag 的 `A_lag[k]`
- **AND** 系统 MUST 保存 `A0`
- **AND** 系统 MUST 保存最终传入模块 3 的 `A_cls`

#### Scenario: 图标题标明 attention-guided 路径

- **WHEN** 系统生成 attention-guided 模块 2 可视化图
- **THEN** 图标题或文件名 SHOULD 标明 `attention-guided` 或 `attn_nts_notears`
- **AND** 对 `A0` 的标题 MUST 表明其为同时间片残余依赖图
