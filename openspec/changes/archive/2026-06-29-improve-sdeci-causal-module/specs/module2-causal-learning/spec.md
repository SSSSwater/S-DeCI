## ADDED Requirements

### Requirement: 模块 2 支持 DAGMA 风格 log-det 无环约束
模块 2 SHALL 支持通过配置选择 `dagma_logdet` 或等价 log-det / M-matrix 风格的 DAG penalty，用于在较大 ROI 图上稳定约束有向无环结构。

#### Scenario: 选择 DAGMA 风格方法
- **GIVEN** 用户设置 `causal_graph_method == "dagma_logdet"`
- **WHEN** 模块 2 初始化因果图学习器
- **THEN** 系统 MUST 使用 log-det / M-matrix 风格无环约束计算 DAG penalty
- **AND** penalty MUST 可参与 PyTorch autograd 反向传播
- **AND** 系统 MUST 保留 `nts_notears` 和 `dag_sampling` 作为可配置对照方法

#### Scenario: DAGMA 数值稳定
- **GIVEN** 模块 2 接收到形状为 `[B, N, F]` 的节点特征
- **WHEN** 系统计算 `dagma_logdet` penalty
- **THEN** 系统 MUST 对矩阵计算使用安全 margin 或 eps
- **AND** 系统 MUST 避免对不可逆矩阵直接求逆导致训练崩溃
- **AND** 诊断信息 MUST 暴露当前 DAG penalty 数值

### Requirement: 模块 2 输入标准化
模块 2 SHALL 在因果图学习器内部支持对输入节点特征 `C` 进行独立标准化，降低模块一输出尺度漂移对图学习的影响。

#### Scenario: 启用输入标准化
- **GIVEN** 用户设置 `causal_input_norm` 为非 `none` 值
- **WHEN** 模块 2 接收 `C: [B, N, F]`
- **THEN** 系统 MUST 在 reconstruction 和 adjacency 学习前对 `C` 执行对应标准化
- **AND** 标准化后的张量 MUST 保持 `[B, N, F]` 形状
- **AND** 模块 2 的输出 `C_hat` MUST 与标准化路径的训练目标一致

#### Scenario: 关闭输入标准化
- **GIVEN** 用户设置 `causal_input_norm == "none"`
- **WHEN** 模块 2 执行 forward
- **THEN** 系统 MUST 保持旧输入路径，不额外改变 `C` 的数值尺度

### Requirement: 模块 2 支持共享图加样本残差图
模块 2 SHALL 支持在全局共享 adjacency 之外生成可选的样本级残差 adjacency，使最终传给下游图模块的 `A_effective` 能表达样本差异。

#### Scenario: 样本残差图关闭
- **GIVEN** 用户设置 `use_sample_graph_residual == 0`
- **WHEN** 模块 2 完成 forward
- **THEN** 系统 MUST 输出共享图 `A_shared`
- **AND** `A_effective` MUST 等于共享图路径
- **AND** 系统 MUST 保持与旧版全局共享图行为兼容

#### Scenario: 样本残差图开启
- **GIVEN** 用户设置 `use_sample_graph_residual == 1`
- **WHEN** 模块 2 接收一个 batch 的节点特征
- **THEN** 系统 MUST 生成形状为 `[B, N, N]` 的样本残差图 `A_delta`
- **AND** 系统 MUST 生成形状为 `[B, N, N]` 的最终图 `A_effective`
- **AND** `A_effective` 的每个样本对角线 MUST 为 `0`
- **AND** `A_delta` MUST 受 L1 或 deviation 正则约束

#### Scenario: 样本残差图幅度受控
- **GIVEN** 样本残差图已启用
- **WHEN** 系统计算模块 2 辅助 loss
- **THEN** 系统 MUST 支持配置 `lambda_sample_graph_l1`
- **AND** 系统 MUST 支持配置 `lambda_sample_graph_deviation`
- **AND** 系统 MUST 支持配置 `sample_graph_delta_scale`

### Requirement: 模块 2 支持因果 loss 权重调度
模块 2 SHALL 支持对 reconstruction、DAG、L1 和样本残差图正则项使用训练 epoch 相关的动态权重。

#### Scenario: 使用 constant 调度
- **GIVEN** 用户设置 `causal_loss_schedule == "constant"`
- **WHEN** 训练流程计算模块 2 辅助 loss
- **THEN** 系统 MUST 使用用户配置的固定 loss 权重

#### Scenario: 使用 warmup 调度
- **GIVEN** 用户设置 `causal_loss_schedule == "warmup"`
- **WHEN** 当前 epoch 小于对应 warmup epoch
- **THEN** 系统 MUST 按调度比例缩放 DAG、L1 或样本残差图正则权重
- **AND** 当前 epoch 达到 warmup 结束后 MUST 使用完整配置权重

### Requirement: 模块 2 暴露扩展诊断信息
模块 2 SHALL 暴露足够诊断信息，用于判断因果图是否在更新、是否有方向性、是否存在样本级差异。

#### Scenario: 输出共享图诊断
- **GIVEN** 模块 2 完成一次 forward
- **WHEN** 训练流程读取模块 2 诊断字典
- **THEN** 诊断信息 MUST 包含 `A_shared` 或可视化等价项
- **AND** MUST 包含 DAG penalty、L1 loss 和 adjacency 方向性指标

#### Scenario: 输出样本残差图诊断
- **GIVEN** `use_sample_graph_residual == 1`
- **WHEN** 模块 2 完成一次 forward
- **THEN** 诊断信息 MUST 包含 `A_delta` 或其 batch 统计
- **AND** MUST 包含 `A_effective` 或其 batch 统计
- **AND** 可视化输出 SHOULD 区分共享图和样本图

### Requirement: 模块 2 更新说明文档
系统 SHALL 新增一份项目当前实现说明文档，用中文描述模块二改造后的方法选择、输入语义、loss 构成和回退路径。

#### Scenario: 新说明文档存在
- **GIVEN** 本变更实现完成
- **WHEN** 开发者查看项目文档目录
- **THEN** 系统 MUST 提供一份新的模块二更新说明文档
- **AND** 该文档 MUST 不覆盖 `docs/` 中作为初始参考的原始设计文档
- **AND** 文档 MUST 说明当前 feature-level DAG 与严格 NTS/DYNOTEARS 的区别
