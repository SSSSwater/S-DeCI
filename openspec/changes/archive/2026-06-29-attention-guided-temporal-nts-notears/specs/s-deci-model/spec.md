## ADDED Requirements

### Requirement: S-DeCI 可选择 attention-guided 模块 2

`S-DeCI` SHALL 支持选择 attention-guided temporal NTS-NOTEARS 作为模块 2 因果图学习方法，同时保持现有 temporal NTS-NOTEARS 路径可回退。

#### Scenario: 初始化 attention-guided learner

- **GIVEN** `use_causal_module2 == 1`
- **AND** 用户选择 `causal_graph_method == "attn_nts_notears"` 或等价配置
- **WHEN** `S-DeCI` 初始化模块 2
- **THEN** 模型 MUST 初始化 attention-guided temporal causal learner
- **AND** learner MUST 接收 `channel`、`temporal_lag_order`、attention head 数和 head dim 等必要配置

#### Scenario: 保持旧方法可回退

- **GIVEN** 用户未选择 `attn_nts_notears`
- **WHEN** `S-DeCI` 初始化模块 2
- **THEN** 系统 MUST 保持现有 temporal NTS-NOTEARS 路径可用
- **AND** attention-guided 方法失败或验证效果不佳时 MUST 能通过配置切回现有路径

### Requirement: S-DeCI 将 A_lag_mean 作为 attention-guided 主图

`S-DeCI` SHALL 在 attention-guided 模块 2 启用时，优先使用 `A_lag.mean(dim=0)` 作为 learned causal graph。

#### Scenario: 模块 3 使用 attention-guided learned graph

- **GIVEN** attention-guided 模块 2 已输出 `A_lag`
- **AND** `classification_graph_source` 为 `learned` 或 `causal`
- **WHEN** 模块 3 解析 adjacency
- **THEN** 模块 3 MUST 使用 `A_lag.mean(dim=0)` 或等价 `a_effective`
- **AND** 模块 3 MUST NOT 默认使用 `A0`

#### Scenario: 与样本相关矩阵融合

- **GIVEN** attention-guided 模块 2 已输出 learned graph
- **AND** batch 提供 `correlation_matrix`
- **AND** `classification_graph_source == "blend"`
- **WHEN** 模块 3 解析 adjacency
- **THEN** 系统 MUST 使用现有 `module2_sample_correlation_blend` 语义融合 learned graph 与样本相关矩阵
- **AND** 融合后的 `A_cls` MUST 保持可微地连接到模块 2 learned graph

#### Scenario: 分类图尺度可控

- **GIVEN** attention-guided 模块 2 的 `A_lag` 来自 parent 维度 softmax
- **WHEN** 系统构造传给模块 3 的 learned adjacency
- **THEN** 系统 SHOULD 支持 `temporal_attention_graph_scale`
- **AND** 默认 SHOULD 为 `1.0`
- **AND** 原始 `A_lag` MUST 保留用于可解释可视化和稀疏约束

### Requirement: 分类 loss 可回传到 attention-guided 图参数

`S-DeCI` SHALL 保持分类 loss 经模块 3 使用的 learned adjacency 回传到 attention-guided 模块 2 图学习参数。

#### Scenario: 单次 backward 联合训练

- **GIVEN** 模块 2、模块 3 和模块 4 均已启用
- **AND** 模块 3 使用 attention-guided learned graph 或其 blend 图
- **WHEN** 训练循环执行 `total_loss.backward()`
- **THEN** 分类 loss MUST 能回传到 attention-guided 模块 2 的结构门控参数
- **AND** 系统 MUST NOT 新增阻断分类 loss 到模块 2 learned graph 的开关

### Requirement: S-DeCI 缓存 attention-guided 中间量

`S-DeCI` SHALL 缓存 attention-guided 模块 2 的关键中间量，供可视化、日志和诊断使用。

#### Scenario: 缓存图结构与诊断量

- **WHEN** attention-guided 模块 2 完成 forward
- **THEN** `S-DeCI` MUST 缓存 `A_lag`、`A0`、`A_lag_mean` 和最终分类图 `A_cls`
- **AND** `S-DeCI` SHOULD 缓存 attention entropy、gate mass、graph mass 和 directionality 诊断量

#### Scenario: 可视化不泄漏测试标签

- **GIVEN** 系统对测试集保存 attention-guided 中间量可视化
- **WHEN** `S-DeCI.forward()` 执行
- **THEN** 测试集真实 label MUST NOT 作为模型输入
- **AND** 测试集 label 只能用于可视化标题、t-SNE 颜色或训练后诊断对照
