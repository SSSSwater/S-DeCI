## ADDED Requirements

### Requirement: 训练入口暴露 attention-guided 模块 2 选择项

训练入口 SHALL 支持选择 attention-guided temporal NTS-NOTEARS 作为模块 2 方法。

#### Scenario: CLI 可选择新方法

- **WHEN** 用户运行 `run_cv.py --help` 或根目录测试脚本的 `--help`
- **THEN** 参数列表 MUST 支持 `causal_graph_method == "attn_nts_notears"` 或等价方法名
- **AND** 帮助文本 SHOULD 用中文说明其为“Attention-guided Temporal NTS-NOTEARS”

### Requirement: 训练入口仅暴露少量 attention 参数

训练入口 SHALL 仅暴露少量 attention-guided 模块 2 必需参数，避免日常训练命令过于复杂。

#### Scenario: 暴露核心 attention 参数

- **WHEN** 用户查看训练脚本参数
- **THEN** 训练入口 MUST 支持 attention head 数
- **AND** MUST 支持 attention head dim
- **AND** MUST 支持 attention dropout
- **AND** temporal 预测、稀疏、平滑和 DAG 权重 MUST 继续复用现有模块 2 参数

#### Scenario: 不为无效辅助损失新增窗口

- **WHEN** 用户查看 attention-guided 模块 2 相关参数
- **THEN** 训练入口 MUST NOT 新增 raw attention 对比损失、真实因果监督损失或额外 prototype 类辅助损失参数

### Requirement: 训练日志显示 attention-guided 诊断量

训练流程 SHALL 在按间隔打印 epoch 结果时显示 attention-guided 模块 2 关键诊断量。

#### Scenario: 打印图学习诊断

- **WHEN** attention-guided 模块 2 参与训练且到达日志打印间隔
- **THEN** 日志 MUST 至少打印 `dag_loss`
- **AND** MUST 打印 `A_lag` 或 learned graph 的 mass/directionality
- **AND** SHOULD 打印 attention entropy 或 gate mass
- **AND** `A0` 相关统计 MUST 与 `A_lag` 区分命名

### Requirement: 可视化区分 A0 与主因果图

训练入口 SHALL 在 attention-guided 路径可视化中明确区分 `A0` 与 `A_lag/A_cls`。

#### Scenario: 保存模块 2 图可视化

- **WHEN** 用户启用中间量可视化并完成一个 fold 训练
- **THEN** 输出图中 MUST 分别保存 `A0` 和 `A_lag_mean`
- **AND** 图标题或副标题 MUST 标明 `A0` 为“同时间片残余依赖”
- **AND** 图标题或副标题 MUST 标明 `A_lag_mean` 为“跨时间主因果图”

### Requirement: 根目录测试脚本支持新方法快速检查

根目录训练测试脚本 SHALL 支持以当前默认数据集配置快速检查 attention-guided 模块 2 是否能跑通训练。

#### Scenario: MDD 快速训练检查

- **WHEN** 用户运行 `test_mdd_best_config.py` 并显式选择 attention-guided 模块 2
- **THEN** 脚本 MUST 能完成至少一个 fold 的训练与测试
- **AND** MUST 输出 accuracy、precision、recall、macro F1 和 ROC AUC
- **AND** MUST 保存最终结果到既有 result 文件体系
