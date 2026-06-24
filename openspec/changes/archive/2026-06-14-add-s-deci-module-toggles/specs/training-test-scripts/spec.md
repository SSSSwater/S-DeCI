## ADDED Requirements

### Requirement: 训练入口暴露模块开关参数

训练入口 SHALL 暴露 `S-DeCI` 模块 1、模块 2、模块 3/4 的启用/禁用参数，并将其传递给模型配置。

#### Scenario: run_cv 暴露模块开关
- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST 包含 `use_deci_module1`
- **AND** 参数列表 MUST 包含 `use_causal_module2`
- **AND** 参数列表 MUST 包含 `use_hyperbolic_modules34`
- **AND** 这些参数的 help 文本 MUST 使用中文描述，必要关键词 MAY 保留英文

#### Scenario: 测试脚本传递模块开关
- **WHEN** 用户运行根目录训练测试脚本
- **THEN** 测试脚本 MUST 接收模块开关参数
- **AND** 测试脚本 MUST 将模块开关写入构造出的 experiment args
- **AND** 默认参数 MUST 保持当前已验证的 `S-DeCI` 主路径可运行

#### Scenario: 兼容旧模块 3 和模块 4 参数
- **GIVEN** 训练入口仍保留 `use_hgcn_module3` 或 `use_hpec_module4`
- **WHEN** 用户传入这些旧参数
- **THEN** 训练入口 MUST 将旧参数归一到模块 3/4 联合开关语义
- **AND** 不一致组合 MUST 清晰失败或被明确覆盖为 `use_hyperbolic_modules34` 的值

### Requirement: 训练流程覆盖模块开关组合

训练与验证流程 SHALL 覆盖关键模块开关组合，确保每条退化路径都能跑通。

#### Scenario: 全模块路径可训练
- **GIVEN** `use_deci_module1 == 1`
- **AND** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 1`
- **WHEN** 执行低预算训练验证
- **THEN** 训练 MUST 完成至少一个 fold
- **AND** 指标打印 MUST 包含 loss、accuracy、precision、recall、macro F1 和 ROC AUC

#### Scenario: 模块 1 禁用路径可训练
- **GIVEN** `use_deci_module1 == 0`
- **WHEN** 执行低预算训练验证
- **THEN** 训练 MUST 完成至少一个 fold
- **AND** 模型 MUST 使用 raw projected feature 进入后续图路径

#### Scenario: 模块 2 禁用路径可训练
- **GIVEN** `use_causal_module2 == 0`
- **WHEN** 执行低预算训练验证
- **THEN** 训练 MUST 完成至少一个 fold
- **AND** batch MUST 向模型提供 sample correlation matrix
- **AND** 总 loss MUST NOT 包含模块 2 auxiliary loss

#### Scenario: 模块 3/4 禁用路径可训练
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** 执行低预算训练验证
- **THEN** 训练 MUST 完成至少一个 fold
- **AND** 模型 MUST 使用 GCN fallback 完成分类
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss

### Requirement: 训练日志和可视化标注当前路径

训练日志与中间量可视化 SHALL 标注当前使用的模块路径，便于区分不同消融实验。

#### Scenario: 日志打印模块开关状态
- **WHEN** 训练开始一个 fold
- **THEN** 日志 MUST 打印 `use_deci_module1`、`use_causal_module2` 和 `use_hyperbolic_modules34` 的当前值
- **AND** 日志 MUST 标明当前分类路径是 `hgcn_hpec` 还是 `gcn_fallback`

#### Scenario: 可视化保存当前路径中间量
- **GIVEN** 显式启用中间量可视化
- **WHEN** 当前 fold 训练结束
- **THEN** 可视化输出 MUST 包含当前节点特征来源
- **AND** 可视化输出 MUST 包含当前 adjacency 来源
- **AND** 若使用 GCN fallback，输出 MUST 包含 GCN hidden 或 readout 表征
