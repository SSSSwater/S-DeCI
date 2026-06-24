## ADDED Requirements

### Requirement: 训练流程兼容相关矩阵 batch

训练流程 SHALL 兼容 Dataset 返回 `(x_enc, label)` 或 `(x_enc, label, correlation_matrix)` 两种 batch 格式。

#### Scenario: 处理二元组 batch

- **GIVEN** DataLoader 返回 `(x_enc, label)`
- **WHEN** 训练、验证、可视化或 t-SNE 流程执行
- **THEN** 系统 MUST 保持现有 `model(x_enc)` 调用路径

#### Scenario: 处理三元组 batch

- **GIVEN** DataLoader 返回 `(x_enc, label, correlation_matrix)`
- **WHEN** 训练、验证、可视化或 t-SNE 流程执行
- **THEN** 系统 MUST 将 `correlation_matrix` 移动到与 `x_enc` 相同 device
- **AND** MUST 调用 `model(x_enc, correlation_matrix=correlation_matrix)`
- **AND** 测试集 label MUST NOT 作为模型输入

### Requirement: 训练入口暴露相关矩阵回退参数

训练入口 SHALL 暴露模块 2 关闭时加载样本相关矩阵的配置参数。

#### Scenario: run_cv 参数

- **WHEN** 用户运行 `run_cv.py`
- **THEN** 系统 MUST 支持配置是否启用 sample correlation fallback
- **AND** MUST 支持配置相关矩阵负值处理模式
- **AND** 参数 help 文本 MUST 使用中文描述，必要关键词可以保留英文

#### Scenario: 根目录测试脚本参数

- **WHEN** 用户运行 `test_training_smoke.py` 或 `test_matai_small_sample.py`
- **THEN** 脚本 MUST 支持传入模块 2 关闭、模块 3 开启和 sample correlation fallback 参数
- **AND** 脚本 MUST 能用低预算训练验证该路径

### Requirement: 模块 2 关闭路径训练可验证

系统 SHALL 提供可重复的低预算训练验证，确认模块 2 关闭时模块 3 能使用样本相关矩阵跑通。

#### Scenario: 低预算训练完成

- **GIVEN** 数据集中存在对应样本的 correlation matrix 文件
- **WHEN** 用户运行低预算训练并设置 `use_causal_module2=0`、`use_hgcn_module3=1`
- **THEN** 训练 MUST 完成至少一个 fold
- **AND** MUST 打印 accuracy、precision、recall、macro F1 和 ROC AUC
- **AND** 输出 loss 中 MUST 不包含模块 2 auxiliary loss 的有效贡献

#### Scenario: 相关矩阵缺失时清晰失败

- **GIVEN** 用户启用 sample correlation fallback
- **AND** 数据集中缺少某些样本的 correlation matrix
- **WHEN** 训练启动
- **THEN** 脚本 MUST 以非零退出码失败
- **AND** 错误信息 MUST 指向缺失的样本或候选路径
