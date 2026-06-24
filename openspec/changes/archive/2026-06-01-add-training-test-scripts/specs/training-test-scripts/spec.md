## ADDED Requirements

### Requirement: 训练冒烟测试脚本
系统 SHALL 提供一个根目录 Python 脚本，用于运行最小化的端到端训练验证。该脚本 MUST 复用现有 dataset loader、experiment class、model construction、training loop、validation loop、checkpoint handling 和 metric aggregation。

#### Scenario: 冒烟训练完成
- **WHEN** 用户在仓库根目录运行冒烟测试脚本，并且数据集已放在 `dataset/` 下
- **THEN** 脚本 MUST 至少完成一次低预算 cross-validation 训练，并打印结果指标

#### Scenario: 冒烟训练清晰失败
- **WHEN** 必需的本地数据、导入、模型配置或训练步骤不可用
- **THEN** 脚本 MUST 以非零退出码退出，并输出清晰错误信息，指出失败的前置条件或阶段

### Requirement: Mātai 小样本训练脚本
系统 SHALL 提供一个根目录 Python 脚本，用于对本地 Mātai 数据集运行低预算训练评估，并报告适合后续变更跟踪的小样本训练验证指标。

#### Scenario: Mātai 训练完成
- **WHEN** 用户在仓库根目录运行 Mātai 小样本脚本，并且 `dataset/Mātai` 存在
- **THEN** 脚本 MUST 使用与 Mātai 兼容的默认参数，通过现有 experiment pipeline 完成训练，并打印 accuracy、precision、recall、macro F1 和 ROC AUC 指标

#### Scenario: Mātai 数据集缺失
- **WHEN** 用户运行 Mātai 小样本脚本，但无法解析本地 Mātai 数据集目录
- **THEN** 脚本 MUST 以非零退出码退出，并在可能时输出期望的数据集路径和当前可用的数据集目录

### Requirement: 可配置且可重复的测试执行
测试脚本 SHALL 暴露命令行参数，用于配置运行预算和核心 experiment 设置，同时保留适合重复本地验证的安全默认值。

#### Scenario: 用户自定义运行预算
- **WHEN** 用户传入 epochs、folds、model、batch size、learning rate、device usage 或 checkpoint directory 等选项
- **THEN** 脚本 MUST 将这些选项应用到构造出的 experiment 参数中，且不要求用户修改源码

#### Scenario: 重复运行不污染 benchmark 产物
- **WHEN** 任一脚本成功完成
- **THEN** 生成的 checkpoints MUST 隔离在测试专用位置，并且模型权重清理 MUST 默认启用
