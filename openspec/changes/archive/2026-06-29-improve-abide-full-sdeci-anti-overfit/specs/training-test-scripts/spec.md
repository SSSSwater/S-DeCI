## ADDED Requirements

### Requirement: ABIDE 专用测试脚本默认完整四模块

根目录 ABIDE 测试脚本 SHALL 默认运行完整四模块 `S-DeCI`，用于验证用户设计框架下的 ABIDE-120 抗过拟合效果。

#### Scenario: 默认参数启用完整框架

- **WHEN** 用户直接运行 `python test_abide_best_config.py`
- **THEN** 脚本 MUST 使用 `data=Abide`
- **AND** 脚本 MUST 使用 `seq_len=120`
- **AND** 脚本 MUST 默认设置 `use_deci_module1=1`
- **AND** 脚本 MUST 默认设置 `use_causal_module2=1`
- **AND** 脚本 MUST 默认设置 `use_hyperbolic_modules34=1`
- **AND** 脚本 MUST 默认启用 HPEC 模块 4

#### Scenario: 默认关闭多站点策略

- **WHEN** 用户直接运行 `python test_abide_best_config.py`
- **THEN** 脚本 MUST 默认设置 `time_series_harmonization=none`
- **AND** 脚本 MUST 默认设置 `use_site_adversarial=0`

### Requirement: ABIDE 脚本暴露抗过拟合参数

ABIDE 测试脚本 SHALL 暴露模块 1 去噪、模块 2 因果稳定、模块 3 双曲正则和模块 4 原型正则相关参数。

#### Scenario: 暴露模块 1 去噪参数

- **WHEN** 用户查看 `test_abide_best_config.py --help`
- **THEN** 参数列表 MUST 包含随机时间窗、temporal dropout、ROI dropout 或等价模块 1 去噪参数
- **AND** 参数列表 MUST 包含模块 1 denoising loss 权重

#### Scenario: 暴露模块 2 稳定参数

- **WHEN** 用户查看 `test_abide_best_config.py --help`
- **THEN** 参数列表 MUST 包含因果图稳定性 loss 权重
- **AND** 参数列表 MUST 包含样本残差图幅度与正则参数

#### Scenario: 暴露模块 3/4 正则参数

- **WHEN** 用户查看 `test_abide_best_config.py --help`
- **THEN** 参数列表 MUST 包含因果 edge dropout 或等价图正则参数
- **AND** 参数列表 MUST 包含 HPEC prototype warm-start、margin、半径正则和 diversity 相关参数

### Requirement: ABIDE 脚本默认保存可选诊断

ABIDE 测试脚本 SHALL 支持显式开启完整四模块中间量诊断，并默认不大量保存图片。

#### Scenario: 默认不保存大量图片

- **WHEN** 用户直接运行 `python test_abide_best_config.py`
- **THEN** 脚本 MUST 默认不保存大量 heatmap
- **AND** 脚本 MUST 仍打印关键 loss 与指标

#### Scenario: 显式开启可视化

- **WHEN** 用户设置 `--visualize-causal 1`
- **THEN** 脚本 MUST 保存模块 1 去噪输出、模块 2 因果图、模块 3 双曲表示和模块 4 prototype energy 相关中间量
- **AND** 脚本 MUST 保存 train/test t-SNE

### Requirement: ABIDE 训练验证报告过拟合诊断

训练流程 SHALL 为 ABIDE 输出训练集与验证集的差异，帮助判断过拟合是否缓解。

#### Scenario: 打印 train/test 指标

- **WHEN** ABIDE 脚本按 epoch 间隔打印日志
- **THEN** 日志 MUST 同时包含训练集和验证集 accuracy、precision、recall、macro F1、ROC AUC
- **AND** 日志 MUST 包含 total loss、HPEC loss、因果辅助 loss 和 prototype loss

#### Scenario: 汇总最佳 fold 结果

- **WHEN** ABIDE 脚本完成一个或多个 fold
- **THEN** 脚本 MUST 输出最终 accuracy、precision、recall、macro F1 和 ROC AUC
- **AND** 脚本 SHOULD 标注使用的最佳阈值或 energy-based prediction 规则
