## ADDED Requirements

### Requirement: 训练入口支持多 prototype 超参数

训练入口和根目录测试脚本 SHALL 支持配置 HPEC 多 prototype 数量、温度和 prototype loss 权重。

#### Scenario: run_cv 暴露多 prototype 参数

- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST 包含 `hpec_prototypes_per_class`
- **AND** MUST 包含 `hpec_proto_temperature`
- **AND** MUST 包含 `lambda_hpec_mle`、`lambda_hpec_pcl` 和 `lambda_hpec_pal`

#### Scenario: 测试脚本传递多 prototype 参数

- **WHEN** 用户运行 `test_training_smoke.py` 或 `test_matai_small_sample.py`
- **THEN** 脚本 MUST 能接收多 prototype 相关 CLI 参数
- **AND** 脚本 MUST 将这些参数写入构造出的 experiment args

### Requirement: 训练日志打印 prototype loss

训练流程 SHALL 在按间隔打印指标时显示新增 prototype loss，便于判断多 prototype 是否参与训练。

#### Scenario: 打印 prototype loss

- **GIVEN** `print_metric_every > 0`
- **WHEN** 训练流程打印 epoch 指标
- **THEN** 日志 MUST 包含 `hpec_mle_loss`
- **AND** MUST 包含 `hpec_pcl_loss`
- **AND** MUST 包含 `hpec_pal_loss`
- **AND** MUST 包含 prototype auxiliary loss 的加权总贡献

#### Scenario: 关闭 prototype loss 时日志稳定

- **GIVEN** `lambda_hpec_mle == 0`
- **AND** `lambda_hpec_pcl == 0`
- **AND** `lambda_hpec_pal == 0`
- **WHEN** 训练流程打印 epoch 指标
- **THEN** 日志 MUST 正常输出
- **AND** prototype loss 字段 MUST 显示为 0 或等价的空贡献

### Requirement: 多 prototype 训练冒烟测试

系统 SHALL 提供低预算训练验证，确认多 prototype HPEC 能完成 forward、loss、backward 和指标输出。

#### Scenario: 多 prototype 冒烟训练完成

- **WHEN** 用户运行低预算训练并设置 `use_hpec_module4=1`、`hpec_prototypes_per_class > 1`
- **THEN** 训练 MUST 至少完成一个 fold 的一个 epoch
- **AND** MUST 输出 accuracy、precision、recall、macro F1 和 ROC AUC
- **AND** MUST 不出现 NaN loss

#### Scenario: 单 prototype 回退训练完成

- **WHEN** 用户运行低预算训练并设置 `hpec_prototypes_per_class=1`
- **THEN** 训练 MUST 完成
- **AND** 输出形状和指标计算 MUST 与当前单 prototype HPEC 路径兼容
