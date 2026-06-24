## ADDED Requirements

### Requirement: 训练入口暴露模块 4 HPEC 参数

训练入口 SHALL 暴露模块 4 HPEC 相关参数，使用户能在 `run_cv.py` 和根目录测试脚本中显式开启或关闭模块 4。

#### Scenario: 配置 HPEC 开关

- **WHEN** 用户运行训练入口
- **THEN** 系统 MUST 支持配置 `use_hpec_module4`
- **AND** 当 `use_hpec_module4 == 1` 时 MUST 同时启用或要求启用 `use_hgcn_module3`

#### Scenario: 配置 HPEC 超参数

- **WHEN** 用户配置模块 4
- **THEN** 系统 MUST 支持配置 `hpec_prototype_radius`、`hpec_cone_k`、`hpec_margin`、`hpec_trainable_prototypes`、`hpec_init_steps` 和 `hpec_eps`
- **AND** 参数 help 文本 SHOULD 使用中文描述，必要关键词 MAY 保留英文

### Requirement: 训练流程支持模型 primary loss

训练流程 SHALL 支持由模型内部提供 primary/classification loss，以便模块 4 使用 HPEC energy loss 替代外部 criterion。

#### Scenario: 使用 HPEC loss 训练

- **GIVEN** 模块 4 已启用
- **WHEN** 训练循环完成一次 forward
- **THEN** 训练流程 MUST 优先读取模型提供的 HPEC primary loss
- **AND** 总 loss MUST 继续合并模块 2 auxiliary loss
- **AND** 总 loss MUST 只执行一次 `backward()`

#### Scenario: 普通模型保持兼容

- **GIVEN** 当前模型未提供 primary/classification loss
- **WHEN** 训练流程计算 loss
- **THEN** 系统 MUST 回退到现有外部 criterion 逻辑
- **AND** 非 S-DeCI 模型的训练行为 MUST 保持兼容

### Requirement: 训练指标支持 energy-based prediction

训练和验证 SHALL 在模块 4 启用时使用 energy-based prediction 和 probability 汇报指标。

#### Scenario: 汇报 HPEC 指标

- **GIVEN** 模型缓存了 HPEC prediction 和 probability
- **WHEN** 训练流程汇总 accuracy、precision、recall、macro F1 和 ROC AUC
- **THEN** 系统 MUST 优先使用 HPEC prediction
- **AND** ROC AUC 或概率相关指标 MUST 使用 `softmax(-energy_matrix)` 得到的概率

#### Scenario: 按间隔打印 HPEC loss

- **WHEN** 用户配置 `print_metric_every`
- **THEN** 训练流程 MUST 能按间隔打印 total loss、HPEC loss、模块 2 auxiliary loss、reconstruction loss、DAG loss、L1 loss
- **AND** MUST 同时打印训练集和测试集或验证集的 accuracy、precision、recall、macro F1 和 ROC AUC

### Requirement: 测试脚本支持低预算 HPEC 验证

根目录测试脚本 SHALL 支持低预算验证模块 4 HPEC 能跑通训练、loss 和指标流程。

#### Scenario: 冒烟测试启用 HPEC

- **WHEN** 用户在根目录运行冒烟测试并显式开启模块 4
- **THEN** 测试 MUST 能完成至少一个低预算 cross-validation fold
- **AND** MUST 能打印 HPEC loss、总 loss 和分类指标

#### Scenario: Mātai 小样本测试启用 HPEC

- **WHEN** 用户在根目录运行 Mātai 小样本测试并显式开启模块 4
- **THEN** 测试 MUST 能使用比冒烟测试更高的训练预算完成训练
- **AND** MUST 能保存或打印足够信息用于比较模块 3 线性分类与模块 4 HPEC 分类效果
