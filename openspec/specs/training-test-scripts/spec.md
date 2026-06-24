## Purpose

定义根目录训练测试脚本的行为：这些脚本用于快速验证本地数据集、模型构建、交叉验证训练、指标汇报、S-DeCI 模块 2/3 参数入口和显式可视化能力。
## Requirements
### Requirement: 训练冒烟测试脚本

系统 SHALL 提供一个根目录 Python 脚本，用于运行最小化的端到端训练验证。该脚本 MUST 复用现有 dataset loader、experiment class、model construction、training loop、validation loop、checkpoint handling 和 metric aggregation。

#### Scenario: 冒烟训练完成

- **WHEN** 用户在仓库根目录运行 `test_training_smoke.py`，并且数据集已放在 `dataset/` 下
- **THEN** 脚本 MUST 至少完成一次低预算 cross-validation 训练
- **AND** MUST 打印 accuracy、precision、recall、macro F1 和 ROC AUC 指标

#### Scenario: 冒烟训练清晰失败

- **WHEN** 必需的本地数据、导入、模型配置或训练步骤不可用
- **THEN** 脚本 MUST 以非零退出码退出
- **AND** MUST 输出清晰错误信息，指出失败的前置条件或阶段

### Requirement: Mātai 小样本训练脚本

系统 SHALL 提供一个根目录 Python 脚本，用于对本地 Mātai 数据集运行能体现部分训练效果的评估，并报告适合后续变更跟踪的小样本训练验证指标。

#### Scenario: Mātai 训练完成

- **WHEN** 用户在仓库根目录运行 `test_matai_small_sample.py`，并且 `dataset/Mātai` 存在
- **THEN** 脚本 MUST 使用与 `scripts/DeCI/Mātai.sh` 兼容的默认参数，通过现有 experiment pipeline 完成训练
- **AND** 默认训练预算 MUST 高于单纯冒烟测试，使用 5 fold、10 epoch、batch size 16、layer 2 和 `d_model` 64
- **AND** MUST 打印 accuracy、precision、recall、macro F1 和 ROC AUC 指标

#### Scenario: Mātai 数据集缺失

- **WHEN** 用户运行 `test_matai_small_sample.py`，但无法解析本地 Mātai 数据集目录
- **THEN** 脚本 MUST 以非零退出码退出
- **AND** MUST 在可能时输出期望的数据集路径和当前可用的数据集目录

### Requirement: 可配置且可重复的测试执行

测试脚本 SHALL 暴露命令行参数，用于配置运行预算和核心 experiment 设置，同时保留适合重复本地验证的安全默认值。

#### Scenario: 用户自定义运行预算

- **WHEN** 用户传入 epochs、folds、model、batch size、learning rate、device usage 或 checkpoint directory 等选项
- **THEN** 脚本 MUST 将这些选项应用到构造出的 experiment 参数中
- **AND** MUST 不要求用户修改源码

#### Scenario: 用户自定义 S-DeCI 模块参数

- **WHEN** 用户传入模块 2 或模块 3 相关参数
- **THEN** 脚本 MUST 能配置 `use_causal_module2`、`causal_graph_method`、`lambda_causal_recon`、`lambda_causal_dag`、`lambda_causal_l1`
- **AND** MUST 能配置 `use_hgcn_module3`、`hgcn_hidden_dim`、HGCN 层数、曲率、Backclip 半径和邻接归一化方式
- **AND** 默认 `hgcn_hidden_dim` MUST 为 `128`

#### Scenario: 按间隔打印训练指标

- **WHEN** 用户传入 `print_metric_every`
- **THEN** 训练流程 MUST 能按指定 epoch 间隔打印 total loss、classification loss、模块 2 auxiliary loss、reconstruction loss、DAG loss、L1 loss
- **AND** MUST 同时打印训练集和验证集的 accuracy、precision、recall、macro F1 和 ROC AUC

#### Scenario: 按类别分行打印 epoch 结果

- **GIVEN** `print_metric_every > 0` 或 `print_process` 启用
- **WHEN** 训练流程打印某个 epoch 的训练与验证结果
- **THEN** 输出 MUST 使用分隔线区分不同 epoch
- **AND** 输出 MUST 将 loss 字段放在 `[Loss]` 行
- **AND** 输出 MUST 将训练集指标放在 `[Train Metrics]` 行
- **AND** 输出 MUST 将验证集指标放在 `[Validation Metrics]` 行
- **AND** 输出 SHOULD 避免每个字段单独占一行，以控制日志高度

#### Scenario: 保留关键 loss 字段

- **WHEN** 训练流程打印 `[Loss]` 行
- **THEN** 输出 MUST 包含 `total_loss`、`cls_loss` 和 `val_loss`
- **AND** 若 S-DeCI 模块 2 或模块 4 启用，输出 SHOULD 包含对应 auxiliary loss、HPEC loss 或 prototype loss 字段

#### Scenario: 重复运行不污染 benchmark 产物

- **WHEN** 任一脚本成功完成
- **THEN** 生成的 checkpoints MUST 隔离在测试专用位置
- **AND** 模型权重清理 MUST 默认启用

### Requirement: 显式训练可视化

测试脚本 SHALL 支持显式开启 S-DeCI 中间量可视化，并默认避免保存大量图片。

#### Scenario: 保存 train/test 中间量 heatmap

- **WHEN** 用户显式开启 `visualize_causal`
- **THEN** 每个 fold 训练结束后 SHOULD 保存训练集 batch 和测试集 batch 的中间量 heatmap
- **AND** 文件名 SHOULD 能区分 `train` 和 `test`
- **AND** 测试集 label MUST NOT 作为模型 forward 输入

#### Scenario: 保存最终 epoch t-SNE

- **WHEN** 用户显式开启 `visualize_causal`
- **THEN** 每个 fold 最后一个 epoch 结束后 SHOULD 保存训练集和测试集联合 t-SNE 图
- **AND** t-SNE 图 MUST 用样式区分 train/test
- **AND** t-SNE 图 MUST 用颜色区分真实 label

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

### Requirement: ?????? HPEC energy ? prototype ?????

系统 SHALL 支持模块 4 的 combined energy 与 prototype warm-start，使 HPEC cone energy 在训练早期更稳定，并能和距离项共同作为分类依据。

#### Scenario: run_cv 暴露 HPEC 参数

- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST 包含 `hpec_distance_weight`
- **AND** 参数列表 MUST 包含 `hpec_data_init`
- **AND** 参数 help SHOULD 使用中文说明其作用

#### Scenario: MDD 最佳配置传递 HPEC 参数

- **WHEN** 用户运行 `test_mdd_best_config.py`
- **THEN** 脚本 MUST 支持 `--hpec-distance-weight` 与 `--hpec-data-init`
- **AND** 脚本 MUST 将这些参数写入 experiment args，供 `S-DeCI` 模块 4 使用

### Requirement: 支持 fold 内输入时序 harmonization

系统 SHALL 支持输入时序层面的多站点 harmonization，统计量 MUST 只从当前训练 fold 估计，避免验证/测试 fold 信息泄漏。

#### Scenario: 启用 site_zscore harmonization

- **WHEN** 参数 `time_series_harmonization == "site_zscore"`
- **THEN** 数据集 MUST 从样本路径解析或提供站点 id
- **AND** 每个 fold MUST 只用训练集估计站点级 ROI 均值和标准差
- **AND** 训练集与验证集 MUST 使用训练集统计量 transform
- **AND** 当某个站点训练样本数不足时 SHOULD 回退到训练集全局统计量

#### Scenario: 禁用 harmonization

- **WHEN** 参数 `time_series_harmonization == "none"`
- **THEN** 数据集 MUST 保持原始时序输入

#### Scenario: 打印站点与标签分布

- **WHEN** `print_data_info == 1`
- **THEN** 数据加载流程 MUST 打印每个 fold 的训练集与验证集 `site × label` 计数表
- **AND** 该诊断信息 SHOULD 用于判断 site-adversarial 是否可能擦除诊断信号

### Requirement: S-DeCI 支持可消融的站点对抗分支

系统 SHALL 在 S-DeCI 中支持可选的 site-adversarial head，使 `z_global/z_tangent` 表征能在消融实验中被约束为更难预测站点。

#### Scenario: batch 携带 site label

- **WHEN** `use_site_adversarial == 1`
- **THEN** Dataset 或 fold subset MUST 解析当前样本的站点 id
- **AND** DataLoader batch MUST 在 `(x, label)` 或 `(x, label, correlation_matrix)` 后追加 `site_label`

#### Scenario: 计算站点对抗 loss

- **GIVEN** S-DeCI 模块 3 输出 `z_tangent`
- **AND** 当前 batch 包含 `site_label`
- **WHEN** `lambda_site_adversarial > 0`
- **THEN** 模型 MUST 通过 gradient reversal layer 将 `z_tangent` 输入 site classifier
- **AND** 总 loss MUST 加入加权后的 site adversarial loss
- **AND** 训练日志 SHOULD 打印 site adversarial loss

#### Scenario: 默认关闭站点对抗

- **WHEN** `use_site_adversarial == 0`
- **THEN** DataLoader MUST 不额外返回 `site_label`
- **AND** 总 loss MUST NOT 包含 site adversarial loss
- **AND** MDD 默认测试配置 SHOULD 保持 `time_series_harmonization == "site_zscore"` 且 `use_site_adversarial == 0`
