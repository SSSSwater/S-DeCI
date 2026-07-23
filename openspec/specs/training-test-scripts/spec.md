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
- **THEN** 脚本 MUST 能配置 `use_causal_module2`、`causal_graph_method`、`temporal_lag_order`、`lambda_temporal_pred`、`lambda_temporal_sparse`、`lambda_temporal_smooth` 和 `lambda_causal_dag`
- **AND** MUST 能配置 `use_hgcn_module3`、`hgcn_hidden_dim`、HGCN 层数、曲率、Backclip 半径和邻接归一化方式
- **AND** 默认 `hgcn_hidden_dim` MUST 为 `128`
- **AND** 设计原因 MUST 写明：模块 2 默认学习的是从历史 BOLD/ALFF 时序预测未来时间点的 Temporal NTS-NOTEARS 图，旧的 `lambda_causal_recon`、静态 feature reconstruction 和 `dag_sampling` 只作为 legacy/debug 参数，不应出现在推荐测试脚本的主参数分组中

#### Scenario: 用户自定义 attention-guided 模块 2 参数

- **WHEN** 用户选择 `causal_graph_method == "attn_nts_notears"`
- **THEN** 训练入口和根目录测试脚本 MUST 能配置 `temporal_attention_heads`
- **AND** MUST 能配置 `temporal_attention_head_dim`
- **AND** MUST 能配置 `temporal_attention_dropout`
- **AND** MUST 能配置 `temporal_attention_graph_scale`
- **AND** 参数 help SHOULD 使用中文说明，必要英文关键词 MAY 保留

#### Scenario: 按间隔打印训练指标

- **WHEN** 用户传入 `print_metric_every`
- **THEN** 训练流程 MUST 能按指定 epoch 间隔打印 total loss、classification loss、模块 2 temporal prediction loss、sparse loss、smooth loss、DAG loss 和关键 HPEC loss
- **AND** MUST 同时打印训练集和验证集的 accuracy、precision、recall、macro F1 和 ROC AUC
- **AND** 模块 2 loss 字段 MUST 对应下式中的各项：

$$
\mathcal{L}_{\mathrm{module2}}
=
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0).
$$

- **AND** 该打印口径 MUST 与“过去时间窗预测未来时间点”的时序因果设计一致，不得把默认模块 2 描述为静态重构训练

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

训练流程 SHALL 支持由模型内部提供 primary/classification loss。默认 `hgcn_hpec` 主路线中，模型内部 loss 应以最终融合 logits $\hat{Y}$ 的分类损失为主，并合并 HPEC energy/prototype auxiliary loss；只有显式 energy-only 实验路径 MAY 让 HPEC energy loss 完全替代外部 criterion。

#### Scenario: 使用 HPEC loss 训练

- **GIVEN** 模块 4 已启用
- **WHEN** 训练循环完成一次 forward
- **THEN** 训练流程 MUST 优先读取模型提供的 primary/classification loss
- **AND** 总 loss MUST 继续合并模块 2 auxiliary loss
- **AND** 总 loss MUST 只执行一次 `backward()`
- **AND** 默认 `hgcn_hpec` 路线中，该 primary/classification loss MUST 可写作：

$$
\mathcal{L}_{\mathrm{cls}}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{
\exp(\hat{Y}_{b,y_b})
}{
\sum_{k=1}^{K}\exp(\hat{Y}_{b,k})
},
$$

其中：

$$
\hat{Y}_{b}
=
\ell^{\mathrm{base}}_{b}
+
\lambda_{\mathrm{hyp}}g_b r^{\mathrm{hyper}}_{b}.
$$

- **AND** 设计原因 MUST 写明：训练流程读取模型内部 loss 是为了让模块 4 的 HPEC evidence、校准双曲 evidence 增量 和欧氏局部结构 logits 在同一处组成最终监督，避免训练端和指标端分别使用不同分类证据。

#### Scenario: 普通模型保持兼容

- **GIVEN** 当前模型未提供 primary/classification loss
- **WHEN** 训练流程计算 loss
- **THEN** 系统 MUST 回退到现有外部 criterion 逻辑
- **AND** 非 S-DeCI 模型的训练行为 MUST 保持兼容

### Requirement: 训练指标支持 HPEC 融合预测

训练和验证 SHALL 在默认 HGCN-HPEC 模块 4 启用时使用最终融合 logits $\hat{Y}$ 汇报指标。energy-based prediction 只作为历史实验口径保留。

#### Scenario: 汇报默认 HPEC 融合指标

- **GIVEN** 模型输出最终融合 logits $\hat{Y}\in\mathbb{R}^{B\times K}$
- **WHEN** 训练流程汇总 accuracy、precision、recall、macro F1 和 ROC AUC
- **THEN** 系统 MUST 使用如下预测类别：

$$
\hat{y}_b
=
\operatorname*{argmax}_{k}\hat{Y}_{b,k}.
$$

- **AND** ROC AUC 或概率相关指标 MUST 使用如下概率：

$$
p_{b,k}
=
\operatorname{softmax}(\hat{Y}_b)_k
=
\frac{\exp(\hat{Y}_{b,k})}
{\sum_{r=1}^{K}\exp(\hat{Y}_{b,r})}.
$$

- **AND** 设计原因 MUST 写明：默认主路线中 HPEC energy 是双曲原型证据，不是唯一分类器；指标必须和训练时的最终融合 logits 对齐，否则会出现 loss 优化目标和控制台指标不一致。

#### Scenario: 汇报显式 energy-only 实验指标

- **GIVEN** 用户显式启用 energy-only 或 `lp_brain_hpec` 对照路径
- **WHEN** 模块缓存了 `energy_matrix`，且该路径声明 energy 是最终分类依据
- **THEN** 预测类别 MAY 使用：

$$
\hat{y}_b
=
\operatorname*{argmin}_{k}E_{b,k}.
$$

- **AND** 概率 MAY 使用：

$$
p_{b,k}
=
\operatorname{softmax}(-E_b)_k.
$$

- **AND** 文档 MUST 明确该逻辑不是默认 `hgcn_hpec` 主路线。

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

### Requirement: 训练入口支持 HPEC prototype 超参数

训练入口和根目录测试脚本 SHALL 支持配置 HPEC 多 prototype 数量、温度、能量函数等默认主线参数，并保留可选 distillation 校准、半径约束和 prototype separation 作为可选消融参数。

#### Scenario: run_cv 暴露多 prototype 参数

- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST 包含 `hpec_prototypes_per_class`
- **AND** MUST 包含 `hpec_proto_temperature`
- **AND** MUST 包含 `hpec_loss_mode`、`hpec_energy_mode`、`hpec_energy_loss_weight`
- **AND** MUST 包含 `hpec_teacher_distill_weight`、`hpec_z_radius_loss_weight` 和 `hpec_prototype_separation_loss_weight`
- **AND** MUST 包含 `hpec_trainable_prototypes`、`hpec_use_sinkhorn_ema` 或等价 prototype 更新控制参数

#### Scenario: 测试脚本传递多 prototype 参数

- **WHEN** 用户运行 `test_training_smoke.py` 或 `test_matai_small_sample.py`
- **THEN** 脚本 MUST 能接收多 prototype 相关 CLI 参数
- **AND** 脚本 MUST 将这些参数写入构造出的 experiment args

### Requirement: 训练日志打印 HPEC prototype 诊断

训练流程 SHALL 在按间隔打印指标时显示当前 HPEC energy 和 prototype 诊断；当 distillation 校准、半径约束或 prototype separation 被显式启用时，SHALL 同步显示对应诊断，便于判断多 prototype 是否参与训练和是否坍缩。

#### Scenario: 打印 HPEC prototype 诊断

- **GIVEN** `print_metric_every > 0`
- **WHEN** 训练流程打印 epoch 指标
- **THEN** 日志 MUST 包含 `hpec_final_ce_loss`
- **AND** MUST 包含 `hpec_energy_weighted_loss`
- **AND** 当启用 distillation 校准时 MUST 包含 `hpec_teacher_distill_loss` 或其加权值
- **AND** 当启用半径或 prototype separation 约束时 MUST 包含对应未加权值和加权贡献

#### Scenario: 关闭 HPEC auxiliary 项时日志稳定

- **GIVEN** `hpec_energy_loss_weight == 0`
- **AND** `hpec_teacher_distill_weight == 0`
- **AND** `hpec_z_radius_loss_weight == 0`
- **AND** `hpec_prototype_separation_loss_weight == 0`
- **WHEN** 训练流程打印 epoch 指标
- **THEN** 日志 MUST 正常输出
- **AND** 关闭的 HPEC auxiliary 字段 MUST 显示为 0 或等价的空贡献

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

#### Scenario: MDD best-config 默认使用 Temporal NTS-NOTEARS 主路线

- **WHEN** 用户直接运行 `test_mdd_best_config.py`
- **THEN** 默认 `causal_graph_method` SHOULD 为 `nts_notears`
- **AND** 该默认值 MUST 表示时间序列预测式 Temporal NTS-NOTEARS，而不是旧静态 feature reconstruction 路线
- **AND** 用户 MAY 通过命令行切换到 `attn_nts_notears` 做对照实验
- **AND** 当用户选择 `attn_nts_notears` 时，attention 参数 SHOULD 保留可配置入口：`temporal_attention_heads`、`temporal_attention_head_dim`、`temporal_attention_dropout`、`temporal_attention_graph_scale`

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

#### Scenario: 完整 5-fold 结果写入 result.xlsx

- **WHEN** 根目录 best-config 脚本完成完整 `kfold` 训练
- **THEN** 系统 SHOULD 将最终测试集平均 accuracy、precision、recall、macro F1 和 AUC 写入 `result.xlsx`
- **AND** 表格 MUST 记录模块设计相关参数，包括 `causal_graph_method`、`module2_sample_correlation_blend`、temporal loss 权重和 attention 参数
- **AND** 指标 SHOULD 以百分比形式保留两位小数

#### Scenario: 可视化保存当前路径中间量
- **GIVEN** 显式启用中间量可视化
- **WHEN** 当前 fold 训练结束
- **THEN** 可视化输出 MUST 包含当前节点特征来源
- **AND** 可视化输出 MUST 包含当前 adjacency 来源
- **AND** 若使用 GCN fallback，输出 MUST 包含 GCN hidden 或 readout 表征

### Requirement: 训练日志保留 LP-Brain-HPEC 历史诊断结论

LP-Brain-HPEC 已退出当前训练路径。历史实验文档 SHALL 保留几何稳定性、入出边聚合和 HPEC energy 诊断结论，当前 TensorBoard 不应继续注册无执行路径的专用标量。

#### Scenario: 查阅历史 LP 诊断

- **WHEN** 开发者检查 LP-Brain-HPEC 负向实验
- **THEN** 证据文档 MUST 说明 Lorentz constraint、bridge/MAC 半径和 HBR 诊断结论
- **AND** 当前 TensorBoard SHOULD NOT 注册只可能恒为零或缺失的 LP 专用 tag
- **AND** 当前训练 MUST 继续记录现有 Loss、Metrics、Module2、Module34 和 Final 指标

### Requirement: 训练入口支持 HPEC combined energy 与 prototype warm-start

系统 SHALL 支持模块 4 的 combined energy 与 prototype warm-start，使 HPEC cone energy 在训练早期更稳定，并能和距离项共同作为分类依据。设计依据是 HPEC / entailment cone 中“角度是否落入类原型锥体”表达层级归属，而距离项用于避免仅靠角度导致近原点样本能量不稳定。

模块 4 的 combined energy SHOULD 写成：

$$
E_{b,c,k}
=
\max(0,\phi(z_b,p_{c,k})-\psi(p_{c,k}))
+
\lambda_{\mathrm{dist}}d_{\mathbb{B}}(z_b,p_{c,k}).
$$

多原型聚合 SHOULD 写成：

$$
E_{b,c}
=
-\tau \log
\sum_{k=1}^{K}
\exp\left(-\frac{E_{b,c,k}}{\tau}\right).
$$

其中 $z_b$ 为模块 3 输出的双曲中心点，$p_{c,k}$ 为类别 $c$ 的第 $k$ 个 prototype，$\phi$ 为样本到 prototype 的共形角，$\psi$ 为 prototype aperture，$d_{\mathbb{B}}$ 为 Poincare ball 距离。这样设计的原因是：softmin 聚合允许一个类别存在多个子型 prototype，距离项提供局部几何稳定性，warm-start 避免 prototype 在训练初期完全由随机点主导。

#### Scenario: run_cv 暴露 HPEC 参数

- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST 包含 `hpec_distance_weight`
- **AND** 参数列表 MUST 包含 `hpec_data_init`
- **AND** 参数 help SHOULD 使用中文说明其作用

### Requirement: 训练入口不暴露已退役 LP-Brain-HPEC 参数

正式训练入口 SHALL 只暴露当前 HGCN-HPEC 路径参数，不再暴露 `module34_arch=lp_brain_hpec` 以及只服务于该路径的 Lorentz、bridge、MAC/HBR 参数。

#### Scenario: run_cv 不提供 LP 架构选择

- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST NOT 包含 `module34_arch=lp_brain_hpec` 选择
- **AND** SHOULD 移除 `lorentz_layers`、`lorentz_curvature`、`lorentz_alpha_out_init`、`module34_geo_dtype`
- **AND** SHOULD 移除 `mac_min_radius`、`mac_max_radius`、`hbr_safe_radius` 和 `hbr_loss_weight`
- **AND** help 文本 MUST 使用中文说明，必要英文关键词 MAY 保留

#### Scenario: best config 脚本不传递 LP 参数

- **WHEN** 用户运行根目录 best-config 脚本
- **THEN** 脚本 MUST NOT 要求 LP-Brain-HPEC 相关 CLI 参数
- **AND** 历史 LP 指标 MUST 通过结果文件和证据台账查阅

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

### Requirement: 训练入口支持可靠原型与多阶因果消融

训练入口 SHALL 提供可靠 TP EMA 与多阶因果编码的中文参数，并记录其训练诊断和完整实验结果；互补遮挡及相关 loss 已退出正式接口。

#### Scenario: TensorBoard 与完整结果记录
- **WHEN** 用户启用任一新增机制训练
- **THEN** TensorBoard MUST 写入 `PrototypeUpdate/*` 或 `CausalReachability/*` 标量
- **AND** SHOULD NOT 注册无执行路径的 `Complementary/*` 标量
- **AND** 完整 5-fold/50+ epoch 训练 MUST 将 Accuracy、Precision、Recall、Macro-F1、AUC、训练时长和模块设计参数写入 `result.xlsx`
