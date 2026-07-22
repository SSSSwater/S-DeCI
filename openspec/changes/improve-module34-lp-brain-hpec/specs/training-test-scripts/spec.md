## ADDED Requirements

### Requirement: 训练入口支持 LP-Brain-HPEC 参数

训练入口 SHALL 支持配置 `module34_arch=lp_brain_hpec` 以及 LP-Brain-HPEC 所需的核心几何和损失参数。

#### Scenario: run_cv 暴露 LP-Brain-HPEC 参数
- **WHEN** 用户运行 `python run_cv.py --help`
- **THEN** 参数列表 MUST 包含 `module34_arch`
- **AND** MUST 包含 Lorentz hidden dim、Lorentz layer 数、入出边平衡系数或等价 Directed Lorentz GCN 参数
- **AND** MUST 包含 `module34_geo_dtype`
- **AND** MUST 包含 MAC/HBR 相关参数：`mac_min_radius`、`mac_max_radius`、`hbr_safe_radius` 和 `hbr_loss_weight`
- **AND** help 文本 MUST 使用中文说明，必要英文关键词 MAY 保留

#### Scenario: best config 脚本传递 LP 参数
- **WHEN** 用户运行 `test_mdd_best_config.py`
- **THEN** 脚本 MUST 能接收 LP-Brain-HPEC 相关 CLI 参数
- **AND** 脚本 MUST 将这些参数写入 `Exp_Main` 使用的 args

### Requirement: 训练日志和 TensorBoard 记录 LP-Brain-HPEC 诊断

训练流程 SHALL 在 LP-Brain-HPEC 路径启用时记录几何稳定性、入出边聚合和 HPEC energy 诊断。

#### Scenario: 控制台打印 LP 诊断
- **GIVEN** `module34_arch == "lp_brain_hpec"`
- **AND** `print_metric_every > 0`
- **WHEN** 训练流程打印 epoch 指标
- **THEN** 日志 MUST 包含 Poincare 半径或 MAC 后半径
- **AND** MUST 包含 MAC clip 比例
- **AND** MUST 包含 HBR loss
- **AND** SHOULD 包含入边聚合强度和出边聚合强度

#### Scenario: TensorBoard 记录 LP 诊断
- **GIVEN** `use_tensorboard == 1`
- **AND** `module34_arch == "lp_brain_hpec"`
- **WHEN** 训练流程写入 epoch 标量
- **THEN** TensorBoard MUST 记录 MAC 半径、HBR loss、Lorentz constraint error 或等价几何诊断
- **AND** MUST 继续记录现有 Loss、Metrics、Module2 和 Final 指标

### Requirement: result.xlsx 记录 LP-Brain-HPEC 完整实验

完整 5-fold 测试 SHALL 将 LP-Brain-HPEC 配置和最终测试指标追加到 `result.xlsx`。

#### Scenario: 记录完整 5-fold 指标
- **GIVEN** MDD AAL116 完整 5-fold 训练完成
- **WHEN** 训练脚本保存结果
- **THEN** `result.xlsx` MUST 记录 `module34_arch`
- **AND** MUST 记录 accuracy、precision、recall、macro F1 和 AUC
- **AND** 指标 MUST 使用最后稳定 epoch 的测试集结果，而不是最佳 epoch

### Requirement: LP-Brain-HPEC 必须完成对比验证

该变更实现后 SHALL 至少完成 smoke test 和 MDD 完整 5-fold 对比，以判断模块 3/4 是否相对现有路径有正向作用。

#### Scenario: 单元级 forward/backward 检查
- **WHEN** 使用随机 `[B,N,D]` 节点特征和 `[B,N,N]` adjacency 测试 LP-Brain-HPEC
- **THEN** forward MUST 输出 finite logits/energy
- **AND** backward MUST 能更新 Lorentz GCN 和 HPEC 参数
- **AND** 输出中 MUST 无 NaN 或 Inf

#### Scenario: MDD 完整 5-fold 对比
- **GIVEN** MDD AAL116 数据集可用
- **WHEN** 运行完整对比实验
- **THEN** 至少 MUST 比较 `gcn_fallback`、`hgcn_hpec` 和 `lp_brain_hpec`
- **AND** 每组 MUST 记录完整 5-fold final-epoch 指标
- **AND** 如果 LP-Brain-HPEC 未优于现有路径，报告 MUST 如实记录并保留可回滚默认配置
