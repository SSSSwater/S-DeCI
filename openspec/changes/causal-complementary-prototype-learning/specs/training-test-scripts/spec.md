## ADDED Requirements

### Requirement: 训练入口暴露互补学习消融参数

训练入口和超参数扫描 SHALL 暴露可靠 prototype、因果显著性互补视图、多阶因果编码及其损失/日程参数。

#### Scenario: CLI 和默认脚本参数
- **WHEN** 用户查看 `run_cv.py`、`test_*_best_config.py` 或 `sweep_hparam.py`
- **THEN** 系统 MUST 提供 `hpec_prototype_update_mode`、可靠阈值和 EMA 参数
- **AND** MUST 提供互补视图开关、mask 日程、显著性融合权重和视图 loss 权重
- **AND** MUST 提供多阶因果编码开关、hop 数和残差 scale
- **AND** 所有参数的 `help` 文本 MUST 为中文，必要关键词可保留英文

### Requirement: 完整消融和结果记录

系统 SHALL 支持分别切换每项新机制，并把完整训练指标与训练时长写入既有结果记录。

#### Scenario: 运行完整消融
- **GIVEN** 用户使用同一数据划分运行 5-fold、至少 50 epoch 的训练
- **WHEN** 分别启用可靠 TP EMA、互补视图、视图一致性和多阶因果编码
- **THEN** 系统 MUST 在 `result.xlsx` 中记录各配置的 Accuracy、Precision、Recall、Macro-F1、AUC 和训练时长
- **AND** TensorBoard summary MUST 记录 `Complementary/*`、`PrototypeUpdate/*` 和 `CausalReachability/*` 分组
- **AND** 结果不得以 smoke test 替代完整消融结论
