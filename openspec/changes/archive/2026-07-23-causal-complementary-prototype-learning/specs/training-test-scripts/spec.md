## MODIFIED Requirements

### Requirement: 训练入口暴露保留的消融参数

训练入口和超参数扫描 SHALL 暴露可靠 prototype 与多阶因果编码参数，不再暴露已退役互补学习参数。

#### Scenario: CLI 和默认脚本参数
- **WHEN** 用户查看 `run_cv.py`、`test_*_best_config.py` 或 `sweep_hparam.py`
- **THEN** 系统 MUST 提供 `hpec_prototype_update_mode`、可靠阈值和 EMA 参数
- **AND** MUST 提供多阶因果编码开关、hop 数和残差 scale
- **AND** MUST NOT 要求互补 mask、双视图一致性、InfoNCE 或 masked CE 参数
- **AND** 所有 help 文本 MUST 为中文，必要关键词 MAY 保留英文

### Requirement: TensorBoard 记录有效机制

训练流程 SHALL 记录可靠 prototype 更新和启用时的多阶因果编码诊断。

#### Scenario: 写入每 epoch 诊断
- **WHEN** TensorBoard writer 写入训练标量
- **THEN** MUST 记录可靠 TP 比例、prototype 分配/位移和未更新统计
- **AND** 多阶编码启用时 MUST 记录每 hop gate、编码范数和残差范数
- **AND** SHOULD NOT 注册已无执行路径的 `Complementary/*` 标量

### Requirement: 完整实验结论可追溯

系统 SHALL 保留可靠 TP EMA、互补视图和多阶编码的完整五折实验结果及训练时长。

#### Scenario: 查阅结果记录
- **WHEN** 开发者查阅 `result.xlsx` 与模型修改证据台账
- **THEN** MUST 能区分保留的可靠 TP EMA、可选多阶编码和已移除互补学习
- **AND** MUST 使用各 fold 最后稳定 epoch 的测试指标，而不是训练最佳 epoch
