## MODIFIED Requirements

### Requirement: 训练入口只暴露当前正式模块 3/4 参数

训练入口 SHALL 暴露 Poincare HGCN、HPEC 多原型和 dual-view evidence fusion 所需参数，不再暴露已退役 LP-Brain-HPEC 的专用参数。

#### Scenario: 查看训练参数
- **WHEN** 用户运行 `python run_cv.py --help` 或打开 `test_mdd_best_config.py`
- **THEN** 系统 MUST 提供模块 3/4 开关、HGCN readout、HPEC prototype 和 evidence fusion 参数
- **AND** MUST NOT 提供无执行路径的 `module34_arch=lp_brain_hpec`
- **AND** SHOULD 移除只服务于 LP 的 Lorentz、MAC 和 HBR 参数
- **AND** help 文本 MUST 使用中文，必要英文关键词 MAY 保留

### Requirement: 记录 LP 完整负向实验结论

系统 SHALL 保留 LP-Brain-HPEC 的完整五折 final-epoch 指标和退出正式路线的原因，避免后续重复引入同一失败设计。

#### Scenario: 查阅实验证据
- **WHEN** 开发者查阅 `result.xlsx` 或模型修改证据台账
- **THEN** MUST 能确认 LP-Brain-HPEC 在 MDD/AAL116 5-fold、50 epoch 下的 Accuracy 为 62.63%
- **AND** MUST 能确认 Macro-F1 为 59.43%、AUC 为 62.50%
- **AND** MUST 说明该路径低于 Poincare HGCN-HPEC 主线且训练更慢
- **AND** MUST NOT 将 smoke test 当作正式保留 LP 能力的依据

### Requirement: TensorBoard 不记录无执行路径的 LP 专用标量

训练流程 SHALL 记录当前实际执行路径的 loss、指标和几何诊断。

#### Scenario: 使用当前 HGCN-HPEC 主路线训练
- **WHEN** 每个 epoch 写入 TensorBoard
- **THEN** 系统 MUST 记录当前 HGCN/HPEC 与 prototype 诊断
- **AND** SHOULD NOT 继续注册只可能恒为零或缺失的 LP Lorentz constraint、MAC clip 或 HBR tag
