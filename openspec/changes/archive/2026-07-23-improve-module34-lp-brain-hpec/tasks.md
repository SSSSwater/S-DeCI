## 1. 设计与基线核对

- [x] 1.1 核对 Poincare HGCN-HPEC、GCN fallback 与拟议 LP-Brain-HPEC 的数据流、loss 和诊断接入点。
- [x] 1.2 从 `result.xlsx`、TensorBoard 和实验文档确认同划分对比基线与 final-epoch 指标口径。

## 2. LP-Brain-HPEC 原型实现与数值验证

- [x] 2.1 历史实现 Lorentz lifting、有向入/出边传播、Lorentz readout、stereographic bridge、MAC 与 HBR。
- [x] 2.2 完成随机张量 forward/backward、形状、NaN/Inf 和 MDD 单折 smoke 检查。
- [x] 2.3 记录 Lorentz constraint、bridge/MAC 半径、clip 比例和 HBR 等几何诊断。

## 3. 完整实验与结论

- [x] 3.1 完成 MDD/AAL116 5-fold、50 epoch 的 LP-Brain-HPEC 完整训练。
- [x] 3.2 将 LP final-epoch 结果写入 `result.xlsx`：Accuracy 62.63%、Macro-F1 59.43%、AUC 62.50%。
- [x] 3.3 对照当前 Poincare HGCN-HPEC 主线，确认 LP 路径未带来正收益且训练更慢。

## 4. 负向路径清理

- [x] 4.1 从正式模型移除 `lp_brain_hpec` 架构分支与专用 layer。
- [x] 4.2 从 `run_cv.py`、各 best-config 参数对象和结果摘要中移除遗留 Lorentz、LP readout、MAC/HBR 专用参数。
- [x] 4.3 保留完整实验指标和失败原因，不把 LP 写成当前可执行能力。
- [x] 4.4 从 `models/S_DeCI.py` 与 `exp/exp_classification_CV.py` 移除恒零 `lp_hbr_weighted_loss`、LP TensorBoard tag、epoch accumulator 和打印字段。

## 5. 规范与文档

- [x] 5.1 更新 proposal、design、delta specs 和主规范，明确当前正式路线为 Poincare HGCN `mean_std` + HPEC。
- [x] 5.2 在 `docs/S-DeCI模型修改证据台账.md` 保留 LP 负向证据及未来重新研究的准入条件。
