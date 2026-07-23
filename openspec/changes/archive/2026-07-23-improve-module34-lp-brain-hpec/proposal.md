## Why

模块 3/4 需要利用模块 2 的有向因果图，同时保持适合脑网络层级结构的双曲表示。该变更曾实现 LP-Brain-HPEC：使用 Lorentz 有向图传播、Lorentz-to-Poincare bridge、MAC 和 HBR，希望强化几何一致性。完整 MDD/AAL116 5-fold、50 epoch 结果表明，该路径的 Accuracy 为 62.63%、Macro-F1 为 59.43%、AUC 为 62.50%，低于当前 Poincare HGCN-HPEC 主线且训练更慢。因此本变更的最终目标调整为：如实保留实验结论，退出无正收益的 LP 正式能力，并明确当前模块 3/4 主路线。

## What Changes

- 保留 LP-Brain-HPEC 的实现过程、失败机制和完整指标作为研究反例，不再把它描述为当前可执行能力。
- 正式模块 3 固定使用 Poincare HGCN，并以原点切空间 `mean_std` readout 生成图级双曲表示。
- 正式模块 4直接接收模块 3 的 Poincare `z_global`，使用 HPEC 多原型能量产生双曲分类证据，不经过 Lorentz-to-Poincare bridge、MAC 或 HBR。
- 最终预测使用欧氏局部结构证据与双曲层级原型证据的 dual-view evidence fusion；两种证据共同参与分类，不向双曲坐标直接注入 FC 向量。
- 从正式代码和训练入口移除 `lp_brain_hpec` 架构选择、专用 layer、Lorentz/MAC/HBR 参数和只服务于该路径的诊断项。
- 完整实验结果保留在 `result.xlsx` 与 `docs/S-DeCI模型修改证据台账.md`，作为后续跨流形设计必须超过的基线。

## Capabilities

### New Capabilities

无。本变更最终不引入新的正式能力。

### Modified Capabilities

- `module3-hgcn-readout`: 明确正式路线为 Poincare HGCN 与 `mean_std` readout，LP-Lorentz 路径退出当前实现。
- `module4-hpec-classification`: 明确 HPEC 直接消费 Poincare 图级表示，并通过双视角 evidence fusion 参与最终预测。
- `s-deci-model`: 移除 `module34_arch=lp_brain_hpec` 正式分支，保留当前 HGCN-HPEC 主线和 GCN fallback 消融路径。
- `training-test-scripts`: 不再暴露无执行路径的 LP 专用参数；继续记录正式主线指标、几何诊断和完整五折结果。

## Impact

- 当前正式代码不应包含 `layers/lp_brain_hpec_layer.py` 或 `module34_arch=lp_brain_hpec` 分支。
- MDD 默认配置保持模块 3/4 开启、`hgcn_readout_mode=mean_std`、HPEC 多原型与 dual-view evidence fusion。
- 该变更不新增依赖；移除 LP 路径可降低维护成本与训练开销。
- 若未来重新研究 Lorentz 路径，必须建立新 OpenSpec change，并先解决距离注意力、流形原生 readout、bridge 半径压缩和坐标污染问题，再完成相同划分下的完整五折比较。
