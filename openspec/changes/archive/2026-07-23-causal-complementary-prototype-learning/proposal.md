## Why

该变更最初同时研究可靠 true-positive (TP) 多原型 EMA、因果显著性互补遮挡和多阶因果可达性编码。完整 MDD/AAL116 5-fold、50 epoch 消融表明：可靠 TP EMA 适合作为当前 HPEC prototype 的慢更新策略；互补遮挡、InfoNCE 与 masked CE 未超过可靠 TP EMA 基线，且增加额外前向和损失复杂度；多阶可达性编码保留为默认关闭的独立消融能力。因此最终规范必须区分“正式保留”“可选研究”和“完整负向后移除”三种状态。

## What Changes

- 保留 `hpec_prototype_update_mode=reliable_tp_ema`：只使用预测正确且置信度达标的训练样本，在 `optimizer.step()` 后以无梯度 EMA 移动每类多个 Poincare prototype。
- 保留 `sinkhorn_ema` 和 `none` 作为明确对照；可靠 TP EMA 默认不得调用 Sinkhorn，也不得作为总 loss 项。
- 保留默认关闭的多阶因果可达性编码：从 `A_cls[parent,child]` 构造小 hop 有向传播，以残差方式增强模块 3 输入，不改写因果图。
- 移除训练期因果显著性互补视图、Gumbel ROI mask、Poincare 双视图一致性、InfoNCE 和 masked CE 的正式能力、训练参数与有效执行路径。
- 在结果和证据台账中保留互补学习的完整负向结论，避免后续重复堆叠同类监督。

## Capabilities

### New Capabilities

- `reliable-ema-prototype-updates`: 独立于总 loss 的可靠 TP 多原型 EMA 更新。
- `multi-hop-causal-reachability-encoding`: 默认关闭的小 hop 有向因果可达性输入编码。

### Removed Capabilities

- `causal-salience-complementary-learning`: 完整五折未获得稳定正收益，退出当前正式代码与训练接口。

### Modified Capabilities

- `multi-prototype-hpec-classification`: 默认 prototype 更新使用可靠 TP EMA，保留 Sinkhorn/none 对照。
- `module3-hgcn-readout`: 只保留可选多阶因果输入增强，不再运行标准/互补双分支。
- `module4-hpec-classification`: 只调度标准视图的独立 prototype 更新，不再接收互补点。
- `s-deci-model`: 保持 `forward()` 返回标准最终 logits，并暴露训练后 prototype 更新接口。
- `training-test-scripts`: 保留可靠 EMA 与多阶编码参数和诊断，移除互补学习参数与标量。

## Impact

- 影响 `models/S_DeCI.py`、`layers/hyperbolic_gcn_layer.py`、`layers/hpec_energy_layer.py`、`exp/exp_classification_CV.py` 和训练入口。
- 不新增外部依赖。
- 当前默认推理只执行标准模块 3/4 路径；prototype 数据驱动更新仅发生在训练期。
- 互补学习的负向结果保留在 `docs/S-DeCI模型修改证据台账.md`，但不保留可调用参数和训练 loss。
