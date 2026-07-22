# S-DeCI 模型修改证据台账

## 使用规则

本文件记录模型设计、训练入口和损失函数的实验性修改，作为后续改造的唯一决策索引。

- **已验证正向**：在完整交叉验证中相对对应基线改善，或直接修复已定位的实现错误。相关能力必须保留在模型中，并提供明确开关用于消融。
- **已验证负向**：在相同数据、折分、epoch 和主指标口径下完成完整交叉验证后无收益或退化。只保留本文件中的反例记录，不保留模型分支、损失、训练参数或 TensorBoard 映射。
- **证据不足**：仅完成单折快筛、指标有冲突，或配置与当前主线已不一致。它不是当前能力，也不能作为负向结论；需要重新做完整对照后才可升级为正向或负向。

主指标统一使用每个 fold **最后一个 epoch** 的 raw 测试集 `accuracy`、`precision`、`recall`、`macro_f1` 和 `roc_auc`，不用训练过程中的最佳 epoch 替代最终结果。

## 当前主线

当前默认目标是小样本静息态 fMRI 分类：模块 1 提取低频生理表征，模块 2 以过去时间窗预测未来 BOLD innovation 学习有向滞后图，模块 3 在 Poincare ball 聚合因果图，模块 4 用 HPEC 多原型给出双曲分类证据。模块 3/4 的图级证据与 FC 局部结构证据在 logits 层进行联合决策。

下列开关是当前正式消融接口：

| 能力 | 训练入口参数 | 默认 | 依据 |
| --- | --- | --- | --- |
| 模块 1 生理低频表征 | `--use-deci-module1 1 --module1-feature-mode alff` | 开 | 跨数据集 ALFF/fALFF 分布差异明显低于原始时序统计，见 `ALFF跨数据集差异比较.md`。 |
| 模块 1 消融 | `--use-deci-module1 0` 或 `--module1-feature-mode raw/deci` | 关/ALFF | 用于检验低频生理锚定本身的贡献。 |
| 时序因果模块 2 | `--use-causal-module2 1 --causal-learning-target temporal_sem` | 开 | temporal innovation、identity decoder、弱 `A0`、有向图传播放入完整 MDD 五折后，较修复前提升 Acc、Macro-F1、AUC，并减少训练时间，见 `S-DeCI模块12时序因果修复实验-2026-07-17.md`。 |
| 因果图消融 | `--use-causal-module2 0` | 关 | 下游自动使用样本相关矩阵；用于检查时序图相对 FC 的增益。 |
| 因果到分类图的连续门控 | `--classification-graph-source causal_soft_masked_fc` | 开 | 保留样本 FC 边权并由时序因果候选边提供有向支持，避免仅二值图或仅 FC。 |
| 模块 3 双曲图传播 | `--use-hgcn-module3 1` | 开 | 当前正式路线使用 `hgcn_hpec` 的 Poincare HGCN。 |
| 模块 4 多原型 HPEC | `--use-hpec-module4 1 --hpec-prototype-update-mode reliable_tp_ema` | 开 | 使用可靠真阳性样本的慢 EMA 更新多原型，避免原型自由追逐训练集。 |
| 模块 3/4 消融 | `--use-hgcn-module3 0 --use-hpec-module4 0` | 关 | 退化到 GCN fallback，检验双曲层级与原型边界的增益。 |
| 有向 GCN fallback | `--gcn-fallback-directional-propagation 1 --gcn-fallback-readout-mode mean_std` | 开 | 入边、出边及其差异分别编码；完整模块 1/2 实验显示其与因果图联合能提升 AUC。 |

## 已验证正向修改

| 修改 | 证据 | 保留位置 |
| --- | --- | --- |
| ALFF/fALFF 生理特征替代仅原始统计 | 六数据集比较中，ALFF/fALFF 的跨数据集均值距离约为原始统计的 `50.1%`，协方差距离约为 `6.5%`。 | `models/S_DeCI.py` 模块 1，参数见上表。 |
| 时间序列 innovation 预测、无偏 identity decoder、弱同时间残差 `A0` | MDD/AAL116 五折/50 epoch：后续加入有向入/出传播与 `mean_std` 读出后为 `Acc=71.97%`、`Macro-F1=61.55%`、`AUC=65.43%`，且训练时间低于修复前版本。 | `layers/temporal_sem_causal_learner.py`、`layers/gcn_fallback_layer.py`。 |
| 因果图与 FC 的联合图分类 | 完整五折中联合路径相对 FC-only 提升 Acc 和 AUC；但 graph-only 波动较大，因此不将因果图当作唯一分类证据。 | `classification_graph_source` 与 GCN fallback。 |
| 可靠真阳性多原型 EMA | 在小样本情形下避免可训练原型直接追逐训练集；它是当前 HPEC 原型更新默认策略。 | `layers/hpec_energy_layer.py`，`--hpec-prototype-update-mode reliable_tp_ema`。 |

## 已验证负向修改：已从当前代码移除

| 修改 | 完整实验结果 | 反例原因 | 处理 |
| --- | --- | --- | --- |
| 高频频谱跨样本混合 | MDD/AAL116、5 fold、50 epoch：开启 `p=0.25` 为 `Acc=66.91%`、`Macro-F1=58.78%`、`AUC=61.85%`；关闭为 `68.43%`、`59.87%`、`62.16%`。 | 虽保留低频与相位，但仍向训练样本注入了与个体/疾病状态耦合的高频变化；该数据集上不能把它视作纯站点风格。 | 删除频谱混合实现、CLI、TensorBoard 标量和训练入口。 |
| focal 分类损失 | MDD 五折/50 epoch：`hgcn_hpec + focal_gamma=1` 为 `Acc=70.45%`、`Macro-F1=52.51%`，低于对应基线。 | 对困难样本的额外强调破坏了当前 raw 概率边界的校准，召回未换来 Macro-F1 改善。 | 删除 focal 参数与损失分支。 |
| LP-Brain-HPEC 替代模块 3/4 | MDD 五折/50 epoch：LP 完整路线为 `Acc=62.63%`、`Macro-F1=59.43%`、`AUC=62.50%`，准确率显著低于 Poincare HGCN-HPEC 主线，且训练更慢。 | Lorentz 消息范数虽可局部修复，图级判别边界仍弱；不能以几何叙事替代实际分类能力。 | 删除 LP readout、架构选择与专用参数；结论保留为后续跨流形设计反例。 |
| 因果显著性互补视图与 InfoNCE/遮挡 CE | MDD 五折/50 epoch：最佳互补组合没有超过可靠 TP EMA 基线，InfoNCE 接近 `log(batch_size)`，未学到有效实例对应关系。 | 遮挡后的第二视图没有形成有意义的互补因果语义，额外前向和损失只增加复杂度。 | 删除互补视图、遮挡、InfoNCE、遮挡 CE 及其参数。 |

## 单折未证实修改

以下项目不再作为默认能力或论文结论。其现有代码将逐步只保留真正有独立复验价值的部分；再次尝试前必须先登记完整对照计划。

| 修改方向 | 单折现象 | 不应得出的结论 |
| --- | --- | --- |
| 模块 3/4 FiLM 注入 FC 条件 | AUC 与分类未改善。 | 不能据此认定 FC 完全无用，只能说明直接调制双曲中心点不合适。 |
| HPEC energy 辅助损失、prototype CE、SupCon、batch center loss | 多数设置使 raw 边界更不稳定或没有改善。 | 不能把“增加监督”当作提升双曲可分性的通用方案。 |
| 固定原型半径、可训练原型、tangent-direction 原型 | 原型可分散但分类反而下降。 | 原型之间更远不等价于类别边界更好。 |
| radius head、径向校准、causal attention、network stats | 可改善半径或注意力诊断，但未稳定提高最终指标。 | 几何量看起来正常不等价于模型具有判别信息。 |
| 修改最终 logit 融合、margin 放大、单独 energy 分类 | 局部 AUC/Macro-F1 有时上升，raw accuracy 常下降。 | 不应用测试阈值或 logit 后处理掩盖训练期边界不足。 |

## 后续新增修改的准入流程

1. 在本文件增加“候选修改”，明确模块、参数、理论依据和预期诊断变化。
2. 先做 1 fold 快筛，仅用于排除数值错误和明显退化。
3. 通过快筛后，再固定 seed、MDD/AAL116、5 fold、50 epoch，与当前主线做完整对照。
4. 指标、耗时、折间波动与可视化共同判断；只依据最后 epoch raw 测试指标归类。
5. 正向项目保留实现与开关；负向项目删除实现，仅在本文件保留反例与原因。
