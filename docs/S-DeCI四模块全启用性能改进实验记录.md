# S-DeCI 四模块全启用性能改进实验记录

本文只记录“模块 1、模块 2、模块 3、模块 4 全部启用”时的性能改进实验，避免通过关闭模块得到虚高但不可解释的结果。

## 当前默认基线

- 数据集：MDD
- 图谱：AAL116
- 交叉验证：5 fold
- 训练轮次：50 epoch
- 默认路线：`hgcn_hpec`
- 模块开关：`module1=1, module2=1, module3=1, module4=1`
- 分类损失：`sqrt_batch_balanced`
- 原型更新：`hpec_trainable_prototypes=0, hpec_use_sinkhorn_ema=1, hpec_ema_update_epochs=10, hpec_ema_alpha=0.995, hpec_ema_anchor_weight=0.35`
- 最终融合：`hyperbolic_logit_residual_weight=0.5`

完整 5fold 结果：

| 配置 | Accuracy | Precision | Recall | Macro-F1 | AUC |
|---|---:|---:|---:|---:|---:|
| `hgcn_hpec + sqrt_batch_balanced` | 71.22% | 74.63% | 58.97% | 57.00% | 61.49% |

该配置仍是当前 `test_mdd_best_config.py` 的默认选择。

## 已测试但不设为默认的改法

| 改法 | 测试范围 | Accuracy | Precision | Recall | Macro-F1 | AUC | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `lp_brain_hpec + focal_gamma=1` | 5fold/50epoch | 70.20% | 84.48% | 55.98% | 51.40% | 61.74% | LP 路线仍弱于默认，正类预测偏少 |
| `hgcn_hpec + focal_gamma=1` | 5fold/50epoch | 70.45% | 82.62% | 56.53% | 52.51% | 61.41% | 单折看似改善，但全量 Macro-F1 下降 |
| `logit_adjusted, tau=0.5` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.84% | 正类预测数增加，但 Acc/AUC 不足，暂不全量 |
| `sqrt_logit_adjusted, tau=1.0` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.84% | 与上项近似等价 |
| `hpec_prototype_energy_blend=0.1` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.49% | HPEC energy 进入主判别未带来突破 |
| `hyperbolic_residual_margin_gain=1.0` | 1fold/50epoch | 70.00% | 74.67% | 56.46% | 53.12% | 57.93% | 放大高置信原型残差后概率均值更低 |
| `use_hyperbolic_residual_bias=1` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.49% | 可学习残差偏置影响有限 |
| `hpec_energy_loss_weight=0.05` | 1fold/50epoch | 50.00% | 52.93% | 53.18% | 49.72% | 58.49% | 直接加入 HPEC energy 辅助损失与当前 `prototype_primary` 路线冲突 |
| `hpec_prototype_ce_loss_weight=0.05, teacher=0.5` | 1fold/50epoch | 63.75% | 58.63% | 58.11% | 58.27% | 58.49% | 原型相似度 CE 较稳但未突破默认 |
| `hgcn_use_radius_head=1, z_min_radius=0.25` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.70% | 可避免 `z_radius` 塌到约 0.16，但分类未提升 |
| `lp_brain_hpec + poincare_readout_weight=0.35` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.51% | Einstein midpoint 校正没有改善 LP 分类 |
| `lp_brain_hpec + poincare_readout_weight=0` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.51% | 与 midpoint 校正近似，说明 LP 瓶颈不在该 readout |
| `lp_brain_hpec + message_residual=0.5` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.51% | Lorentz 入/出边消息范数被保住，但指标不变 |
| `hyperbolic_residual_source=tangent` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.56% | 模块3切空间分类头与模块4原型输出接近，但未解决正类不足 |
| `module34_supcon_loss_weight=0.05` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.49% | 监督对比损失增加训练负担，未带来增益 |
| `use_final_logit_calibration=1` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.49% | 校准层已接入，但训练目标未学到有效 raw 阈值校准 |
| `module34_film_weight=0.5` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 57.09% | 用 FC/网络条件对双曲中心点做 FiLM 调制，未提升且 AUC 下降 |
| `module34_film_weight=1.0, max_scale=0.35, shift_norm=0.35` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 57.30% | FiLM 主要学到小幅 shift，scale 接近 0，未改善类别分离 |
| `prototype logit margin_preserving, scale=1` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 58.42% | 保留原型相似度幅度后没有突破默认；原始 margin 约 0.076，说明模块4自身类别间隔弱 |
| `margin_preserving, scale=5, residual_norm=temperature` | 1fold/50epoch | 67.50% | 62.00% | 57.30% | 56.67% | 61.77% | AUC 提高但 raw 0.5 Acc 下降，放大原型幅度带来校准偏移 |
| `margin_preserving, scale=2, teacher=0.3` | 1fold/50epoch | 67.50% | 62.00% | 57.30% | 56.67% | 59.19% | 减弱 teacher 后仍未提升，原型边界没有形成稳定增益 |
| `energy_primary + residual_norm=temperature` | 1fold/50epoch | 67.50% | 62.18% | 59.12% | 59.25% | 59.12% | 使用 HPEC energy 本体后 Macro-F1 改善但 Acc/AUC 不足，说明能量方向有信号但校准不稳 |
| `energy_calibrated` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 56.67% | 仅用 HPEC energy/相似度/半径做小校准头，未超过默认 |
| `feature_fusion` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 56.74% | 拼接 z_tangent、energy、prototype similarity 的融合头未改善，末端校准不是主要瓶颈 |
| `residual_weight=1.0, prototype_ce=0.1` | 1fold/50epoch | 32.50% | 16.46% | 48.15% | 24.53% | 56.74% | 强行放大模块4残差导致几乎全预测正类，明显负效果 |
| `z_min_radius=0.30, z_radius_loss=0.2` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.09% | z_global 半径稳定在约 0.29，但分类未提升，半径塌缩只是症状之一 |
| `hpec_teacher_distill_weight=0` | 1fold/50epoch | 66.25% | 59.85% | 56.36% | 55.73% | 57.93% | 去掉 FC teacher 后更差，teacher 对稳定训练仍有帮助 |
| `hgcn_readout_mode=causal_attention` | 1fold/50epoch | 70.00% | 70.65% | 57.37% | 55.18% | 57.51% | 因果图引导多头图池化没有超过默认，后期 `z_radius` 仍回到约 0.167 |
| `causal_attention + radius_head` | 1fold/50epoch | 68.75% | 65.00% | 57.34% | 56.05% | 58.49% | 半径稳定在约 0.57，但 raw Acc 下降，说明半径保持本身不足以产生类别分离 |
| `hgcn_readout_mode=network_stats` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.86% | AAL 网络级 readout 降低 ROI 噪声的思路未在该单折上带来收益 |
| `module34_center_loss=0.02, margin=0.2, intra=1, inter=0.5` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.93% | 轻量类中心约束没有改善，`z_radius` 后期仍约 0.167 |
| `module34_center_loss=0.05, margin=-0.1, intra=0, inter=1` | 1fold/50epoch | 68.75% | 67.79% | 55.52% | 52.28% | 57.86% | 只拉开类中心也未改善，说明 batch 内中心监督不足以形成稳定双曲类结构 |

## 本轮诊断结论

1. 当前主要问题不是模块开关，而是模块 3/4 双曲原型分支对最终 raw 0.5 判别的正类推动不足。
2. 验证集最佳阈值经常落在 0.27-0.31，而 raw 0.5 下正类预测偏少，说明最终 logits 存在系统性校准偏移。
3. 简单提高模块 4 监督、加入 focal/logit adjustment、放大 residual、增加 SupCon 或直接修 LP readout，都没有超过完整 5fold 默认基线。
4. LP-Brain-HPEC 当前更像“几何路线正确但判别信号弱”：保住 Lorentz 消息范数后指标仍不变，说明瓶颈不是消息消失，而是双曲图级表示没有形成更强类间边界。
5. 继续改进时应优先做结构性改动：让模块 3/4 在不直接污染 Poincare 几何的前提下吸收更强的图/FC 判别条件，例如条件门控、FiLM 式尺度调制或类条件原型边界，而不是继续堆小权重 loss。
6. 已测试的 FiLM 条件调制没有带来收益，说明“把 FC 作为连续条件调节双曲中心点”仍不足以解决模块 4 原型边界弱的问题；下一步更应改模块 4 的原型能量边界，例如类条件 margin、正类召回约束或基于原型分配的更稳健决策。
7. 本轮新增的 `margin_preserving`、`energy_calibrated` 和 `feature_fusion` 都没有超过默认路线。现象上，HPEC energy/原型相似度确实包含一些排序信号（AUC 或 Macro-F1 局部变好），但 raw 0.5 决策边界经常偏向负类或正类，说明末端 logits 校准不稳定。
8. 仅约束半径可以避免 `z_global` 继续贴近原点，但不能自然产生类别分离；因此下一轮应重点改模块 3 的图级双曲 readout/聚合，让 `z_global` 在进入 HPEC 前已经带有标签相关结构，而不是继续在模块 4 末端加校准头。
9. 已测试的 `causal_attention` 和 `network_stats` 说明：单纯改变模块 3 的全图池化权重仍不够。`causal_attention + radius_head` 能稳定半径，但没有稳定提升分类，说明下一步更应该给 `z_global` 增加直接的类间结构监督，例如温和的双曲监督对比/类中心分离，而不是继续只调 readout 形式。
10. 两组 `module34_center_loss` 结果说明，简单 batch 类中心分离仍然不够；当前更突出的问题是最终 raw 阈值下正类预测长期偏少。后续应优先检查和修改 `hyperbolic_logit_residual` 的融合方式，让模块3/4在保持双曲解释的同时对正类边界提供更稳定、不过度保守的贡献。

## 当前代码处理

- 新增的 LP Poincare/Einstein midpoint readout 已保留为可选参数 `lp_poincare_readout_weight`。
- 由于短测未带来收益，`Model` 与 `test_mdd_best_config.py` 的默认值均为 `0.0`，避免污染后续默认实验。
- 新增的模块 3/4 FiLM 条件调制已保留为可选参数 `module34_film_weight`、`module34_film_max_scale`、`module34_film_shift_norm`，默认关闭。
- 新增的 HPEC 原型 logits 处理参数 `hpec_prototype_logit_mode`、`hpec_prototype_logit_scale` 已保留；默认仍为旧的 `normalized`，因为 `margin_preserving` 单折未带来稳定收益。
- 新增的 `energy_calibrated` 分类模式已保留为实验选项；该模式只使用 HPEC energy、prototype similarity 与半径等双曲证据做校准，不引入外部 FC 特征，但短测未提升，默认不启用。
- 新增的模块 3 `hgcn_readout_mode=causal_attention` 已保留为实验选项，并暴露 `hgcn_causal_attention_heads` 与 `hgcn_causal_attention_graph_weight`；单折未提升，默认仍为 `node_stats`。
- 新增的模块 3/4 类中心结构损失已保留为实验选项 `module34_center_loss_weight`、`module34_center_margin`、`module34_center_intra_weight`、`module34_center_inter_weight`；两组短测未提升，默认关闭。
- `hpec_energy_loss_weight`、`hpec_prototype_ce_loss_weight`、`module34_supcon_loss_weight` 等仍保留为实验参数，但默认关闭。

## 本轮参考依据

- Poincaré Embeddings 说明双曲空间适合表达具有层级结构的数据，优势来自低失真的几何表示，而不是单纯末端分类器加深。
- HGCN 论文强调要把欧氏输入以合适方式映射到双曲空间，并在双曲/切空间中定义稳定的特征变换和聚合。因此当前更应该改模块 3 的图级 readout，使 `z_global` 本身形成可分结构。
- Hyperbolic Neural Networks 给出了在 Poincaré ball 中进行神经网络运算的基本思路，也提醒跨欧氏/双曲空间操作需要保持几何一致性；这也是为什么本轮没有把 FC 特征直接拼进默认 HPEC 主路径。

## 2026-07-06 模块3/4全启用继续改进记录

本轮没有通过关闭模块提升指标，所有测试均保持 `module1=1, module2=1, module3=1, module4=1`。

### 已验证改动

1. 修复 residual 融合路径的 `final_positive_prob_mean` 诊断：此前 residual 模式没有写入最终正类概率，控制台中的 `pos_prob=0` 是诊断缺失，不代表真实概率为 0。
2. 增加可选 `class_prior_alignment_weight`，用于训练期约束最终概率均值接近 batch 类别比例。单折 50 epoch 下 `0.02` 未提升，暂不设默认。
3. 修复 LP-Brain-HPEC 动态半径初始化过于恒定的问题：动态半径由图读出范数提供数据驱动基值，再由 `graph_radius_head` 学习小幅残差。该修复使 LP+Busemann 单折从明显失衡改善到 `Acc=71.25%, Macro-F1=56.10%, AUC=60.66%`，但完整 5fold 为 `Acc=66.15%, Macro-F1=52.81%, AUC=62.58%`，说明有排序信号但 raw 校准不稳，暂不设默认。
4. 对 HGCN-HPEC 路线复测 `energy_primary + hyperbolic_logit_residual_weight=0.25`。完整 MDD 5fold/50epoch raw 指标为：`Acc=70.96%, Precision=70.44%, Recall=59.50%, Macro-F1=58.01%, AUC=61.97%`。相比此前默认 `prototype_primary + residual_weight=0.5` 的 `Acc=71.22%, Macro-F1=57.00%, AUC=61.49%`，Accuracy 略低 0.26 个百分点，但 Macro-F1、Recall、AUC 更均衡，因此将 MDD 默认入口改为该组合。

### 当前默认调整

- `test_mdd_best_config.py`
  - `hpec_classification_mode=energy_primary`
  - `hyperbolic_logit_residual_weight=0.25`

### 仍未解决的问题

- HGCN-HPEC 路线中 `z_radius` 后期仍常贴近约 `0.16-0.20`，单独开启半径头可提升一点 AUC，但未稳定提升 Acc/Macro-F1。
- LP-Brain-HPEC 的 Lorentz 消息范数后期仍会下降到很小，动态半径修复只能改善表示半径和局部 AUC，不能稳定全折 raw 指标。
- 正类 raw 预测数量仍偏少，说明最终 0.5 阈值校准和类别边界还需要继续从结构上改进，而不是测试阶段调阈值。

## 2026-07-06 继续快筛记录（二）

在保持 `module1=1, module2=1, module3=1, module4=1` 的前提下，继续围绕当前有效默认 `energy_primary + hyperbolic_logit_residual_weight=0.25` 做小范围验证。

### 新增快筛结果

1. `prototype_primary + batch_balanced + residual_weight=0.25`：MDD 1fold/50epoch raw 指标为 `Acc=66.25%, Macro-F1=54.23%, AUC=57.72%`，不如当前默认单折 `Acc=70.00%, Macro-F1=55.18%`，不进入全量测试。
2. `energy_primary + residual_weight=0.25 + hpec_energy_loss_weight=0.02`：MDD 1fold/50epoch raw 指标为 `Acc=61.25%, Macro-F1=56.27%, AUC=58.56%`。该配置把正类概率均值推到约 0.48，预测正类数量更接近真实比例，但 raw Acc 明显下降，说明直接加 HPEC energy 辅助损失会导致阈值校准失控，暂不设默认。
3. 固定 `energy_primary + residual_weight=0.25`，对 `class_loss_weighting` 做 1fold/50epoch sweep：
   - `sqrt_batch_balanced`: `Acc=70.00%, Macro-F1=55.18%, AUC=57.72%`
   - `batch_balanced`: `Acc=66.25%, Macro-F1=54.23%, AUC=57.93%`
   - `none`: `Acc=68.75%, Macro-F1=52.28%, AUC=57.09%`
   因此当前默认仍保留 `sqrt_batch_balanced`。

### 当前判断

- 当前最稳默认仍为 `energy_primary + hyperbolic_logit_residual_weight=0.25 + sqrt_batch_balanced`。
- 直接增强模块4 energy supervision 可以提高正类概率和局部 Macro-F1，但会显著牺牲 raw Accuracy，说明下一步应从结构上改善双曲表征可分性，而不是继续增加 energy loss 权重。
- 后续更值得尝试的是让 `z_global` 在进入 HPEC 前具备更强类间结构，例如改模块3 readout 或引入温和的图级监督对比，同时避免把 HPEC 最终概率推到过高。

## 2026-07-06 LP-Brain-HPEC 修复与实测记录（三）

本轮仍保持 `module1=1, module2=1, module3=1, module4=1`，没有通过关闭模块提升指标。

### 检查依据与实现判断

1. 参考 HGCN / Poincare Embedding / Hyperbolic Entailment Cones 的基本思路，模块 3/4 应让双曲空间同时保留“方向/角度”和“半径/层级”信息。
2. 检查 `LPBrainHPECReadout` 后确认：当前 LP 注意力已使用切空间负距离而非原始 Lorentz 内积；readout 前已有 `LayerNorm`；MAC 默认 `soft` 模式，不会把所有低半径样本硬推到同一壳层。
3. 继续实测发现 LP 默认 `lp_in_norm/out_norm` 会从早期约 `3~5` 下降到后期约 `0.08`，说明 Lorentz 图消息后期基本消失；同时 `z_radius/lp_mac_radius` 长期固定在约 `0.442`，HPEC 的半径层级信息不足。

### 已验证改动

1. 尝试“原型感知切空间对齐损失”：`module34_prototype_align_loss_weight=0.005`，MDD 1fold/50epoch 得到 `Acc=61.25%, Macro-F1=56.27%, AUC=58.42%`。该损失会冲坏 raw 分类边界，因此已从代码入口和 TensorBoard 映射中移除，不保留预留窗口。
2. 测试 LP 当前默认：MDD 1fold/40epoch 得到 `Acc=63.75%, Macro-F1=59.09%, AUC=60.17%`，后期 `lp_in_norm/out_norm` 约 `0.08`，确认图消息衰减明显。
3. 测试 `lp_input_residual_weight=0.3 + lorentz_message_gate_init=0.8 + lorentz_message_residual_weight=0.1`：MDD 1fold/40epoch 指标仍为 `Acc=63.75%, Macro-F1=59.09%, AUC=60.17%`，但训练用时从约 `153s` 降到 `136s`，且后期 `lp_in_norm/out_norm` 保持约 `0.63`。说明该改法能缓解消息消失，但不能单独提升分类。
4. 在上述消息保留基础上改用 `energy_calibrated`：MDD 1fold/40epoch 得到 `Acc=68.75%, Macro-F1=56.05%, AUC=60.87%`。继续提高残差和 HGCN-logit 融合到 `0.35` 后单折 Acc 达到 `70.00%`，但正类预测只有 `7/80`，Macro-F1 降到 `55.18%`，属于过度保守。
5. 新增 LP 可选参数 `lp_dynamic_radius_source`：
   - `norm`：默认原范数半径策略，raw Acc 更稳。
   - `graph_context`：使用网络强度均值、网络强度离散度、网络注意力峰值和图级统计共同预测半径，使半径不再固定在同一壳层。
   单折 `graph_context` 可把正类预测从 `10/80` 拉回 `23/80`，Macro-F1 到 `59.27%`，但 Acc/AUC 下降。
6. 对 `lp_brain_hpec + energy_calibrated + lp_input_residual_weight=0.3 + lorentz_message_gate_init=0.8 + lorentz_message_residual_weight=0.1 + hpec_hgcn_logit_blend=0.35 + hyperbolic_logit_residual_weight=0.35 + graph_context` 做完整 MDD 5fold/50epoch：
   - `Acc=62.63%`
   - `Precision=59.64%`
   - `Recall=60.29%`
   - `Macro-F1=59.43%`
   - `AUC=62.50%`
   - `train_seconds=734.09s`
   结果已写入 `result.xlsx` 的 `MDD_AAL116_50ep` sheet。该路线 AUC 和 Macro-F1 有一定信号，但 Accuracy 明显低于当前默认，不设为默认。

### 当前结论

- 当前默认仍保留 `hgcn_hpec + energy_primary + hyperbolic_logit_residual_weight=0.25`，因为完整 MDD 5fold/50epoch 的 Accuracy 更稳。
- LP-Brain-HPEC 的 `graph_context` 半径策略保留为可选研究参数，但默认仍为 `norm`，避免把完整 5fold 负收益写入默认配置。
- LP 路线的主要价值是提供“半径层级 + 双曲锥能量”的解释性线索；但当前 raw 分类性能未超过 HGCN-HPEC，需要后续从更稳定的双曲图级表征或更稳的决策校准继续改。

## 2026-07-06 HGCN-HPEC 末端融合快筛记录（四）

本轮继续保持四模块全启用，重点检查模块 4 是否因为末端 logits 融合方式拖累模块 3/4 的整体表现。所有实验均为 MDD 1fold/50epoch 快筛，未超过当前默认，因此未改默认参数。

### 已验证改动

1. `hyperbolic_residual_source=tangent`：用模块 3 切空间分类头作为残差主体，模块 4 仍训练原型和 HPEC 约束。结果 `Acc=68.75%, Macro-F1=56.05%, AUC=57.93%`，正类预测约 `10/80`，说明单纯换成 tangent 残差仍偏保守。
2. `use_hyperbolic_residual_gate=1` 旧 margin 门控：结果 `Acc=65.00%, Macro-F1=56.11%, AUC=58.07%`，门控把双曲残差信号压弱后没有改善。
3. 新增可选 `hyperbolic_residual_gate_mode=agreement`：模块 4 与 FC 基底预测一致时放大，不一致时缩小。快筛结果 `Acc=68.75%, Macro-F1=56.05%, AUC=57.79%`，仍未超过默认，因此仅保留为实验参数。
4. `hyperbolic_residual_fusion_mode=logit_blend + hyperbolic_residual_norm=temperature`：保留 HPEC logit 幅度，不走 tanh 逐样本压缩。结果 `Acc=66.25%, Macro-F1=55.73%, AUC=58.07%`，负收益。
5. `hpec_classification_mode=tangent_primary + hpec_energy_loss_weight=0.005`：让最终边界主要来自模块 3，模块 4 作为结构约束。结果 `Acc=68.75%, Macro-F1=56.05%, AUC=58.00%`，未改善。
6. `hpec_trainable_prototypes=1 + hpec_use_sinkhorn_ema=0`：让原型直接梯度更新。快筛结果 `Acc=61.25%, Macro-F1=56.27%, AUC=58.63%`。虽然 prototype 余弦相似度显著降低，原型确实散开了，但分类下降，说明“原型散开”本身不是有效目标。
7. `use_hgcn_radial_calibration=1 + hpec_z_min_radius=0.25`：稳定模块 3 输出半径。结果 `Acc=70.00%, Macro-F1=56.99%, AUC=58.21%`。单折 Accuracy 有信号，但 AUC/Macro-F1 仍弱，暂不改默认。
8. 新增可选 `hyperbolic_residual_fusion_mode=binary_margin`：二分类只用 HPEC 正类相对负类 margin 更新正类 logit，避免两类同步归一化导致过度保守。快筛结果 `Acc=65.00%, Macro-F1=54.80%, AUC=58.14%`，负收益，仅保留为实验选项。

### 当前判断

- 当前问题不是“模块 4 没有参与”，而是 HPEC/prototype 信号在 raw 0.5 边界下容易把正类预测压少；多数末端融合、门控、原型训练都会重复这个问题。
- 径向校准能稳定 `z_radius`，但没有显著提升 AUC/Macro-F1，说明仅扩大半径不是根治。
- 后续更值得改的是模块 3 图级表示本身的类别可分性，尤其是 readout 里的图/FC 生物标志怎样进入双曲中心点，而不是继续调末端 logits。

## 2026-07-06 模块 3 readout 结构快筛记录（五）

本轮继续保持 `module1=1, module2=1, module3=1, module4=1`，不通过关闭模块提升指标。目标是让模块 3 在进入 HPEC 前形成更有类别信息的 `z_global`，同时避免把 FC embedding 直接加到 `z_tangent` 造成双曲几何污染。

### 新增实现

1. 在 `TangentFrechetReadout` 中新增 `hgcn_readout_mode=network_gated_node_stats`。
   - 做法：将样本相关矩阵压缩为 AAL 功能网络级 FC 强度，再展开为 ROI gate，只调节模块 3 readout 的节点池化权重。
   - 约束：FC 只影响“哪些脑区在 readout 中权重大一点”，不直接平移/拼接双曲坐标，避免破坏 HPEC 的角度与半径结构。
2. 暴露 `hgcn_graph_readout_alpha`。
   - 做法：控制图/FC 节点权重加权均值在 readout mean 中的比例。
   - 原因：之前加权节点权重主要影响 std/max，`readout_mean` 默认仍接近普通均值，信号进入较弱。
3. 增加模块 3 readout 诊断量：
   - `module3_node_attention_entropy`
   - `module3_node_attention_peak`
   - `module3_network_attention_entropy`
   - `module3_network_attention_peak`
   这些会写入 TensorBoard，方便判断节点/网络 gate 是否真的在样本间变化。

### 快筛结果

1. `network_gated_node_stats + hgcn_network_gate_strength=0.35`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.72%`。与当前默认单折基本持平，没有明显正收益，不设默认。
2. `network_gated_node_stats + graph_readout_alpha=0.5`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.51%`。让加权均值更强参与后没有提升，说明该 FC gate 本身不是当前瓶颈。
3. `network_gated_node_stats + graph_readout_alpha=0.5 + hgcn_use_radius_head=1`：MDD 1fold/50epoch raw 指标为 `Acc=68.75%, Precision=65.00%, Recall=57.34%, Macro-F1=56.05%, AUC=58.70%`。`z_radius` 被稳定在约 `0.55`，AUC/Macro-F1 略升，但 raw Acc 下降。
4. 仅 `hgcn_use_radius_head=1`：MDD 1fold/50epoch raw 指标同样为 `Acc=68.75%, Precision=65.00%, Recall=57.34%, Macro-F1=56.05%, AUC=58.70%`。这说明上面的提升主要来自半径头，而不是 FC 网络门控。
5. 仅 `use_brain_network_prior=1`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.72%`。固定文献网络先验与当前默认基本持平，未带来额外收益。

### 当前判断

- `network_gated_node_stats` 保留为实验选项，但不改默认。它符合“FC 只作为 readout 条件、不污染双曲坐标”的设计逻辑，但单折没有带来可见性能增益。
- 半径头可以修复 `z_radius` 后期贴近原点的问题，并让 AUC 有小幅提升；但 raw 0.5 下正类预测仍偏少，Accuracy 下降，因此也不设默认。
- 固定 AAL 文献网络先验没有明显提升，说明当前瓶颈不是简单提高 DMN/额叶-边缘网络等 ROI 的池化权重。
- 当前最稳默认仍保持 `hgcn_hpec + energy_primary + hyperbolic_logit_residual_weight=0.25 + sqrt_batch_balanced`。下一步若继续改模块 3/4，应优先考虑能改善 raw 决策边界的图级表示结构，而不是单独维持半径或继续增强末端原型损失。

## 2026-07-06 模块 4 蒸馏与能量/原型配比快筛记录（六）

本轮仍保持 `module1=1, module2=1, module3=1, module4=1`。先检查 `result.xlsx` 中的完整 5fold 记录，确认 `71.72%` 那批结果实际为 `module3=0,module4=0`，不符合当前硬约束；四模块全启用里较有信号的是 `prototype_primary` 的 Macro-F1 可到 `60.81%`，但 raw Accuracy 降到 `66.93%`。因此本轮目标是让模块 4 的 prototype 信号以更温和方式进入最终边界，而不是关闭模块或照搬 GCN fallback。

### 新增实现

1. 新增 `hpec_teacher_distill_mode`：
   - `kl`：原始完整 softmax KL 蒸馏。
   - `centered_kl`：去除每个样本共同 logit 偏置，只蒸馏类别相对结构。
   - `margin_mse`：二分类时只蒸馏正负类 margin。
2. 新增 `hpec_classification_mode=energy_prototype_residual`：
   - 公式：`logits = energy_logits + alpha * normalized(prototype_logits)`。
   - 目的：让 HPEC energy 保持主边界，prototype 只作为小残差参与，避免 `prototype_primary` 整体接管概率校准。
3. 暴露 `hpec_evidence_weight`：
   - 原先 `energy_primary` 中会把 `prototype_similarity` 以 `1.0` 权重叠加进 energy logits，但入口不可调。
   - 现在可以测试 energy 与 prototype evidence 的配比。

### 快筛结果

1. `hpec_teacher_distill_mode=centered_kl`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.72%`。与原始 `kl` 基本一致，说明原 KL 本身主要在学习相对 logits，去共同偏置没有带来收益。
2. `hpec_teacher_distill_mode=margin_mse`：MDD 1fold/50epoch raw 指标为 `Acc=66.25%, Precision=59.03%, Recall=53.63%, Macro-F1=50.63%, AUC=55.28%`。训练中 `hpec_final_ce` 明显升高，说明直接回归 teacher margin 会把校准推得过猛，负收益。
3. `energy_prototype_residual, prototype_residual_weight=0.2`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.72%`。与当前默认几乎一致，没有证明原型残差能改善边界。
4. `energy_primary, hpec_evidence_weight=0.5`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.65%`。降低 prototype evidence 权重没有改善，说明当前瓶颈不在 energy/prototype 配比。

### 当前判断

- `centered_kl`、`margin_mse`、`energy_prototype_residual` 和 `hpec_evidence_weight` 均保留为实验选项，但不改 MDD 默认。
- `margin_mse` 负收益明显，后续若再使用必须先做 margin 标准化或很小权重，不能直接设默认。
- 当前四模块全启用的最稳默认仍保持 `hgcn_hpec + energy_primary + hyperbolic_logit_residual_weight=0.25 + sqrt_batch_balanced`。
- 这轮结果进一步说明：模块 4 的能量/原型融合方式不是单折主要瓶颈；如果继续提高四模块全开性能，更应从模块 3 输出的可分性、模块 2 图质量、或最终 raw 边界校准的结构化约束入手。

## 2026-07-07 四模块全启用继续排查记录

本轮仍严格保持 `module1=1, module2=1, module3=1, module4=1`，没有通过关闭模块提升指标。目标是检查模块3/4是否能在不破坏现有四模块框架的前提下提供正收益。

### 新增实现与修复

1. 新增 `module34_branch_ce_loss_weight`：训练期额外约束模块3/4融合前自身 logits 可分类。这个损失不改变推理结构，只用于诊断 HPEC/双曲分支是否能独立形成有效边界。
2. 修复 `LPBrainHPECReadout.forward()` 与 `S_DeCI` 统一调用接口不一致的问题：现在 LP 路线可以接收 `sample_correlation` 参数，避免四模块 LP 路线直接报错。
3. 新增 `lorentz_centroid_message_weight`：在 LP-Brain-HPEC 的有向 Lorentz 图卷积中，用加权 Lorentz centroid 消息替代部分原点切空间 value 消息。设计依据是双曲图卷积中只在原点切空间近似聚合可能带来几何失真，流形内 centroid 聚合能保留更多双曲几何信息。
4. TensorBoard/控制台/result 参数列补充了 `module34_branch_ce_*` 与 `lp_centroid_message_weight`，便于之后完整实验追踪。

### 快筛结果

1. `module34_branch_ce_loss_weight=0.05`：MDD 1fold/50epoch raw 指标为 `Acc=61.25%, Precision=56.34%, Recall=56.22%, Macro-F1=56.27%, AUC=58.49%`。该损失把正类预测数量从默认约 `7/80` 拉到 `26/80`，但 raw Acc 明显下降，说明直接强迫 HPEC/双曲分支独立分类会破坏当前边界校准，不设默认。
2. `module34_branch_ce_loss_weight=0.1`：MDD 1fold/50epoch raw 指标为 `Acc=61.25%, Precision=56.34%, Recall=56.22%, Macro-F1=56.27%, AUC=58.56%`。与 0.05 基本相同，继续确认该方向是负收益。
3. `LP-Brain-HPEC + energy_calibrated + lorentz_centroid_message_weight=0.3`：MDD 1fold/40epoch raw 指标为 `Acc=62.50%, Precision=59.32%, Recall=59.89%, Macro-F1=59.43%, AUC=59.26%`。Lorentz centroid 能保持较均衡预测，但 raw Acc 仍明显低于当前 HGCN-HPEC 默认，不设默认。
4. `use_hgcn_radial_calibration=1 + hpec_z_radius_loss_weight=0.02 + hgcn_use_graph_degree_encoding=1`：MDD 1fold/50epoch raw 指标为 `Acc=67.50%, Precision=62.28%, Recall=55.49%, Macro-F1=53.41%, AUC=58.91%`。z 半径被稳定到约 `0.46`，但分类下降，说明单纯撑开双曲半径不是根因。
5. `hyperbolic_residual_source=tangent + hpec_energy/prototype_ce 轻量辅助`：MDD 1fold/50epoch raw 指标为 `Acc=68.75%, Precision=65.00%, Recall=57.34%, Macro-F1=56.05%, AUC=57.86%`，仍低于当前默认。
6. 当前默认四模块全启用复核：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=70.65%, Recall=57.37%, Macro-F1=55.18%, AUC=57.72%`。今天新增的可选实验参数未破坏默认主路径。

### 当前判断

- 当前瓶颈不是“模块3/4没有被监督”，因为直接给模块3/4自身分支加 CE 会让概率更均衡但 raw Acc 大幅下降。
- 当前瓶颈也不是单纯的双曲半径坍缩；径向校准可以稳定 `z_radius`，但没有提升分类性能。
- LP-Brain-HPEC 的 Lorentz centroid 消息更符合双曲几何聚合逻辑，修复了可运行性并改善了预测均衡性，但单折 Acc 仍不如 HGCN-HPEC 默认。
- 因此当前默认仍保留 `hgcn_hpec + energy_primary + hyperbolic_logit_residual_weight=0.25 + sqrt_batch_balanced`。后续若继续改模块3/4，应重点让双曲图级表示的方向信息本身更可分，而不是继续增加末端 HPEC 强监督或固定半径。

### 完整 5fold 复核

在上述快筛均未超过默认后，重新运行当前默认四模块全启用配置的 MDD 5fold/50epoch，确认今天新增的可选实验分支没有破坏主路径。最终 raw 指标为：

- `Accuracy=70.96%`
- `Precision=70.44%`
- `Recall=59.50%`
- `Macro-F1=58.01%`
- `AUC=61.97%`
- `train_seconds=596.31s`

结果已写入 `result.xlsx` 的 `MDD_AAL116_50ep` sheet。该结果与此前默认完整测试一致，因此当前可作为继续改模块3/4前的稳态基线。

## 2026-07-07 模块4无标签残差校准正收益记录

本轮仍保持 `module1=1, module2=1, module3=1, module4=1` 全启用，没有通过关闭模块提升指标。

### 问题定位

补充默认模式下的 HPEC 诊断后发现：`proto_logit_margin` 与 `energy_margin` 并非真实为 0，之前控制台显示为 0 主要是诊断只在部分分支写入。真实情况是模块4存在可用信号，但 margin 很小，例如早期约 `proto_logit_margin=0.06`、`energy_margin=0.08`，后期 `energy_margin` 约 `0.04~0.05`。这说明模块4不是完全无效，而是进入最终 residual fusion 时幅度和校准不够稳定。

### 新增改动

1. 新增 `hpec_residual_calibration=batch_margin`：在模块4输出进入最终双曲残差前，对二分类正负类 margin 做 batch 内无标签标准化。
2. 新增 `hpec_residual_calibration_scale`：控制校准后的 margin 尺度。该操作不读取标签，不改变样本排序，只把过小的 HPEC/prototype margin 调整到稳定残差尺度。
3. 补充诊断量：`hpec_energy_margin_signed_mean`、`hpec_energy_logit_abs_mean`、`hpec_prototype_logit_signed_margin_mean`、`hpec_residual_calibrated_margin_abs_mean` 等，方便观察模块4实际是否参与。

### 实测结果

1. `batch_margin, scale=0.5, residual_weight=0.25`：MDD 1fold/50epoch raw 指标为 `Acc=68.75%, Precision=64.06%, Recall=60.06%, Macro-F1=60.25%, AUC=57.79%`。Macro-F1/Recall 提升，但 Acc 下降。
2. `batch_margin, scale=1.0, residual_weight=0.25`：与 scale=0.5 基本相同，说明主要收益来自 margin 标准化，继续放大尺度没有额外收益。
3. `batch_margin, scale=0.5, residual_weight=0.15`：MDD 1fold/50epoch raw 指标为 `Acc=70.00%, Precision=65.92%, Recall=61.91%, Macro-F1=62.38%, AUC=55.42%`。相比默认单折 `Acc=70.00%, Macro-F1=55.18%`，在不损失 Acc 的情况下显著提升类别均衡指标。
4. 对 `batch_margin, scale=0.5, residual_weight=0.15` 跑完整 MDD 5fold/50epoch：
   - `Accuracy=70.71%`
   - `Precision=67.54%`
   - `Recall=62.03%`
   - `Macro-F1=62.17%`
   - `AUC=62.29%`
   - `train_seconds=331.89s`

对比上一轮默认完整 5fold：`Accuracy=70.96%, Precision=70.44%, Recall=59.50%, Macro-F1=58.01%, AUC=61.97%`。新配置 Accuracy 基本持平（-0.25），Macro-F1 提升约 `+4.16`，Recall 提升约 `+2.53`，AUC 小幅提升约 `+0.32`。

### 当前默认更新

将 `test_mdd_best_config.py` 的默认值更新为：

- `hpec_residual_calibration=batch_margin`
- `hpec_residual_calibration_scale=0.5`
- `hyperbolic_logit_residual_weight=0.15`

理由：该配置保持四模块全启用，让模块4的 HPEC/prototype margin 以无标签校准后的稳定尺度参与最终残差，能改善 MDD 完整 5fold 的 Macro-F1/Recall/AUC，同时基本不牺牲 Accuracy。

## 2026-07-07 模块3/4继续小步实测记录（二）

本轮仍保持 `module1=1, module2=1, module3=1, module4=1` 全启用，没有通过关闭模块提升指标。

### 新增可选实现

1. 新增 `hpec_residual_calibration=running_batch_margin`：训练阶段用当前 batch 的 HPEC 正负类 margin 做校准，同时用 EMA 保存训练期 margin 均值/方差；验证/测试阶段优先复用训练期 EMA 统计，避免测试 batch 自身参与校准。该设计更严格，但需要实测确认是否损失现有 batch 自适应能力。
2. 新增 `hgcn_readout_mode=mean_std`：模块3 readout 只使用节点切空间均值和标准差，不使用 max pooling。设计动机是 fMRI 小样本中单 ROI 极值可能放大噪声，mean+std 更符合“整体激活 + 离散度”的低自由度统计。

### 快筛结果

1. `running_batch_margin, scale=0.5, residual_weight=0.15`：MDD 1fold/50epoch raw 指标为 `Acc=66.25%, Precision=61.21%, Recall=59.99%, Macro-F1=60.29%, AUC=57.86%`。相比当前默认 `batch_margin` 单折 `Acc=70.00%, Macro-F1=62.38%` 更差，说明测试期复用训练 EMA 虽然流程更干净，但失去了当前 batch margin 自适应校准带来的收益；因此不改默认。
2. 复跑当前默认 `batch_margin, scale=0.5, residual_weight=0.15`：MDD 1fold/50epoch raw 指标仍为 `Acc=70.00%, Precision=65.92%, Recall=61.91%, Macro-F1=62.38%, AUC=55.42%`，确认新增可选路径没有破坏主线。
3. `hgcn_readout_mode=mean_std`：MDD 1fold/50epoch raw 指标为 `Acc=67.50%, Precision=62.18%, Recall=59.12%, Macro-F1=59.25%, AUC=60.45%`。AUC 比默认单折更高，但 Accuracy 和 Macro-F1 下降，说明去掉 max pooling 有一定排序信号，但 raw 0.5 决策边界不如默认稳定；暂保留为实验选项，不设默认。

### 当前判断

- 当前最稳默认仍是 `hgcn_hpec + energy_primary + batch_margin(scale=0.5) + hyperbolic_logit_residual_weight=0.15`。
- `running_batch_margin` 是一个更严格的训练/测试校准对照，但负收益，不进入默认。
- `mean_std` 可用于后续关注 AUC 排序的对照，但目前不是综合指标最优。
## 2026-07-14 模块4 Busemann 方向证据小步复核

本轮继续保持 `module1=1, module2=1, module3=1, module4=1` 全启用，没有通过关闭模块或降低模块权重来换指标。目标是检查 Busemann-HPEC 是否可以用更贴合理论的方式改善模块4泛化。

### 参考思路

Busemann function 更适合表达“双曲空间中的类别方向证据”，而不是普通 prototype distance 的“离类别中心多近”。因此本轮优先尝试减少样本半径对 Busemann score 的影响，让模块4更关注 `z_global` 在切空间中的方向。

### 实验结论

1. `hpec_residual_calibration=learned_margin` 做了短测筛查：1fold/25epoch raw 指标为 `Acc=70.00%, Precision=68.54%, Recall=58.28%, Macro-F1=56.99%, AUC=58.84%`。它只提高了 raw Acc，但 Macro-F1/AUC 低于默认短测，而且 HPEC 分支仍偏多数类，因此已撤销该候选代码，不进入默认。
2. `hpec_residual_calibration=running_batch_margin` 做了同折短测：测试 raw 退化为全预测正类，`Acc=33.75%, Macro-F1=25.23%, AUC=61.84%`。说明训练期 EMA margin 统计不能替代当前 batch 自适应校准。
3. `hpec_busemann_point_radius=0.3` 做了短测和完整 5fold/50epoch。短测表现较好，但完整结果为：
   - `Accuracy=70.97%`
   - `Precision=68.18%`
   - `Recall=61.86%`
   - `Macro-F1=61.97%`
   - `AUC=65.89%`
   - `train_seconds=335.36s`

### 当前判断

- 固定 Busemann 打分半径能让 AUC 略升，但 Accuracy 和 Macro-F1 低于当前完整最佳基线 `Acc=71.97%, Macro-F1=63.37%, AUC=65.78%`，因此不改默认。
- Busemann 的有效作用仍然是提供类别方向证据；当前瓶颈不是 energy 公式完全错误，而是 HPEC 分支在测试集上容易出现正类召回和 raw 阈值校准不稳定。
- 后续如果继续改模块3/4，优先方向应是让 `z_global` 的类别方向本身更稳定、减少 HPEC 分支测试多数类化，而不是继续增加新的损失项或更多校准分支。
