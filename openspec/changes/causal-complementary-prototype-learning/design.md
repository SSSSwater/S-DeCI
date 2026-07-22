## Context

S-DeCI 的默认主路线为：模块 1 提取低频生理相关节点特征，模块 2 从完整时间序列学习有向 `A_lag` 并生成分类图 `A_cls`，模块 3 在 Poincare Ball 中传播，模块 4 以多 HPEC prototype 的 Busemann/energy evidence 参与最终分类。当前多 prototype 可采用 Sinkhorn-EMA 路径，但 Sinkhorn 的“批内均衡分配”并不等价于“只由可靠类别样本代表类别”，在小样本 fMRI 中可能让错误预测或站点特异样本移动 prototype。

BrainCL 的对应机制是：以预测正确的标准视图样本、其标准/遮挡视图一致性权重和 EMA 更新类别 prototype；同时以高显著 ROI 的动态遮挡迫使网络学习较少被注意到的脑区。该论文的原型在欧氏空间且每类一个，本设计保留 S-DeCI 的 Poincare 多 prototype HPEC 能量，不将 BrainCL 的余弦 prototype 分类器替换进来。

`docs/` 是现有设计的初始参考，不修改。本变更只新增实现后的设计说明和 OpenSpec 增量。所有新机制都必须由显式参数启用，能够在同一数据划分上消融；默认推理只使用标准视图。

## Goals / Non-Goals

**Goals:**

- 用可靠 TP 训练样本独立、缓慢地移动每类多个 HPEC prototype，替换 Sinkhorn 作为默认更新策略。
- 在不改动模块 2 输入和训练目标的前提下，构建共享模块 3/4 的因果显著性互补视图。
- 以一个几何一致性损失测试互补视图是否改善测试泛化，而非堆叠多个监督项。
- 用小 hop 有向因果可达性编码强化模块 3 对模块 2 图的利用，并提供可独立关闭的消融开关。
- 将新增状态、更新质量、图和双曲表示写入 TensorBoard、可视化和 `result.xlsx` 实验记录。

**Non-Goals:**

- 不以 MST、相关矩阵或静态重构替代模块 2 的 Temporal NTS-NOTEARS 因果发现。
- 不在训练或测试时使用真实因果矩阵监督。
- 不对原始 BOLD 时间序列或模块 1 输出到模块 2 的时间序列进行遮挡，不学习第二张因果图。
- 不把 BrainCL 的 Transformer、PRWS 或欧氏单 prototype 分类器整体引入当前默认模型。
- 不让互补分支在评估/推理运行，不新增部署期双分支成本。

## Decisions

### 1. 可靠 TP EMA 取代 Sinkhorn 更新

新增 `hpec_prototype_update_mode`：

- `reliable_tp_ema`：本变更默认；不经过总 loss 或 optimizer 更新 prototype。
- `sinkhorn_ema`：保留现有路径作为 legacy 对照。
- `none`：冻结初始化 prototype，用于消融。

训练 batch 在标准视图前向与分类 loss 的 `optimizer.step()` 后，使用 detach 后的标准双曲点 `z_b` 更新 prototype。可靠集合为：

$$
\mathcal{R}_k=
\left\{
b\ \middle|\ y_b=k,\ \arg\max_r\hat{Y}_{b,r}=k,\ p_{b,k}\ge\theta_c
\right\}.
$$

若互补视图启用，再以切空间余弦一致性定义权重：

$$
u_b=
\frac{1+\cos\!\left(\log_0^c z_b,\log_0^c\tilde z_b\right)}{2},
\qquad
u_b\ge\theta_v.
$$

每个可靠样本分配给其真实类别内 energy 最低的 prototype：

$$
m_b=\arg\min_m E_{b,k,m},\qquad y_b=k.
$$

这不是 Sinkhorn 的强制均衡分配。某 prototype 没有可靠样本时保持不动；诊断会报告长期未更新 prototype，而不是人为把样本分给它。分配到 `(k,m)` 的可靠点以加权切空间均值形成目标：

$$
\bar z^{\tan}_{k,m}=
\frac{\sum_{b\in\mathcal R_{k,m}}u_b\log_0^c(z_b)}
{\sum_{b\in\mathcal R_{k,m}}u_b+\epsilon}.
$$

先同初始化锚点混合，再 EMA 和半径壳投影：

$$
t_{k,m}=(1-\lambda_a)\bar z^{\tan}_{k,m}+\lambda_a a^{\tan}_{k,m},
$$

$$
p^{\tan}_{k,m}\leftarrow
\alpha p^{\tan}_{k,m}+(1-\alpha)t_{k,m},
\qquad
p_{k,m}\leftarrow\operatorname{proj}_c\!\left(\exp_0^c(\operatorname{ShellClip}(p^{\tan}_{k,m}))\right).
$$

默认配置：`hpec_prototype_update_start_epoch=20`、`hpec_reliable_confidence_threshold=0.70`、`hpec_reliable_view_consistency_threshold=0.55`、`hpec_reliable_ema_alpha=0.995`、`hpec_reliable_anchor_weight=0.10`、`hpec_reliable_min_samples=2`。若互补视图关闭，则 `u_b=1` 且不应用一致性阈值。prototype 参数在该模式下 `requires_grad=False`，避免 autograd 和 EMA 同时移动同一参数。

选择原因：它直接对应 BrainCL Eq. (13) 的“TP + 双视图一致性 + EMA”，但保留多 prototype 和 HPEC 流形几何。替代方案是保持 Sinkhorn；其优势是覆盖均衡，缺点是在不可靠样本多时会强迫错误样本参与更新。采用 energy winner 而非再次加入均衡约束，满足“用 TP 样本单独训练移动且替换 Sinkhorn”的要求。

### 2. 因果显著性互补视图仅位于模块 2 之后

标准路径先以完整时间序列执行模块 1/2/3/4，得到 `A_cls`、模块 3 节点表示 `H_gcn` 和标准图点 `z`。显著性由有向信息流与标准视图节点表征活跃度组成：

$$
s_i^{\mathrm{topo}}
=
\operatorname{Norm}\left(
\sum_j|A_{\mathrm{cls}}[i,j]|
+
\sum_j|A_{\mathrm{cls}}[j,i]|
\right),
$$

$$
s_{b,i}^{\mathrm{sem}}
=
\operatorname{Norm}\left(
\left\|\log_0^c(H_{b,i}^{\mathrm{gcn}})\right\|_2
\right),
\qquad
s_{b,i}=\beta s_i^{\mathrm{topo}}+(1-\beta)s_{b,i}^{\mathrm{sem}}.
$$

`Norm` 为节点维 z-score 后的 softmax 或等价稳定归一化。分数与 mask 采样过程均 detach，防止分类器通过操控显著性本身规避遮挡。以 Gumbel-top-k 从高分节点中动态采样掩码 `M_b`，只施加于模块 3 输入特征 `C`：

$$
\tilde C_{b,i}=(1-M_{b,i})C_{b,i},
\qquad
\tilde z_b=f_{3,4}(\tilde C_b,A_{\mathrm{cls}}).
$$

遮挡比例使用 warm-up 后的渐进日程：

$$
r(e)=r_{\max}
\min\left(1,\sqrt{\frac{\max(e-e_w,0)}{e_r}}\right).
$$

默认 `use_causal_complementary_learning=0`、`causal_complementary_max_mask_ratio=0.15`、`causal_complementary_warmup_epochs=20`、`causal_complementary_ramp_epochs=20`、`causal_salience_topology_weight=0.5`。互补分支共享模块 3 和模块 4 权重，并复用标准路径 `A_cls`；不调用第二次模块 2。

选择原因：若遮挡 BOLD 或重新学习 `A_cls`，模块 2 的预测误差会把人为缺失误判成时序因果结构，且两视图图不可比较。替代方案是只按图度数遮挡；其缺点是没有任务表征信息。当前分数使用已有节点表征模长而非额外 attention readout，避免重复引入曾经不稳定的高自由度 attention readout。

### 3. 只增加一个 Poincare 双视图一致性项

互补视图开启后可用以下无标签几何一致性项：

$$
\mathcal L_{\mathrm{view}}
=
\frac{1}{B}\sum_b d_c(z_b,\tilde z_b)^2.
$$

总 loss 扩展为：

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{cls}}
+
\mathcal L_{\mathrm{module2}}
+
\lambda_{\mathrm{view}}\mathcal L_{\mathrm{view}}
+
\mathcal L_{\mathrm{manifold}}.
$$

默认 `causal_complementary_view_loss_weight=0.0`，可与 `use_causal_complementary_learning=1` 独立开关。该项只在训练时计算；未启用时返回零标量和明确诊断值。

选择原因：Poincare distance 与当前双曲 HPEC 空间一致，且只新增一个损失族。替代方案是为遮挡分支再加 CE、SupCon 或中心损失；这些会与最终 CE/HPEC 监督重复，并放大小样本过拟合风险。

### 4. 小 hop 有向因果可达性编码作为模块 3 输入残差

新增 `use_multi_hop_causal_encoding=0`。对 `A_cls[parent, child]` 的非负边强度构造行归一化有向转移：

$$
P_{i,j}=
\frac{|A_{\mathrm{cls}}[i,j]|}
{\sum_q|A_{\mathrm{cls}}[i,q]|+\epsilon}.
$$

为保证 child 收到 parent 的前向因果信息，第 `\ell` 阶编码为：

$$
E^{(\ell)}=\left(P^{\ell}\right)^\top C W_\ell,
\qquad
C'=C+\eta\sum_{\ell=1}^{L}g_\ell E^{(\ell)},
\qquad
g=\operatorname{softmax}(a).
$$

`C'` 进入原有 HGCN，`A_cls` 本身不被改写。默认 `causal_reachability_hops=2`、`causal_reachability_scale=0.25`；`g` 是小型全局 hop gate，不增加节点级 Transformer。`use_multi_hop_causal_encoding`、hops、scale 都可扫描。

选择原因：模块 3 目前的单/少层 HGCN 可能无法显式表达 “ROI i 经两步因果影响 ROI j”。该编码吸收 BrainCL 的多阶拓扑思想，但以模块 2 的有向因果图替代静态 Pearson/随机游走。替代方案是更深 HGCN 或 16-hop PRWS；前者易过平滑，后者在 AAL116 上成本高且会稀释方向性。

### 5. 训练接口、诊断与实验顺序

`S_DeCI` 缓存标准和互补输出，但保持 `forward()` 主返回 logits 的兼容性。训练循环只在 `model.training`、batch 为训练 batch、完成 `optimizer.step()` 后调用 `update_reliable_prototypes()`；验证、测试、可视化前向均严格冻结 prototype。

新增 TensorBoard 分组：`Complementary/*`、`PrototypeUpdate/*`、`CausalReachability/*`。记录 mask 比例、语义/拓扑显著性熵、双曲视图距离、可靠 TP 比例、每类/每 prototype 更新数、prototype 位移、未更新 prototype 数、多 hop gate 与编码范数。最后 epoch 保存标准/遮挡 t-SNE、显著性 ROI 图和 prototype 分配统计；真实标签仅用于训练更新和可视化标注。

实验依次为：当前基线、可靠 TP EMA、可靠 TP EMA + 互补视图（无一致性损失）、再启用一致性损失、最后单独叠加多阶编码。短测只能筛除 NaN、极慢和明显退化；任何默认改动必须在 MDD/AAL116、相同 5-fold、至少 50 epoch 下比较 Accuracy、Macro-F1、AUC、训练时长和稳定性，并写入 `result.xlsx`。

## Risks / Trade-offs

- [TP 早期数量不足，prototype 长期不动] → warm-up 后才更新，配置最少样本数并报告每 prototype 更新计数；必要时降低置信度阈值，但不得在测试阶段更新。
- [energy winner 使多个 prototype 塌缩到一个活跃 prototype] → 不加入 Sinkhorn；改以未更新计数、类内 assignment entropy 和 prototype 间距诊断判断，只有完整消融显示必要时再讨论重初始化策略。
- [双分支训练变慢] → 只重复模块 3/4，不重复模块 1/2；互补分支训练期可开关，推理期始终关闭。
- [遮挡破坏稀疏因果图的关键节点] → 遮挡比例从低值渐进到 0.15，图 `A_cls` 保持不变，且必须与随机遮挡消融比较。
- [多阶编码过平滑或覆盖因果图语义] → 限制 hop 为 1-3、用残差 scale 和全局 gate、保留原始 `C`，不重写 `A_cls`。
- [新增机制只是训练集正则而无测试收益] → 每个功能独立开关，默认不因短测转为主路线；无完整收益即保留为实验对照。

## Migration Plan

1. 先加入参数与严格默认兼容路径，`use_causal_complementary_learning=0`、`use_multi_hop_causal_encoding=0`，并保留 `hpec_prototype_update_mode=sinkhorn_ema`。
2. 实现并单测可靠 TP EMA，再实现互补视图与多阶编码；每一步都先运行单 fold smoke/短训。
3. 以完整 5-fold/50+ epoch 消融决定 MDD 默认入口是否切换到 `reliable_tp_ema`。
4. 若出现训练不稳定、prototype 更新异常或完整测试无稳定收益，回退到 `sinkhorn_ema` 并关闭互补视图、多阶编码和一致性 loss，无需改动数据格式、checkpoint 主结构或模块 2。

## Open Questions

- `reliable_tp_ema` 是否应在完整 MDD 对比胜出后成为所有数据集入口默认值，或仅作为默认候选配置。
- 若一个类别的第二个 prototype 长期没有可靠 TP 分配，是否需要后续独立变更引入低频重初始化；本变更不自动重置它。
- 多阶因果编码在不同 atlas 上的 hop 上限是否统一为 2，还是需按节点数量制定扫描范围；实现阶段先固定默认 2。
