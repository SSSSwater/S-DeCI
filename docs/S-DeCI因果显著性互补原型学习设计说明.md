# S-DeCI 因果显著性互补原型学习设计说明

## 1. 目的与边界

本说明描述在既有 S-DeCI 四模块主路线之上新增的训练机制：可靠 TP prototype EMA、因果显著性互补视图和多阶因果可达性编码。它们不替换模块 1 的低频生理特征、模块 2 的时间序列因果图、模块 3 的 Poincare HGCN 或模块 4 的 HPEC 多原型能量分类。

参考 BrainCL 的两项原则：只让可靠样本代表类别原型；主动压制当前最显著 ROI，迫使模型从其他脑区学习互补病理线索。BrainCL 使用静态 FC、Transformer 和欧氏单类别 prototype；S-DeCI 仅借鉴训练逻辑，保持时序因果图与双曲 HPEC 分类不变。

## 2. 数据流

标准视图：

$$
X_{\mathrm{BOLD}}
\rightarrow \text{模块1}
\rightarrow C\in\mathbb{R}^{B\times N\times D}
\rightarrow \text{模块2}
\rightarrow A_{\mathrm{cls}}\in\mathbb{R}^{N\times N}
\rightarrow \text{模块3}
\rightarrow z\in\mathbb{D}^H
\rightarrow \text{模块4}
\rightarrow \hat Y.
$$

互补视图只在训练期产生：标准视图先完整学习 `A_cls` 和节点双曲表示 `H_gcn`；随后仅遮挡送入模块 3 的 `C` 的一部分 ROI，复用相同的 `A_cls` 和相同的模块 3/4 参数：

$$
\tilde C_{b,i}=(1-M_{b,i})C_{b,i},
\qquad
\tilde z_b=f_{3,4}(\tilde C_b,A_{\mathrm{cls}}).
$$

因此遮挡不进入模块 2。原因是模块 2 以完整历史 BOLD 预测未来，若对其输入人为置零，预测误差会把遮挡错误地解释为因果关系变化。

## 3. 因果显著性遮挡

模块 2 图的约定为 `A_cls[parent, child]`。ROI 的拓扑显著性同时看其向外影响和接收影响：

$$
s_i^{\mathrm{topo}}
=
\operatorname{Norm}\left(
\sum_j |A_{\mathrm{cls}}[i,j]|
+
\sum_j |A_{\mathrm{cls}}[j,i]|
\right).
$$

模块 3 标准视图节点点映射到原点切空间后，其范数作为任务表征活跃度：

$$
s_{b,i}^{\mathrm{sem}}
=
\operatorname{Norm}\left(
\left\|\log_0^c(H^{\mathrm{gcn}}_{b,i})\right\|_2
\right).
$$

两者融合：

$$
s_{b,i}=
\beta s_i^{\mathrm{topo}}+(1-\beta)s_{b,i}^{\mathrm{sem}}.
$$

使用 Gumbel-top-k 对高分 ROI 进行带随机扰动的遮挡。显著性会 detach，避免网络通过把所有分数人为拉平来躲避遮挡。遮挡比例在 `warm-up` 后渐进增加：

$$
r(e)=r_{\max}
\min\left(1,\sqrt{\frac{\max(e-e_w,0)}{e_r}}\right).
$$

默认最大比例为 `0.15`，低于 BrainCL 的 `0.3`，因为 AAL116 与 top-k 因果图更稀疏，过大遮挡可能破坏关键传播路径。

## 4. 可靠 TP 多原型 EMA

模块 4 仍保留每类多个 Poincare prototype 与 Busemann/HPEC energy。`reliable_tp_ema` 不使用梯度和总 loss 更新 prototype，也不调用 Sinkhorn。训练 batch 完成 `optimizer.step()` 后，只有满足下式的样本可移动其真实类 prototype：

$$
\mathcal R_k=
\left\{b\mid y_b=k,
\arg\max_r\hat Y_{b,r}=k,
p_{b,k}\ge\theta_c\right\}.
$$

启用互补视图时，再使用标准/互补切空间余弦一致性：

$$
u_b=
\frac{1+\cos(\log_0^c z_b,\log_0^c\tilde z_b)}{2}.
$$

每个可靠样本仅分给其真实类别中 energy 最低的 prototype：

$$
m_b=\arg\min_m E_{b,k,m}.
$$

对分配到 `(k,m)` 的可靠样本做加权切空间均值，并与初始化锚点混合后慢速 EMA：

$$
\bar z_{k,m}^{\tan}=
\frac{\sum_{b\in\mathcal R_{k,m}}u_b\log_0^c(z_b)}
{\sum_{b\in\mathcal R_{k,m}}u_b+\epsilon},
$$

$$
p_{k,m}^{\tan}\leftarrow
\alpha p_{k,m}^{\tan}+(1-\alpha)
\left[(1-\lambda_a)\bar z_{k,m}^{\tan}+\lambda_a a_{k,m}^{\tan}\right].
$$

随后执行半径壳约束并映回 Poincare Ball。样本数低于 `hpec_reliable_min_samples` 时 prototype 保持不动。验证、测试、推理绝不更新 prototype。

`sinkhorn_ema` 仍作为旧对照，`none` 冻结数据驱动更新。新机制的重点是“可信样本代表类别”，不是强制每个 prototype 在每个 batch 都拿到样本。

## 5. 多阶因果可达性编码

模块 3 原本只通过少层 HGCN 获取局部传播。为显式表示多步因果影响，从 `A_cls` 构造前向转移矩阵：

$$
P_{i,j}=
\frac{|A_{\mathrm{cls}}[i,j]|}
{\sum_q|A_{\mathrm{cls}}[i,q]|+\epsilon}.
$$

由于 `A[i,j]` 表示 parent `i` 影响 child `j`，child 应聚合 parent 的信息：

$$
E^{(\ell)}=(P^{\ell})^\top C W_\ell,
\qquad
C'=C+\eta\sum_{\ell=1}^{L}g_\ell E^{(\ell)},
\qquad
g=\operatorname{softmax}(a).
$$

`C'` 才进入原有 HGCN；`A_cls` 不被改写。默认只使用 2-hop，并保留残差 `C`，避免深层传播造成过平滑。

## 6. 损失、开关与诊断

总训练目标只新增一个可选项：

$$
\mathcal L=
\mathcal L_{\mathrm{cls}}
+\mathcal L_{\mathrm{module2}}
+\lambda_{\mathrm{view}}d_c(z,\tilde z)^2
+\mathcal L_{\mathrm{manifold}}.
$$

可靠 TP EMA 不是损失项。它在优化器更新之后独立执行，因此不会为了移动 prototype 再额外引入 CE、SupCon 或中心损失。

常用开关：

| 参数 | 默认 | 含义 |
| --- | ---: | --- |
| `hpec_prototype_update_mode` | `reliable_tp_ema` | `reliable_tp_ema`、`sinkhorn_ema` 或 `none` |
| `hpec_reliable_confidence_threshold` | `0.70` | TP 最小真实类概率 |
| `hpec_ema_start_epoch` | `20` | 开始可靠 EMA 的 epoch |
| `use_causal_complementary_learning` | `0` | 训练期互补视图 |
| `causal_complementary_max_mask_ratio` | `0.15` | 最大 ROI 遮挡比例 |
| `causal_complementary_view_loss_weight` | `0.0` | Poincare 一致性 loss 权重 |
| `causal_complementary_mask_temperature` | `0.7` | 论文式动态 Gumbel 遮挡温度 |
| `causal_complementary_instance_loss_weight` | `0.0` | 双向 InfoNCE 权重，默认关闭 |
| `causal_complementary_instance_temperature` | `0.7` | 双向 InfoNCE 温度 |
| `causal_complementary_masked_ce_weight` | `0.0` | 遮挡分支 HPEC energy CE 权重，默认关闭 |
| `use_multi_hop_causal_encoding` | `0` | 多阶因果可达性编码 |
| `causal_reachability_hops` | `2` | 最大传播 hop |
| `causal_reachability_scale` | `0.25` | 输入残差强度 |

TensorBoard 重点查看：`Complementary/*` 的遮挡比例与双曲距离，`PrototypeUpdate/*` 的可靠 TP 比例、原型更新数与位移，`CausalReachability/*` 的 hop gate 与残差范数。若测试指标未提升，或 prototype 长期只有一个活跃、互补距离持续很大，应分别关闭该机制并与 `sinkhorn_ema` 基线做固定 5-fold/50+ epoch 对比。

## 7. BrainCL 公式对齐与实验结论

为检查早期互补分支“只有前向诊断、没有有效梯度”的问题，补充了两项可选监督。第一项在原点切空间计算标准/遮挡图表示的双向 InfoNCE，对应 BrainCL Eq. (12)：

$$
\mathcal L_{\mathrm{IC}}=-\frac{1}{2B}\sum_i\left[
\log\frac{\exp(\operatorname{sim}(z_i,\tilde z_i)/\tau)}{\sum_j\exp(\operatorname{sim}(z_i,\tilde z_j)/\tau)}+
\log\frac{\exp(\operatorname{sim}(\tilde z_i,z_i)/\tau)}{\sum_j\exp(\operatorname{sim}(\tilde z_i,z_j)/\tau)}
\right].
$$

第二项使用共享 HPEC prototype 的负 energy 作为遮挡分支 logits，并计算交叉熵：

$$
\mathcal L_{\mathrm{mask\_ce}}=
\operatorname{CE}(-E_{\mathrm{HPEC}}(\tilde z),y).
$$

动态遮挡采样改为 BrainCL Eq. (10) 的等价形式：

$$
q=\log\operatorname{softmax}(s/\tau_{\mathrm{mask}})+g,
\qquad g_i=-\log(-\log\epsilon_i),
$$

再对 $q$ 执行 top-k。S-DeCI 仍与原论文存在明确边界：语义显著性来自 HGCN 双曲节点表征而非 Transformer attention inflow；拓扑显著性来自有向时序因果图，而非 Pearson 图上的 PRWS/HWGA；类别监督使用多 HPEC prototype，而非欧氏单 prototype。

MDD/AAL116、5-fold、50 epoch 实验中，mask ratio `0.10/0.20/0.30` 在损失权重为零时结果完全一致，证明仅执行遮挡前向不会形成 complementary learning。启用遮挡分支 CE `0.25` 后，InfoNCE 权重 `0.10/0.30` 的结果分别为 `71.22%/61.95%/65.47%` 与 `71.22%/61.95%/65.54%`（Accuracy/Macro-F1/AUC），均未超过可靠 TP EMA 基线 `71.47%/62.22%/65.57%`。最终 InfoNCE 约为 `3.455`，接近 batch size 32 下均匀匹配的 `log(32)=3.466`，说明当前双曲图表示没有学到有效实例对应关系。因此上述两项监督保留为显式消融开关，默认关闭。
