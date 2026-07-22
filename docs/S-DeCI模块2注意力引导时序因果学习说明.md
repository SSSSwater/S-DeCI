# S-DeCI 模块2注意力引导时序因果学习说明

## 背景

当前模块 2 的核心目标不是从静态节点特征里重构一个相关图，而是利用时间顺序学习 `过去脑区 -> 未来脑区` 的有向关系。原有 `nts_notears` 已经使用历史窗口预测未来时间点，本次新增的 `attn_nts_notears` 在这个路径上加入轻量多头注意力，用来建模不同 lag 和不同脑区之间的候选影响。

注意力权重本身不直接等于因果图。实现中先用 lag-window attention 描述动态依赖，再用可学习结构门控 `G_lag` 把动态依赖沉淀为稳定图 `A_lag`。

## 数据流与公式

设模块 1 输出或保留的时间序列为：

$$
X\in\mathbb{R}^{B\times T\times N}
\quad\text{或}\quad
X\in\mathbb{R}^{B\times T\times N\times D}.
$$

其中 $B$ 为 batch size，$T$ 为时间长度，$N$ 为 ROI 数，$D$ 为每个 ROI 的特征维。attention-guided temporal learner 的详细计算过程如下。

1. 构造历史窗口。对每个可预测时间点 $t>L$，取：

$$
\mathcal{H}_{b,t}
=
\left[
x_{b,t-1},
x_{b,t-2},
\ldots,
x_{b,t-L}
\right].
$$

预测目标为：

$$
y_{b,t}=x_{b,t}.
$$

这一步来源于 Granger causality 的“过去预测未来”原则。设计原因是：如果 ROI $i$ 的历史值对 ROI $j$ 的未来值有稳定预测贡献，则 $i\rightarrow j$ 比静态相关更接近时序因果解释。

2. 生成 child query 和 parent key/value。第 $h$ 个 attention head 的 child query 写作：

$$
q_{b,t,j}^{(h)}
=
W_q^{(h)}\,
\phi_q\left(\mathcal{H}_{b,t,j}\right),
\qquad
q_{b,t,j}^{(h)}\in\mathbb{R}^{d_h}.
$$

第 $\ell$ 个 lag、parent ROI $i$ 的 key/value 写作：

$$
k_{b,t,\ell,i}^{(h)}
=
W_k^{(h)}\,
\phi_k\left(x_{b,t-\ell,i}\right),
\qquad
v_{b,t,\ell,i}^{(h)}
=
W_v^{(h)}\,
\phi_v\left(x_{b,t-\ell,i}\right).
$$

其中 $\phi_q,\phi_k,\phi_v$ 是轻量线性投影或 MLP。该步骤参考 Transformer multi-head attention 的 query-key-value 机制，但这里的 token 不是文本位置，而是 “lag + parent ROI”。设计原因是：不同 lag 和不同 parent ROI 对同一个 child ROI 的贡献不应被同一个标量参数固定死，attention 可以作为候选影响强度的动态估计。

3. 计算 attention score。对 head $h$、lag $\ell$、parent $i$、child $j$：

$$
s_{b,t,\ell,i\rightarrow j}^{(h)}
=
\frac{
\left(q_{b,t,j}^{(h)}\right)^\top
k_{b,t,\ell,i}^{(h)}
}{
\sqrt{d_h}
}.
$$

这一步来源于 scaled dot-product attention。除以 $\sqrt{d_h}$ 的原因是避免维度增大时内积方差过大，使 softmax 过早变成近似 one-hot。

4. 对 parent 维度归一化，并乘以结构门控：

$$
\alpha_{b,t,\ell,i\rightarrow j}^{(h)}
=
\frac{
\exp\left(s_{b,t,\ell,i\rightarrow j}^{(h)}\right)
}{
\sum_{r=1}^{N}
\exp\left(s_{b,t,\ell,r\rightarrow j}^{(h)}\right)
},
$$

$$
g_{\ell,i,j}
=
\sigma(\theta_{\ell,i,j}),
$$

$$
e_{b,t,\ell,i\rightarrow j}^{(h)}
=
\alpha_{b,t,\ell,i\rightarrow j}^{(h)}
g_{\ell,i,j}.
$$

其中 $\alpha$ 表示样本和时间相关的动态注意力，$g$ 表示跨样本共享的稳定结构门控。设计原因是：attention 本身容易随 batch 波动，不能直接当作稳定因果图；结构门控把反复出现的候选影响沉淀为共享图参数。

5. 聚合得到 lag-specific adjacency：

$$
A_{\mathrm{lag}}^{(\ell)}[i,j]
=
\frac{1}{B(T-L)H_g}
\sum_{b=1}^{B}
\sum_{t=L+1}^{T}
\sum_{h=1}^{H_g}
e_{b,t,\ell,i\rightarrow j}^{(h)}.
$$

其中 `A_lag` 表示跨时间主因果图，方向语义是 `parent` 的过去影响 `child` 的未来。

## A0 的作用

`A0` 表示同时间片残余依赖，用于吸收历史窗口无法解释的同步残差。它不是模块 3 默认使用的分类图。

由于 `A0` 没有时间箭头提供方向，因此 DAGMA/NOTEARS 风格无环约束只作用于 `A0`；`A_lag` 不强制 DAG，因为跨时间方向已经由过去指向未来。

## 损失函数

模块 2 的 auxiliary loss 保持简洁：

$$
\mathcal{L}_{\mathrm{module2}}
=
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0).
$$

预测损失为：

$$
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{B(T-L)N}
\sum_{b=1}^{B}
\sum_{t=L+1}^{T}
\sum_{j=1}^{N}
\rho\left(\hat{x}_{b,t,j}-x_{b,t,j}\right).
$$

稀疏项为：

$$
\mathcal{L}_{\mathrm{sparse}}
=
\frac{1}{LN(N-1)}
\sum_{\ell=1}^{L}
\sum_{i\ne j}
A_{\mathrm{lag}}^{(\ell)}[i,j]
+
\rho_0
\frac{1}{N(N-1)}
\sum_{i\ne j}
A_0[i,j].
$$

lag 平滑项为：

$$
\mathcal{L}_{\mathrm{smooth}}
=
\frac{1}{(L-1)N^2}
\sum_{\ell=2}^{L}
\left\|
A_{\mathrm{lag}}^{(\ell)}
-
A_{\mathrm{lag}}^{(\ell-1)}
\right\|_F^2.
$$

$A_0$ 的 DAG 约束为：

$$
h(A_0)
=
\operatorname{tr}
\left(
\exp(A_0\odot A_0)
\right)
-N.
$$

不加入真实因果矩阵监督，不加入 raw attention 对比损失，也不加入 prototype 或反事实类额外窗口。真实因果图如果存在，只用于训练后评估和可视化。

**来源与原因：**

- $\mathcal{L}_{\mathrm{pred}}$ 来自 Granger causality 和 Temporal NTS-NOTEARS，用预测未来保证边方向来自时间。
- $\mathcal{L}_{\mathrm{sparse}}$ 来自 NOTEARS 稀疏结构学习思想，避免小样本 fMRI 中每个 ROI 都连到所有 ROI。
- $\mathcal{L}_{\mathrm{smooth}}$ 约束相邻 lag 的因果图连续变化，符合 BOLD 信号低频、缓慢变化的特点。
- $h(A_0)$ 来自 NOTEARS 的可微 DAG 约束，只用于同时间片残余依赖，因为跨时间图 $A_{\mathrm{lag}}$ 已由时间箭头提供方向。

## 下游图选择

模块 3 默认读取 `A_lag.mean(dim=0)` 或等价的 `a_effective`。当 `classification_graph_source == "blend"` 时，分类图为：

$$
A_{\mathrm{learned}}
=
\frac{1}{L}
\sum_{\ell=1}^{L}
A_{\mathrm{lag}}^{(\ell)}.
$$

$$
A_{\mathrm{cls},b}
=
(1-\beta)A_{\mathrm{learned}}
+
\beta S_b.
$$

其中 $S_b$ 是第 $b$ 个样本的相关矩阵，$\beta$ 对应 `module2_sample_correlation_blend`。设计原因是：$A_{\mathrm{learned}}$ 提供跨样本共享的有向时序结构，$S_b$ 保留个体功能连接差异；二者融合能避免小样本下学习图过强接管分类。

由于 attention 的 `softmax_parent` 会让单条边天然带有约 `1/N` 的尺度，代码保留了 `temporal_attention_graph_scale` 参数用于调节传给分类图的尺度。MDD 完整 5-fold 测试中，强行放大到 `116` 会降低泛化，因此当前默认保持 `1.0`。

## 当前 MDD 完整 5-fold 结果

结果已写入根目录 `result.xlsx` 的 `MDD_AAL116_50ep` sheet。当前有效对比如下：

| 方法 | 关键配置 | Accuracy | Precision | Recall | Macro F1 | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `nts_notears` | Temporal NTS-NOTEARS 主路线，代表性最好结果 | 65.90% | 61.81% | 56.18% | 55.48% | 58.22% |
| `attn_nts_notears` | heads=2, dim=8, dropout=0, scale=1 | 63.37% | 56.19% | 53.98% | 52.76% | 58.14% |
| `attn_nts_notears` | heads=2, dim=8, dropout=0, scale=116 | 61.34% | 55.02% | 53.15% | 52.53% | 56.55% |
| `attn_nts_notears` | heads=1, dim=4, dropout=0.1, scale=1 | 66.42% | 60.35% | 57.48% | 56.81% | 60.63% |
| `attn_nts_notears` | heads=1, dim=4, dropout=0.1, init=-3.5, scale=1 | 62.87% | 57.63% | 55.33% | 54.66% | 59.97% |

这组 `attn_nts_notears + heads=1 + head_dim=4 + dropout=0.1 + graph_scale=1` 是 attention-guided temporal learner 的代表性对照配置，不再作为当前默认路线。当前默认仍使用 `nts_notears`，原因是默认路线需要保持 Temporal NTS-NOTEARS 的可解释性和损失结构稳定；attention 只作为候选边打分增强，用于比较“注意力先验是否能提升时序因果图学习”。

## 诊断结论

- attention 图可以学习到时序预测关系，训练日志中的 `A_lag directionality` 会随 epoch 增大。
- `heads=2, dim=8` 自由度偏高，训练集很快接近 100%，但测试泛化不稳定。
- 直接把 attention 图放大到和 FC 同量级会让下游图过度受不稳定 learned graph 影响，完整 5-fold 结果下降。
- 轻量 attention 加 dropout 是当前更稳的选择。
