# S-DeCI 模块 2 静态因果学习改造说明（legacy）

本文档记录模块 2 早期“静态节点特征图学习”版本的工程实现，主要用于历史追踪和 legacy/debug 对照。当前论文主路线不再以本文档为准；正式设计请以 `docs/新模块设计.md` 和 `docs/S-DeCI模块2时间序列SEM改造说明.md` 为准。

当前主路线已经改为 Temporal NTS-NOTEARS：模块 2 使用模块 1 输出的时间序列，通过历史窗口预测未来时间点来学习跨时间因果图 $A_{\mathrm{lag}}$。因此本文档中的静态 reconstruction loss、静态 feature DAG loss、`dag_sampling` 和样本残差图只用于旧实验复现，不应写入默认训练方案。

## 目标

旧版本模块 2 的目标是从模块 1 输出的节点特征 `C` 中学习一个静态有向图，并把该图传递给后续 HGCN 或 GCN fallback。为了便于对比和回退，历史实现保留旧的 `nts_notears` 和 `dag_sampling`，并新增 `dagma_logdet`。

该旧路径的核心重构公式为：

$$
\hat{C}_{b,j}
=
f_j\left(
\sum_{i=1}^{N}A_{i,j}C_{b,i}
\right),
$$

其中 $C_{b,i}$ 是样本 $b$ 的第 $i$ 个 ROI 静态节点特征，$A_{i,j}$ 表示从 ROI $i$ 到 ROI $j$ 的静态边。该设计来源于 NOTEARS/NTS-NOTEARS 的可微结构学习思想，但因为它不显式使用时间先后关系，容易落入 Markov equivalence class，因此已降级为 legacy。

## 图学习方法

- `nts_notears`：沿用 NTS-NOTEARS 风格的 positive/negative 第一层权重，通过第一层权重的范数得到共享图 `A_shared`，DAG penalty 默认使用稳定的 analytic 约束。
- `dagma_logdet`：复用 NTS-NOTEARS 的可学习边参数，但 DAG penalty 使用 log-det 形式，便于和 DAGMA 思路对齐。该方法通过谱半径保护 `sI - A*A` 的数值稳定性。
- `dag_sampling`：使用 Sinkhorn permutation 和上三角 order mask 构造可微 DAG，保留 hard straight-through permutation 开关。

## 输入归一化

新增 `--causal_input_norm`：

- `none`：默认值，完全保持旧行为。
- `feature_zscore`：对每个样本、每个 ROI 的特征维做 z-score。
- `batch_node_zscore`：按 batch 与特征维对每个 ROI 做 z-score。

旧静态重构损失使用归一化后的输入作为目标，避免归一化开启时 `C_hat` 与目标口径不一致：

$$
\mathcal{L}_{\mathrm{recon}}
=
\frac{1}{BND}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\lVert \hat{C}_{b,i}-C_{b,i}\rVert_2^2.
$$

该公式只描述历史静态图路径。当前 Temporal NTS-NOTEARS 主路线使用的是：

$$
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{B(T-L)N}
\sum_{b,t,j}
\rho(\hat{x}_{b,t,j}-x_{b,t,j}),
$$

其中 $\rho(\cdot)$ 可为 MSE、Huber 或与 BOLD/ALFF 低频能量一致的 `bold_alff` 预测误差。

## 共享图与样本残差图

模块 2 输出现在包含：

- `A_shared`：全训练集共享的因果图。
- `A_delta`：可选样本级残差图，形状为 `[B, N, N]`。
- `A_effective`：下游实际使用的图，等于 `A_shared + A_delta` 后截断为非负，默认关闭残差时等于 `A_shared`。

当前命令行只保留一个常用开关：

- `--use_sample_graph_residual`：是否启用样本级残差图。

残差图幅度、稀疏正则、偏离共享图正则等属于低层实现细节，当前固定为内部默认值，避免日常训练命令过于拥挤。

## Loss 与诊断

本节只描述 legacy/static-feature 对照路径，不描述当前默认 Temporal NTS-NOTEARS 路线。旧静态模块 2 的目标为：

$$
\mathcal{L}_{\mathrm{legacy}}
=
\lambda_{\mathrm{recon}}\mathcal{L}_{\mathrm{recon}}
+
\lambda_{\mathrm{dag}}h(A)
+
\lambda_{1}\|A\|_1
+
\lambda_{\mathrm{sample}}\mathcal{L}_{\mathrm{sample}}.
$$

其中 $\mathcal{L}_{\mathrm{recon}}$ 是静态节点特征重构误差，$h(A)$ 是 NOTEARS/DAGMA 风格的静态 DAG 约束，$\|A\|_1$ 控制图稀疏，$\mathcal{L}_{\mathrm{sample}}$ 控制样本残差图幅度。该路径保留的原因是便于复现早期实验和与 temporal 路线对照；它不利用时间先后关系，因此容易受到 Markov equivalence class 的方向不确定性影响，不应作为默认训练方案。

细粒度 warmup 与 DAGMA log-det 的内部数值参数当前不再暴露为 `run_cv.py` 的命令行参数，默认使用稳定配置。

训练日志会额外显示：

- legacy 路径下原始与加权的 reconstruction、DAG、L1、sample graph 正则。
- DAG penalty 相关谱半径。
- 图平均权重、方向性比例和样本残差图幅度。

## 推荐对比方式

低成本调参时可先比较：

```bash
python run_cv.py --data MDD --data_path dataset/MDD --protocol AAL116 --model S-DeCI --iterations 1 --max_folds 1 --train_epochs 20 --causal_graph_method nts_notears
python run_cv.py --data MDD --data_path dataset/MDD --protocol AAL116 --model S-DeCI --iterations 1 --max_folds 1 --train_epochs 20 --causal_graph_method dagma_logdet --causal_input_norm feature_zscore
python run_cv.py --data MDD --data_path dataset/MDD --protocol AAL116 --model S-DeCI --iterations 1 --max_folds 1 --train_epochs 20 --causal_graph_method dagma_logdet --causal_input_norm feature_zscore --use_sample_graph_residual 1
```

若要回退到旧行为，使用对应 legacy 参数即可：`causal_input_norm=none`、`use_sample_graph_residual=0`。如果目标是当前论文主路线或正式训练，请优先使用 `causal_graph_method=nts_notears` 的 temporal 实现，并检查日志中是否出现 `temporal_pred_loss`、`a_lag_mass` 和 `directionality`。
