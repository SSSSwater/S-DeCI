# S-DeCI 模块2时间序列 NTS-NOTEARS 改造说明

本文档记录当前模块 2 的正式设计：在保留 NOTEARS 可微因果图学习思想的基础上，将输入从静态节点特征 `C: [B, N, D]` 改为时间序列窗口预测。

## 核心目标

- 模块 2 使用模块 1 处理后的时间序列，输入形状为 `[B, T, N]` 或 `[B, T, N, D]`。
- 使用历史窗口预测相对上一时刻的变化，再加回持久性基线 $x_{t-1}$。
- 主因果图为 `A_lag: [lag_order, N, N]`，表示过去脑区到未来脑区的有向影响。
- `A0: [N, N]` 只表示同一时间片的残余依赖，不作为主要分类图。
- 下游模块 3/GCN 默认使用 `A_lag.mean(dim=0)`，可与样本相关矩阵按比例融合。

这样做的原因是：静态 DAG 学习容易只恢复 Markov 等价类，边方向不可靠；时间序列预测可以利用时间箭头，使跨时间边天然具备方向。

## 与 NTS-NOTEARS 的关系

当前实现不删除 NOTEARS，而是把 NOTEARS 的正负权重分解迁移到时间序列版本：

- 每个 lag 都有一组正负分解边权。
- `A_lag` 由第一层边权范数得到，语义为 `A[parent, child]`。
- 预测 decoder 使用历史节点信号和 lag-specific 边权生成 `x_hat`。
- `A_lag` 使用 L1 稀疏和 lag 平滑约束。
- `A0` 使用弱 DAGMA / NOTEARS 风格无环约束。

## Loss 构成

正式 temporal 路径的模块 2 auxiliary loss 为：

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

默认 `innovation` 目标与最终重建为：

$$
\Delta x_t=x_t-x_{t-1},
\qquad
\hat{x}_t=x_{t-1}+\widehat{\Delta x}_t.
$$

其中基础预测项可写作：

$$
\mathcal{L}_{\mathrm{base}}
=
\frac{1}{BT'N}
\sum_{b,t,j}
\rho
\left(
\hat{x}_{b,t,j}-x_{b,t,j}
\right),
$$

$\rho(\cdot)$ 可为 MSE、Huber 或当前默认的 `bold_alff` 组合损失。跨时间 innovation 预测公式为：

$$
\widehat{\Delta x}_{t,j}
=
\sum_{\ell=1}^{L}
\sum_{i=1}^{N}
A_{\mathrm{lag}}^{(\ell)}[i,j]x_{t-\ell,i}
+
r_{t,j}.
$$

其中：

- `temporal_pred_loss` 是主损失，用于让因果图参与时间序列预测。
- `temporal_sparse_loss` 防止因果图变成稠密相关图。
- `temporal_smooth_loss` 防止相邻 lag 的图剧烈跳变。
- `causal_dag_loss` 只约束 `A0`，不约束 `A_lag`，因为跨时间边已经由过去指向未来。
- decoder 默认使用 `identity` 且局部线性层不带 bias，保留零均值 BOLD 的正负动态，避免退化为 ROI 常数均值。
- `A0` 默认只以 `0.03` 的弱尺度修正 lag 预测，避免同时间残余图压倒跨时间主图。

旧静态特征 reconstruction loss、静态 feature DAG loss、`dag_sampling` 等路径仅作为 legacy/debug 对照，不再是正式默认路径。

## 图输出和可视化

训练和可视化需要区分以下图：

- `A_lag`: 每个 lag 的跨时间因果图。
- `A_lag_mean`: 模块 2 的主因果图，也是默认分类图来源。
- `A0`: 同时间片残余依赖图。
- `A_effective`: `A_lag_mean` 加可选样本残差后的图。
- `A_cls`: 实际送入模块 3 或 GCN fallback 的分类邻接矩阵。

默认 `classification_graph_source=causal_soft_masked_fc` 时：

$$
G_{i,j}=\frac{A_{\mathrm{lag\_mean}}[i,j]}
{\max_r A_{\mathrm{lag\_mean}}[r,j]+\epsilon},
\qquad
A_{\mathrm{cls},b}=|S_b|\odot[\alpha+(1-\alpha)G],
$$

其中 $S_b$ 是样本相关矩阵，$G$ 是按 child 归一化的连续有向 gate，$\alpha$ 对应 `module2_graph_residual_alpha`，默认 `0.10`。该方式保留个体 FC 边权，同时让跨时间因果图决定边的方向性支持强度。

如果 `classification_graph_source=learned` 或 `causal`，则直接使用学习到的 temporal 因果图。

## 常用命令

```bash
python run_cv.py --model S-DeCI --data MDD --data_path dataset/MDD --protocol AAL116 --causal_learning_target temporal_sem --temporal_lag_order 2 --iterations 1 --max_folds 1 --train_epochs 5
python test_mdd_best_config.py --max-folds 1 --train-epochs 5 --visualize-causal 1
```

旧静态路径仅用于对照：

```bash
python run_cv.py --model S-DeCI --causal_learning_target static_feature
```
