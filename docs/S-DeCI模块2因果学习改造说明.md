# S-DeCI 模块 2 因果学习改造说明

本文档记录当前项目中模块 2 的工程实现，不覆盖 `docs/新模块设计.md` 与参考论文源码说明。

## 目标

模块 2 的目标是从模块 1 输出的节点特征 `C` 中学习一个有向因果图，并把该图传递给后续 HGCN 或 GCN fallback。为了便于对比和回退，当前实现保留旧的 `nts_notears` 和 `dag_sampling`，并新增 `dagma_logdet`。

## 图学习方法

- `nts_notears`：沿用 NTS-NOTEARS 风格的 positive/negative 第一层权重，通过第一层权重的范数得到共享图 `A_shared`，DAG penalty 默认使用稳定的 analytic 约束。
- `dagma_logdet`：复用 NTS-NOTEARS 的可学习边参数，但 DAG penalty 使用 log-det 形式，便于和 DAGMA 思路对齐。该方法通过谱半径保护 `sI - A*A` 的数值稳定性。
- `dag_sampling`：使用 Sinkhorn permutation 和上三角 order mask 构造可微 DAG，保留 hard straight-through permutation 开关。

## 输入归一化

新增 `--causal_input_norm`：

- `none`：默认值，完全保持旧行为。
- `feature_zscore`：对每个样本、每个 ROI 的特征维做 z-score。
- `batch_node_zscore`：按 batch 与特征维对每个 ROI 做 z-score。

重构损失使用归一化后的输入作为目标，避免归一化开启时 `C_hat` 与目标口径不一致。

## 共享图与样本残差图

模块 2 输出现在包含：

- `A_shared`：全训练集共享的因果图。
- `A_delta`：可选样本级残差图，形状为 `[B, N, N]`。
- `A_effective`：下游实际使用的图，等于 `A_shared + A_delta` 后截断为非负，默认关闭残差时等于 `A_shared`。

当前命令行只保留一个常用开关：

- `--use_sample_graph_residual`：是否启用样本级残差图。

残差图幅度、稀疏正则、偏离共享图正则等属于低层实现细节，当前固定为内部默认值，避免日常训练命令过于拥挤。

## Loss 与诊断

模块 2 保留 reconstruction、DAG、L1 和样本残差图正则。细粒度 warmup 与 DAGMA log-det 的内部数值参数当前不再暴露为 `run_cv.py` 的命令行参数，默认使用稳定配置。

训练日志会额外显示：

- 原始与加权的 reconstruction、DAG、L1、sample graph 正则。
- DAG penalty 相关谱半径。
- 图平均权重、方向性比例和样本残差图幅度。

## 推荐对比方式

低成本调参时可先比较：

```bash
python run_cv.py --data MDD --data_path dataset/MDD --protocol AAL116 --model S-DeCI --iterations 1 --max_folds 1 --train_epochs 20 --causal_graph_method nts_notears
python run_cv.py --data MDD --data_path dataset/MDD --protocol AAL116 --model S-DeCI --iterations 1 --max_folds 1 --train_epochs 20 --causal_graph_method dagma_logdet --causal_input_norm feature_zscore
python run_cv.py --data MDD --data_path dataset/MDD --protocol AAL116 --model S-DeCI --iterations 1 --max_folds 1 --train_epochs 20 --causal_graph_method dagma_logdet --causal_input_norm feature_zscore --use_sample_graph_residual 1
```

若要回退到旧行为，使用默认参数即可：`causal_input_norm=none`、`use_sample_graph_residual=0`。
