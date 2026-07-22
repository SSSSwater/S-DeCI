# S-DeCI 模块2关闭时使用相关矩阵说明

本文档说明 `S-DeCI` 在关闭模块 2 因果学习时，如何让模块 3 使用每个样本对应的相关系数矩阵作为 HGCN adjacency。原始 `docs/新模块设计.md` 不修改。

## 使用场景

当设置：

```text
use_causal_module2 = 0
use_hgcn_module3 = 1
```

模型不再学习 `A_learned`，而是从数据集中读取每个样本对应的 correlation matrix，并传给模块 3。此时模块 3 退化为使用样本自身图结构的 HGCN readout。

如果模块 4 HPEC 开启，则仍然使用模块 3 输出的 `z_global` 计算 HPEC energy loss。

## 文件命名

相关矩阵解析会优先按同一 subject 和 protocol 匹配，支持以下命名：

- 当前数据集中常见命名：`sub-xxx_<protocol>_correlation_matrix.mat`
- 后续数据可能使用的命名：`sub-xxx_xxx_features_sub_correlation_matrix.mat`

`.mat` 文件优先读取 `data` key。如果没有 `data`，会尝试常见 key；仍无法识别时会输出该文件中的可用 keys。

## 参数

`run_cv.py` 使用下划线参数：

```bash
python run_cv.py --model S-DeCI --use_causal_module2 0 --use_hgcn_module3 1 --use_sample_correlation_when_module2_disabled 1
```

根目录测试脚本使用横线参数：

```bash
python test_training_smoke.py --use-causal-module2 0 --use-hgcn-module3 1 --use-sample-correlation-when-module2-disabled 1
```

相关矩阵负值处理由 `sample_correlation_mode` 控制：

- `abs`：默认，将正负相关都作为连接强度。
- `positive`：只保留正相关。
- `raw`：保留原始值。

## 行为差异

- 模块 2 开启时：模块 3 使用 Temporal NTS-NOTEARS 学到的分类图 `A_cls`。模块 2 的训练目标为：

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

其中 $\mathcal{L}_{\mathrm{pred}}$ 来源于 Granger causality / Temporal NTS-NOTEARS 的“历史时间窗预测未来时间点”原则，$\mathcal{L}_{\mathrm{sparse}}$ 控制脑区间因果边数量，$\mathcal{L}_{\mathrm{smooth}}$ 约束相邻 lag 的图不要剧烈跳变，$h(A_0)$ 只约束同时间残余图的 DAG 性质。
- 模块 2 关闭时：模块 3 使用样本 correlation matrix，不计算 $\mathcal{L}_{\mathrm{pred}}$、$\mathcal{L}_{\mathrm{sparse}}$、$\mathcal{L}_{\mathrm{smooth}}$ 和 $h(A_0)$。这样做的目的是保留一个“无因果学习”的图分类对照，用来判断模块 2 是否真正带来收益。
- 测试集 label 仍只用于 loss、metric 和可视化标注，不会输入模型。

## 可视化

开启 `visualize_causal=1` 后，中间量 heatmap 会包含 `Sample correlation adjacency`，并继续显示模块 3 的 normalized adjacency、`H_gcn` 和 `z_global` 等内容。

## 回退

恢复原因果学习路径：

```text
use_causal_module2 = 1
```

或显式关闭相关矩阵回退：

```text
use_sample_correlation_when_module2_disabled = 0
```
