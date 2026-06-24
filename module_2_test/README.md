# 模块 2 因果图学习测试

该目录用于独立测试新模块设计中的模块 2：从 Cycle-like 特征 `C` 学习潜在因果邻接矩阵 `A`。

本测试不接入 `S-DeCI`，不修改 `models/`，也不使用正式训练入口。

## 文件

- `causal_graph_learner.py`: `CausalGraphLearner`、NOTEARS DAG penalty、邻接阈值和结构指标。
- `analytic_dag_constraint.py`: 解析 DAG 约束，参考 `docs/新DAG因果方法.md`，使用 `trace((I - W_scaled)^-1) - N`。
- `synthetic_data.py`: 生成带随机加权 ground-truth adjacency 的 Cycle-like 合成因果数据。
- `train_causal_graph.py`: 独立训练脚本，可训练并对比 NOTEARS 与 Analytic DAG 两种方法。

## 运行

```powershell
.\.venv\Scripts\python.exe module_2_test\train_causal_graph.py
```

默认 `--dag-methods both`，会使用同一份合成数据分别训练：

- `notears`: `trace(matrix_exp(A * A)) - N`
- `analytic`: `trace((I - W_scaled)^-1) - N`

也可以只运行单个方法：

```powershell
.\.venv\Scripts\python.exe module_2_test\train_causal_graph.py --dag-methods analytic
```

快速调试可使用较小节点数：

```powershell
.\.venv\Scripts\python.exe module_2_test\train_causal_graph.py --n-nodes 12 --epochs 300 --dag-methods both
```

## 输出

脚本会打印：

- reconstruction loss
- DAG penalty 与归一化 DAG loss
- 归一化 L1 sparsity loss
- edge precision
- edge recall
- edge F1
- random baseline F1
- `A_learned - A_true` 的权重误差指标
- `A_learned_binary - A_structure_true` 的结构差异指标

默认输出位于 `module_2_test/outputs/`：

- `comparison_summary.json`
- `<method>/A_true.npy`
- `<method>/A_structure_true.npy`
- `<method>/A_learned.npy`
- `<method>/A_learned_binary.npy`
- `<method>/A_diff.npy`
- `<method>/A_structure_diff.npy`
- `<method>/causal_matrix_comparison.png`

`A_true` 只用于训练后的指标和可视化，不参与 loss。

默认生成逻辑先在隐含拓扑顺序上构造 DAG，再随机映射到观测节点编号。因此 `A_true` 通常不是上三角矩阵，但仍满足无环约束。脚本会打印 `Topological order in observed node ids`，便于人工检查。
