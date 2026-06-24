## 1. 目录与文件结构

- [x] 1.1 新增根目录 `module_2_test/`，集中放置本次模块 2 测试相关代码。
- [x] 1.2 在 `module_2_test/` 中新增 `__init__.py`，保证测试模块可被脚本 import。
- [x] 1.3 确认本次不在 `models/` 下新增模块 2 测试文件。
- [x] 1.4 确认本次不修改 `S-DeCI`、`DeCI`、`Exp_Basic.model_dict` 或 `run_cv.py` 的模型注册行为。

## 2. 因果图学习模块

- [x] 2.1 新增 `module_2_test/causal_graph_learner.py`。
- [x] 2.2 实现 `CausalGraphLearner`，接收形状为 `[B, N, F]` 的 Cycle-like 特征 `C`，默认支持 `F=64`。
- [x] 2.3 实现可学习邻接参数，并通过 off-diagonal mask 确保有效邻接矩阵 `A` 的对角线为 `0`。
- [x] 2.4 按 `A[parent, child]` 方向约定实现 `C_hat` 重构，并保证 `C_hat` 与 `C` 形状一致。
- [x] 2.5 实现 `dag_penalty(A) = trace(matrix_exp(A * A)) - N`，并保证可参与 PyTorch autograd。
- [x] 2.6 实现邻接阈值化与结构指标函数，至少包括 edge precision、edge recall、edge F1 和差异矩阵。
- [x] 2.7 新增 `module_2_test/analytic_dag_constraint.py`，实现谱半径缩放后的 Analytic DAG penalty。
- [x] 2.8 扩展 `CausalGraphLearner`，通过 `dag_method="notears"` 或 `dag_method="analytic"` 选择 DAG 约束。

## 3. 合成因果数据

- [x] 3.1 新增 `module_2_test/synthetic_data.py`。
- [x] 3.2 实现默认 ground-truth adjacency `A_true`，包含链式、分叉或多父节点结构中的至少两类，支持 `n_nodes > 8` 时自动扩展随机有向边，并通过隐含拓扑顺序到观测节点编号的随机映射避免矩阵天然上三角。
- [x] 3.3 实现 Cycle-like 特征生成函数，输出 `C: [B, N, 64]` 和 `A_true: [N, N]`。
- [x] 3.4 确认默认 `A_true` 是 DAG，并提供可复现实验的随机种子控制。

## 4. 训练与矩阵一致性检查

- [x] 4.1 新增 `module_2_test/train_causal_graph.py`，可从仓库根目录直接运行。
- [x] 4.2 实现默认 CPU 训练流程，使用合成数据训练 `CausalGraphLearner`。
- [x] 4.3 训练输出 reconstruction loss、DAG penalty、edge precision、edge recall 和 edge F1。
- [x] 4.4 训练完成后输出或保存随机加权 `A_true`、二值结构 `A_structure_true`、连续权重矩阵 `A_learned`、阈值化矩阵 `A_learned_binary`、权重差值矩阵和结构差值矩阵。
- [x] 4.5 调用已有 `utils.tensor_visualization.visualize_tensors` 保存 `A_true`、`A_structure_true`、`A_learned`、`A_learned_binary`、权重差值矩阵和结构差值矩阵 heatmap。
- [x] 4.6 将可视化输出默认保存到 `module_2_test/outputs/`，并打印输出路径。
- [x] 4.7 确认默认训练 loss 不使用 `A_true` 或 `A_structure_true`，二者只用于训练完成后的评估、差值矩阵和可视化。
- [x] 4.8 确认默认训练结果的 edge F1 高于随机猜测基线，并记录无监督恢复的局限。
- [x] 4.9 扩展 `train_causal_graph.py`，支持 `--dag-methods both|notears|analytic`，并在同一份合成数据上对比 NOTEARS 与 Analytic DAG。
- [x] 4.10 将两种方法的输出分别保存到 `output-dir/notears/` 和 `output-dir/analytic/`，并生成 `comparison_summary.json`。

## 5. 文档与验证

- [x] 5.1 新增 `module_2_test/README.md`，说明模块 2 测试目的、运行命令、输出指标和可视化文件。
- [x] 5.2 使用 `.venv` Python 编译检查 `module_2_test/` 下的所有 Python 文件。
- [x] 5.3 运行默认训练脚本，确认训练完成并生成矩阵可视化图片。
- [x] 5.4 记录训练命令、关键指标和可视化输出路径，便于后续 verify/archive。

## 6. 验证记录

- 编译检查：
  - `.\\.venv\\Scripts\\python.exe -m py_compile module_2_test\\__init__.py module_2_test\\analytic_dag_constraint.py module_2_test\\causal_graph_learner.py module_2_test\\synthetic_data.py module_2_test\\train_causal_graph.py`
- 默认长训练检查：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py`
  - 说明：当前默认训练参数为 `n_nodes=116`、`epochs=2000`、`dag-methods=both`，用于较长的全脑节点规模检查。
- 快速训练检查：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --n-nodes 12 --epochs 300 --dag-methods notears --output-dir module_2_test\\outputs\\rollback_module2_smoke --print-every 0`
  - 关键指标：`edge_f1: 0.42857142857090824`
  - 关键指标：`random_baseline_f1: 0.09090909361839294`
- 当前无监督默认训练检查：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --output-dir module_2_test\\outputs\\default_unsupervised_reweighted_v2 --print-every 0`
- 两种 DAG 方法快速对比：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --n-nodes 12 --epochs 300 --dag-methods both --output-dir module_2_test\\outputs\\compare_dag_methods_n12_e300 --print-every 0`
- 随机观测编号 DAG 检查：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --output-dir module_2_test\\outputs\\shuffled_n12_t045 --print-every 0`
- 权重差值检查：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --output-dir module_2_test\\outputs\\weight_diff_check --print-every 0`
- 大于 8 维训练检查：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --n-nodes 16 --epochs 600 --output-dir module_2_test\\outputs\\n16_unsupervised_reweighted --print-every 0`
- 大于 8 维两种 DAG 方法对比：
  - `.\\.venv\\Scripts\\python.exe module_2_test\\train_causal_graph.py --n-nodes 16 --epochs 250 --dag-methods both --output-dir module_2_test\\outputs\\compare_dag_methods_n16_e250 --print-every 0`
- 关键指标：
  - 当前默认 12 节点无监督训练：`reconstruction_loss: 0.1665729284286499`
  - 当前默认 12 节点无监督训练：`dag_penalty: 0.2175455093383789`
  - 当前默认 12 节点无监督训练：`dag_loss_normalized: 0.018128791823983192`
  - 当前默认 12 节点无监督训练：`l1_loss_normalized: 0.07710150629281998`
  - 当前默认 12 节点无监督训练：`lambda_recon: 1.0`
  - 当前默认 12 节点无监督训练：`lambda_dag: 0.001`
  - 当前默认 12 节点无监督训练：`lambda_l1: 0.0001`
  - 当前默认 12 节点无监督训练：`edge_precision: 0.49999999999997224`
  - 当前默认 12 节点无监督训练：`edge_recall: 0.7499999999999375`
  - 当前默认 12 节点无监督训练：`edge_f1: 0.59999999999948`
  - 当前默认 12 节点无监督训练：`random_baseline_f1: 0.09090909361839294`
  - 当前默认 12 节点无监督训练：`shd: 12`
  - 当前默认 12 节点无监督训练：`weight_mae_overall: 0.07765568047761917`
  - 当前默认 12 节点无监督训练：`weight_mae_true_edges: 0.4209306240081787`
  - 当前默认 12 节点无监督训练：`false_edge_mean_abs: 0.05109375715255737`
  - 当前 16 节点无监督训练：`edge_f1: 0.5142857142851852`
  - 当前 16 节点无监督训练：`random_baseline_f1: 0.07083333283662796`
  - 当前 16 节点无监督训练：`weight_mae_true_edges: 0.4328412115573883`
  - 12 节点 300 epoch 对比：NOTEARS `edge_f1: 0.42857142857090824`
  - 12 节点 300 epoch 对比：Analytic DAG `edge_f1: 0.4444444444439177`
  - 16 节点 250 epoch 对比：NOTEARS `edge_f1: 0.39999999999953284`
  - 16 节点 250 epoch 对比：Analytic DAG `edge_f1: 0.11111111111099384`
  - 16 节点 250 epoch 对比：Analytic DAG `analytic_spectral_radius: 0.10218901187181473`
- 矩阵一致性：
  - 当前无监督训练不再出现监督版本的全 0 结构差值；`A_structure_diff.npy` 用于暴露漏边和误检边。
  - `A_diff.npy` 保存 `A_learned - A_true` 的权重差值矩阵。
  - 默认生成逻辑会输出观测节点编号下的 `topological_order`，`A_true` 不再依赖节点编号顺序呈上三角；探针确认 12 节点结构矩阵上下三角均有边。
- 可视化输出：
  - `module_2_test/outputs/causal_matrix_comparison.png`
  - `module_2_test/outputs/A_true.npy`
  - `module_2_test/outputs/A_structure_true.npy`
  - `module_2_test/outputs/A_learned.npy`
  - `module_2_test/outputs/A_learned_binary.npy`
  - `module_2_test/outputs/A_diff.npy`
  - `module_2_test/outputs/A_structure_diff.npy`
  - `module_2_test/outputs/n16/causal_matrix_comparison.png`
  - `module_2_test/outputs/weight_diff_check/causal_matrix_comparison.png`
  - `module_2_test/outputs/shuffled_n12_t045/causal_matrix_comparison.png`
  - `module_2_test/outputs/default_unsupervised_reweighted_v2/causal_matrix_comparison.png`
  - `module_2_test/outputs/n16_unsupervised_reweighted/causal_matrix_comparison.png`
  - `module_2_test/outputs/compare_dag_methods_n12_e300/comparison_summary.json`
  - `module_2_test/outputs/compare_dag_methods_n12_e300/notears/causal_matrix_comparison.png`
  - `module_2_test/outputs/compare_dag_methods_n12_e300/analytic/causal_matrix_comparison.png`
  - `module_2_test/outputs/compare_dag_methods_n16_e250/comparison_summary.json`
