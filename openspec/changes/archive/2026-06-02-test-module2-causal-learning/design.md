## Context

模块 2 的目标是从 Cycle 特征 `C` 中学习脑区间潜在因果邻接矩阵 `A`，后续会服务于图卷积和联合损失。但当前阶段必须独立验证模块 2：不接入 `S-DeCI`，不复用主训练入口，不把代码放进 `models/`。

参考资料包括 `docs/新模块设计.md` 中模块 2 的描述，以及 `reference/NTS-NOTEARS-main` 与 `reference/Differentiable-DAG-Sampling-master`。其中 NOTEARS 路线使用 `trace(expm(A)) - d` 作为 DAG acyclicity penalty；Differentiable-DAG-Sampling 路线通过可微排序和上三角 mask 构造 DAG。为了先验证模块 2 是否能学到已知因果结构，本次优先采用更直接、可读性更高的 NOTEARS penalty。

## Goals / Non-Goals

**Goals:**
- 在根目录新增 `module_2_test/`，所有本次模块 2 测试代码都放在该目录下。
- 实现独立的 `CausalGraphLearner`，输入 Cycle-like 特征 `C`，默认形状 `[B, N, 64]`。
- 维护可学习邻接参数，并通过对角 mask 确保有效邻接矩阵对角线为 `0`。
- 输出 `C_hat`、有效邻接矩阵 `A` 和 DAG penalty。
- 生成有明确因果方向和 ground-truth adjacency 的合成 Cycle 特征数据。
- 编写独立训练脚本，训练模块恢复 ground-truth adjacency，并输出可量化指标。
- 使用已有 `utils.tensor_visualization.visualize_tensors` 可视化 `A_true`、`A_learned` 和差异矩阵，保存到 `module_2_test/outputs/`。

**Non-Goals:**
- 不接入 `S-DeCI`，不要求模块 1 输出真实 `C`。
- 不修改 `models/`、`exp/`、`run_cv.py` 或已有训练测试脚本。
- 不实现模块 3、模块 4、Hyperbolic GCN、HPEC 或联合分类损失。
- 不引入新的外部依赖；优先使用 `torch` 和标准库。
- 不直接搬运 reference 中复杂的可微排序采样框架。

## Decisions

1. 目录结构统一放在 `module_2_test/`。

   原因：用户明确要求本次测试相关代码独立放置，避免污染正式模型目录。建议结构为：
   - `module_2_test/causal_graph_learner.py`
   - `module_2_test/synthetic_data.py`
   - `module_2_test/train_causal_graph.py`
   - `module_2_test/README.md`

   备选方案是在 `models/` 下新增模块并配套测试脚本。该方案更接近最终系统集成，但不符合当前“独立测试模块 2”的目标。

2. 邻接矩阵方向定义为 `A[parent, child]`。

   原因：这符合常见 DAG adjacency 表达，指标评估时更直观。为了重构每个节点特征，前向传播使用 `C_hat = einsum("ij,bif->bjf", A, C)`，即每个 child 聚合所有 parent 的 Cycle 特征。

   备选方案使用 `C_hat = torch.matmul(A, C)` 或 `torch.matmul(C, A)`。这两种写法容易因 `[B, N, F]` 维度顺序导致方向含义不清；使用 `einsum` 可以显式表达边方向。

3. 有效邻接矩阵使用有界权重。

   设计为 `A = sigmoid(A_logits) * off_diag_mask`，保证权重非负且便于与 binary ground truth 比较。训练时可额外使用 L1 penalty 控制稀疏性。

   备选方案直接学习无界 `nn.Parameter(N, N)`。该方案表达能力更强，但阈值评估和稳定训练更困难，且容易通过负权重抵消产生不易解释的结构。

4. DAG penalty 同时支持 NOTEARS trace-expm 与 Analytic DAG Constraint。

   实现为 `h(A) = trace(matrix_exp(A * A)) - N`。`reference/NTS-NOTEARS-main/notears/nts_notears.py` 中使用同类约束；本次采用 `torch.matrix_exp`，避免 SciPy 自定义 autograd，保持 `.venv` 内可直接运行。

   新增 Analytic DAG Constraint，参考 `docs/新DAG因果方法.md`：先计算 `W = A * A`，用截断 power iteration 估计谱半径 `rho(W)`，按 `(1 + margin) * rho(W)` 动态缩放得到 `W_scaled`，再计算 `trace((I - W_scaled)^-1) - N`。该方法通过有限收敛半径的解析函数增强对长程环路的梯度敏感性。训练脚本通过 `--dag-methods both|notears|analytic` 在同一份合成数据上对比两种 DAG 约束。

   备选方案使用 Differentiable-DAG-Sampling 的可微 topological order。该方案更严格地构造 DAG，但实现和调参成本更高，不适合作为首个模块 2 有效性检查。

5. 合成数据用随机加权结构方程生成 Cycle-like 特征。

   默认先在隐含拓扑顺序上保留基础链式和分叉结构，例如 `0 -> 1 -> 2`、`0 -> 3`、`3 -> 4` 等，并在 `n_nodes > 8` 时按拓扑顺序随机扩展额外有向边。随后将隐含拓扑节点随机映射到观测节点编号，使显示出来的 `A_true` 不再因编号顺序天然呈上三角，但仍保持 DAG。每条边的权重在指定范围内随机采样并由 seed 复现。每个节点的 `64` 维特征由外生噪声和父节点特征线性组合得到，输出 `C: [B, N, 64]`、加权 ground-truth adjacency `A_true: [N, N]` 与二值结构矩阵 `A_structure_true: [N, N]`。

   备选方案直接随机生成相关矩阵。该方案只能验证相关性恢复，不能清晰证明因果方向是否学到。

6. 训练目标组合为重构、DAG 和稀疏性。

   推荐损失：
   `loss = lambda_recon * recon_mse + lambda_dag * dag_loss_normalized + lambda_l1 * l1_loss_normalized`。

   `recon_mse` 约束 `C_hat` 与输入 `C` 的重构一致性；`dag_loss_normalized` 使用 `trace(matrix_exp(A * A)) - N` 并按节点数归一化，避免节点数变化时权重尺度过大；`l1_loss_normalized` 对 off-diagonal adjacency 取平均绝对值，控制稀疏性且避免正则强度随 `N^2` 放大。训练 loss 不使用 `A_true` 或 `A_structure_true`，二者只用于训练后的指标、差值矩阵和 heatmap 可视化。

7. 训练检查必须包含矩阵一致性对比和可视化。

   除 edge precision、recall、F1 外，脚本需要保存或打印 `A_true`、`A_structure_true`、阈值化后的 `A_learned_binary`、连续权重矩阵 `A_learned`，并计算 `A_learned - A_true` 的权重差值和 `A_learned_binary - A_structure_true` 的结构差值。可视化调用已有 `utils.tensor_visualization.visualize_tensors(A_true, A_structure_true, A_learned, A_learned_binary, weight_diff, structure_diff, titles=[...], save_path=...)`，不新增重复的 heatmap 工具。

   备选方案只输出指标。该方案不利于发现方向反转、局部错误边或阈值设置问题，因此不满足本次“看因果矩阵与生成样本时因果的一致性”的验证目标。

## Risks / Trade-offs

- [Risk] 仅靠观测重构、DAG 和稀疏性不能严格保证因果方向可识别，可能学到相关结构或 Markov equivalent 结构 -> 本次将 `A_true` 严格限制为评估和可视化用途，并通过 edge F1、SHD、权重差值矩阵和 heatmap 暴露恢复质量；后续如需更强因果识别，需要引入时间、干预、非高斯假设或额外先验。
- [Risk] `trace(matrix_exp(A * A))` 在节点数大时计算较慢 -> 测试默认使用较小 `N`，例如 `8` 或 `12`；接口仍支持未来扩展到 `116`。
- [Risk] 合成数据过于简单会高估模块能力 -> 数据生成加入分叉、链式、多父节点、随机扩展边、随机边权重和噪声，并报告 edge precision、recall、F1、SHD 等结构指标。
- [Risk] 直接用节点编号作为拓扑顺序会让因果矩阵天然上三角 -> 默认对隐含拓扑顺序做观测节点随机映射，保留无环性但移除编号顺序偏置。
- [Risk] sigmoid 邻接限制为非负权重，无法表达负向因果影响 -> 当前目标是恢复边结构，不是估计有符号 SEM；如需要符号权重，可后续扩展 signed edge weight。
- [Risk] 数值指标可能掩盖方向反转或局部结构错误 -> 保存 `A_true`、`A_learned` 和差异 heatmap，作为训练结果的必要人工复核依据。

## Migration Plan

1. 创建 `module_2_test/` 目录和模块文件。
2. 实现 `CausalGraphLearner`、DAG penalty、邻接阈值和指标函数。
3. 实现合成 Cycle-like 因果数据生成。
4. 实现独立训练脚本，默认可在 CPU 上快速运行。
5. 在训练脚本中调用已有 tensor 可视化 helper，输出因果矩阵对比图。
6. 运行编译检查和训练检查，记录指标与可视化输出路径。

回滚方式：删除 `module_2_test/` 目录和对应 OpenSpec 变更内容，不影响现有模型与训练系统。

## Open Questions

- 最终接入真实 `S-DeCI` 输出时，模块 2 是否仍使用 `A[parent, child]` 方向约定，需要在模块 3 设计时保持一致。
- 后续是否要支持 signed adjacency 或多被试个体化 adjacency，本次不处理。
