## Purpose

定义模块 2 因果图学习能力：包括独立测试目录中的合成 DAG 训练检查，以及正式可复用的 PyTorch 因果图学习组件。模块 2 用于从 Cycle/seasonal feature `C` 中学习脑区间共享因果邻接矩阵 `A`，训练 loss 不得依赖真实因果矩阵监督。

## Requirements

### Requirement: 独立模块 2 测试目录

系统 SHALL 将模块 2 因果学习独立验证相关代码集中放置在根目录 `module_2_test/` 下，用于保留合成数据、独立训练和可视化检查能力。

#### Scenario: 测试目录隔离

- **WHEN** 开发者查看仓库根目录
- **THEN** MUST 能找到 `module_2_test/`
- **AND** `module_2_test/` MUST 包含模块实现、合成数据生成和训练检查脚本
- **AND** `models/` MUST 不包含模块 2 独立测试文件

#### Scenario: 保留独立测试用途

- **WHEN** 正式模型接入模块 2 后
- **THEN** `module_2_test/` MUST 继续保留独立测试、合成数据和训练检查用途
- **AND** 正式模型 MUST NOT 从 `module_2_test/` 直接导入生产训练所需组件

### Requirement: 模块 2 提供正式可复用因果学习组件

模块 2 SHALL 提供可被正式模型 import 的 PyTorch 因果图学习组件，用于从 Cycle/seasonal feature 中学习共享脑区因果邻接矩阵。

#### Scenario: 正式模型可复用模块 2

- **WHEN** `S-DeCI` 需要接入模块 2
- **THEN** 系统 MUST 提供不依赖 `module_2_test/` 测试目录的模块 2 核心组件
- **AND** 该组件 MUST 能被 `models/S_DeCI.py` 或其依赖层正常 import

#### Scenario: 接收 Cycle 特征输入

- **WHEN** 模块 2 接收形状为 `[B, N, F]` 的输入特征 `C`
- **THEN** 默认特征维度 `F` MUST 支持 `64`
- **AND** 模块 MUST 保持 batch 维度和节点维度语义不变

#### Scenario: 输出共享因果矩阵

- **WHEN** 模块 2 完成 forward
- **THEN** MUST 输出形状为 `[N, N]` 的连续邻接矩阵 `A_learned`
- **AND** `A_learned` 的对角线 MUST 为 `0`
- **AND** 邻接方向 MUST 按 `A[parent, child]` 解释

#### Scenario: 输出重构特征

- **WHEN** 模块 2 执行 forward
- **THEN** MUST 输出重构特征 `C_hat`
- **AND** `C_hat` 的形状 MUST 与输入 `C` 相同

### Requirement: DAG 无环约束

系统 SHALL 提供可微的 DAG acyclicity penalty，用于约束学习到的邻接矩阵接近无环结构，并支持 NOTEARS 与 Analytic DAG 两种约束方法。

#### Scenario: 计算 NOTEARS DAG penalty

- **WHEN** 给定有效邻接矩阵 `A`
- **THEN** 模块 MUST 能计算 `trace(matrix_exp(A * A)) - N` 形式的 DAG penalty
- **AND** penalty MUST 能参与 PyTorch autograd 反向传播

#### Scenario: 计算 Analytic DAG penalty

- **WHEN** 给定有效邻接矩阵 `A`
- **THEN** 模块 MUST 能计算谱半径缩放后的 `trace((I - W_scaled)^-1) - N` 形式的 Analytic DAG penalty
- **AND** `W_scaled` MUST 由 `W = A * A`、power iteration 谱半径估计和安全 margin 构造
- **AND** penalty MUST 能参与 PyTorch autograd 反向传播

#### Scenario: DAG penalty 接近零

- **WHEN** 输入矩阵是严格无环的邻接矩阵
- **THEN** DAG penalty SHOULD 接近 `0`

### Requirement: 模块 2 无监督因果学习 loss

模块 2 SHALL 为正式训练提供无监督因果学习 loss，不得依赖真实因果矩阵作为训练监督。

#### Scenario: 计算训练用 loss

- **WHEN** 模块 2 接收 Cycle/seasonal feature 并输出重构结果
- **THEN** MUST 能计算 reconstruction loss
- **AND** MUST 能计算归一化 DAG acyclicity loss
- **AND** MUST 能计算归一化 L1 sparsity loss

#### Scenario: 不泄漏真实因果图

- **WHEN** `S-DeCI` 正式训练模块 2
- **THEN** 模块 2 loss MUST NOT 使用 `A_true`、`A_structure_true` 或任何真实因果矩阵
- **AND** 真实因果矩阵若存在 MUST 仅用于独立实验后的指标或可视化诊断

### Requirement: 模块 2 因果图供模块 3 HGCN 联合训练使用

模块 2 SHALL 将学习到的连续邻接矩阵 `A_learned` 作为模块 3 HGCN 的图拓扑输入，并允许分类 loss 经模块 3 回传到因果图学习参数。

#### Scenario: 模块 3 读取 A_learned

- **GIVEN** `S-DeCI` 同时启用模块 2 和模块 3
- **WHEN** 模块 2 完成一次 forward
- **THEN** 模块 3 MUST 使用模块 2 输出的连续邻接矩阵 `A_learned`
- **AND** `A_learned` 的方向语义 MUST 继续按 `A[parent, child]` 解释

#### Scenario: A_learned 保持可微

- **GIVEN** 模块 3 使用 `A_learned` 执行 HGCN 图传播
- **WHEN** 对 `Loss_cls(logmap0(z_global), label)` 执行反向传播
- **THEN** `A_learned` MUST 保持在 autograd graph 中
- **AND** 模块 2 的因果图学习参数 MUST 能收到来自分类 loss 的梯度

#### Scenario: 联合训练不使用真实因果监督

- **GIVEN** 模块 2 和模块 3 联合训练
- **WHEN** 系统计算 `Loss_total`
- **THEN** loss MUST NOT 使用 `A_true`
- **AND** loss MUST NOT 使用 `A_structure_true`
- **AND** 真实因果矩阵若存在 MUST 仅用于独立实验指标或可视化诊断

### Requirement: 模块 2 支持可配置训练权重

模块 2 SHALL 允许训练流程配置 reconstruction、DAG acyclicity 和 L1 sparsity 的 loss 权重。

#### Scenario: 应用 loss 权重

- **WHEN** 训练流程读取模块 2 辅助 loss
- **THEN** 系统 MUST 能分别应用 reconstruction、DAG acyclicity 和 L1 sparsity 的权重
- **AND** 这些权重 MUST 可通过配置参数调整

#### Scenario: 支持关闭模块 2

- **WHEN** 用户通过配置关闭模块 2 或将其总权重设为 `0`
- **THEN** `S-DeCI` MUST 能退化为仅使用模块 1 Cycle/seasonal 分类的训练路径
- **AND** 训练流程 MUST 继续跑通

### Requirement: 合成 Cycle-like 因果数据

系统 SHALL 提供合成数据生成逻辑，用于构造带有已知因果关系的 Cycle-like 特征输入。

#### Scenario: 生成带 ground truth 的数据

- **WHEN** 运行合成数据生成函数
- **THEN** MUST 返回 `C`、随机加权 ground-truth adjacency `A_true` 和二值结构矩阵 `A_structure_true`
- **AND** `C` 的形状 MUST 为 `[B, N, F]`
- **AND** `A_true` 的形状 MUST 为 `[N, N]`
- **AND** `A_structure_true` 的形状 MUST 为 `[N, N]`

#### Scenario: 数据包含有向因果结构

- **WHEN** 生成默认合成数据
- **THEN** `A_structure_true` MUST 包含链式、分叉或多父节点结构中的至少两类
- **AND** `A_structure_true` MUST 是 DAG

#### Scenario: 支持超过 8 个节点

- **WHEN** 用户请求 `n_nodes > 8` 的合成数据
- **THEN** 数据生成函数 MUST 生成对应节点数的 `C`、`A_true` 和 `A_structure_true`
- **AND** 额外节点 MUST 能按拓扑顺序参与随机有向边生成

#### Scenario: 避免观测矩阵天然上三角

- **WHEN** 生成默认合成数据
- **THEN** 数据生成函数 MUST 先在隐含拓扑顺序上构造 DAG
- **AND** MUST 将隐含拓扑节点随机映射到观测节点编号
- **AND** 默认显示的 `A_true` MUST 不依赖节点编号顺序天然呈上三角

#### Scenario: 随机边权重

- **WHEN** 合成数据生成 `A_true`
- **THEN** 每条存在的边 MUST 使用由 seed 控制的随机权重
- **AND** 默认实现 MUST 不使用固定硬编码边权重

### Requirement: 独立训练检查

系统 SHALL 提供独立训练脚本，用于训练模块 2 并验证其是否能恢复合成数据中的已知因果结构。

#### Scenario: 训练脚本可直接运行

- **WHEN** 用户在仓库根目录运行模块 2 训练脚本
- **THEN** 脚本 MUST 使用 `.venv` 中已有依赖完成训练检查
- **AND** 默认训练配置 SHOULD 使用 `n_nodes=116` 和较长训练预算，用于进行全脑节点规模检查
- **AND** MUST 不要求用户先运行 `S-DeCI` 或主训练脚本

#### Scenario: 同时对比多种因果图方法

- **WHEN** 用户使用对比参数运行训练脚本
- **THEN** 脚本 MUST 在同一份合成数据上分别训练支持的因果图学习方法
- **AND** MUST 分别保存不同方法的矩阵、差值矩阵和 heatmap
- **AND** MUST 输出 `comparison_summary.json`

#### Scenario: 输出训练指标

- **WHEN** 训练脚本完成
- **THEN** MUST 输出 reconstruction loss
- **AND** MUST 输出 DAG penalty
- **AND** MUST 输出归一化后的 DAG loss 和 L1 sparsity loss
- **AND** MUST 输出邻接恢复指标，至少包括 edge precision、edge recall 和 edge F1

#### Scenario: 训练 loss 不泄漏真实因果矩阵

- **WHEN** 训练脚本优化模块 2
- **THEN** loss MUST NOT 直接使用 `A_true` 或 `A_structure_true`
- **AND** `A_true` 与 `A_structure_true` MUST 仅用于训练完成后的指标、差值矩阵和可视化

#### Scenario: 对比因果矩阵一致性

- **WHEN** 训练脚本完成
- **THEN** MUST 输出或保存生成训练样本时使用的随机加权 ground-truth adjacency `A_true`
- **AND** MUST 输出或保存对应二值结构矩阵 `A_structure_true`
- **AND** MUST 输出或保存学习得到的连续邻接矩阵 `A_learned`
- **AND** MUST 输出或保存阈值化后的邻接矩阵 `A_learned_binary`
- **AND** MUST 提供 `A_learned - A_true` 的权重差值对比
- **AND** MUST 提供 `A_learned_binary - A_structure_true` 的结构差值对比

#### Scenario: 可视化因果矩阵

- **WHEN** 训练脚本完成
- **THEN** MUST 调用已有 `utils.tensor_visualization.visualize_tensors`
- **AND** MUST 将 `A_true`、`A_structure_true`、`A_learned`、`A_learned_binary`、权重差值矩阵和结构差值矩阵保存为 heatmap 图片
- **AND** 可视化输出路径 MUST 位于用户通过参数指定的位置

#### Scenario: 验证因果学习有效性

- **WHEN** 使用默认 `116` 节点合成数据和默认较长训练配置运行训练脚本
- **THEN** 训练后的 edge F1 SHOULD 高于随机猜测基线
- **AND** 学到的邻接矩阵 SHOULD 能恢复 `A_true` 中的主要有向边

### Requirement: 模块 2 默认使用 Temporal NTS-NOTEARS

S-DeCI 模块 2 SHALL 在正式默认训练路径中使用时间序列预测式 NTS-NOTEARS，而不是静态节点特征重构式 DAG 学习。

#### Scenario: 使用时间序列预测学习因果图

- **GIVEN** `use_causal_module2 == 1`
- **AND** `causal_learning_target == "temporal_sem"`
- **WHEN** `S-DeCI` 执行 forward
- **THEN** 模块 2 MUST 使用形状为 `[B, T, N]` 或 `[B, T, N, D]` 的时间序列输入
- **AND** MUST 使用历史窗口 `x_{t-1}, ..., x_{t-L}` 预测 `x_t`
- **AND** MUST 输出 `A_lag: [lag_order, N, N]` 作为跨时间主因果图
- **AND** MUST 输出 `A0: [N, N]` 作为同时间片残余依赖图

#### Scenario: 保留 NTS-NOTEARS 权重结构

- **GIVEN** 模块 2 使用 temporal SEM 路径
- **WHEN** 模块 2 构造 `A_lag`
- **THEN** `A_lag` MUST 来自 NTS-NOTEARS 风格的正负第一层权重分解
- **AND** `A_lag[parent, child]` MUST 表示过去 parent 脑区对未来 child 脑区的影响
- **AND** `A_lag` MUST 使用 L1 sparsity 和 lag smoothness 约束
- **AND** `A_lag` MUST NOT 强制 DAG acyclicity，因为跨时间方向已经由过去指向未来

#### Scenario: A0 使用弱无环约束

- **GIVEN** 模块 2 输出 `A0`
- **WHEN** 系统计算模块 2 auxiliary loss
- **THEN** DAGMA / NOTEARS 风格 DAG loss MUST 只作用于 `A0`
- **AND** 该 loss MUST NOT 直接作用于 `A_lag`

#### Scenario: 下游优先使用 A_lag_mean

- **GIVEN** 模块 2 已输出 `A_lag`
- **WHEN** 模块 3 或 GCN fallback 需要分类邻接矩阵
- **THEN** 系统 SHOULD 默认使用 `A_lag.mean(dim=0)` 作为 learned causal graph
- **AND** 当 `classification_graph_source == "blend"` 时，系统 MUST 将该 learned graph 与样本相关矩阵按 `module2_sample_correlation_blend` 融合
- **AND** 可视化 SHOULD 输出 `A_lag`、`A_lag_mean`、`A0`、`A_effective` 和最终 `A_cls`

#### Scenario: 旧静态路径仅作为 legacy 对照

- **GIVEN** 用户显式设置 `causal_learning_target == "static_feature"`
- **THEN** 系统 MAY 使用旧静态 `CausalGraphLearner`
- **AND** 该路径 MUST 被视作 legacy/debug 对照，而不是正式默认模块 2 路径
