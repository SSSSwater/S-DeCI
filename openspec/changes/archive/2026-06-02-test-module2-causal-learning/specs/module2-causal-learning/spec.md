## ADDED Requirements

### Requirement: 独立模块 2 测试目录

系统 SHALL 将本次模块 2 因果学习测试相关代码集中放置在根目录 `module_2_test/` 下，并且不得将该测试模块放入 `models/` 或接入现有主训练入口。

#### Scenario: 测试目录隔离
- **WHEN** 开发者查看仓库根目录
- **THEN** MUST 能找到 `module_2_test/`
- **AND** `module_2_test/` MUST 包含模块实现、合成数据生成和训练检查脚本
- **AND** `models/` MUST 不因本变更新增模块 2 测试文件

#### Scenario: 不接入现有训练系统
- **WHEN** 本变更实现完成
- **THEN** `S-DeCI`、`DeCI`、`Exp_Basic.model_dict` 和 `run_cv.py` 的模型注册行为 MUST 不因模块 2 测试而改变

### Requirement: 可微因果图学习模块

系统 SHALL 提供一个独立的 PyTorch 模块，用于从 Cycle-like 特征 `C` 中学习脑区间因果邻接矩阵 `A`。

#### Scenario: 接收 Cycle 特征输入
- **WHEN** 模块接收形状为 `[B, N, F]` 的输入特征 `C`
- **THEN** 默认特征维度 `F` MUST 支持 `64`
- **AND** 模块 MUST 保持 batch 维度和节点维度语义不变

#### Scenario: 输出有效邻接矩阵
- **WHEN** 模块执行 forward
- **THEN** MUST 输出有效邻接矩阵 `A`
- **AND** `A` 的形状 MUST 为 `[N, N]`
- **AND** `A` 的对角线 MUST 为 `0`

#### Scenario: 输出重构特征
- **WHEN** 模块执行 forward
- **THEN** MUST 输出重构特征 `C_hat`
- **AND** `C_hat` 的形状 MUST 与输入 `C` 相同
- **AND** 邻接方向 MUST 按 `A[parent, child]` 解释

### Requirement: DAG 无环惩罚

系统 SHALL 提供可微的 DAG acyclicity penalty，用于约束学习到的邻接矩阵接近无环结构，并支持 NOTEARS 与 Analytic DAG 两种约束方法。

#### Scenario: 计算 DAG penalty
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

### Requirement: 合成 Cycle-like 因果数据

系统 SHALL 提供合成数据生成逻辑，用于构造带有已知因果关系的 Cycle-like 特征输入。

#### Scenario: 生成带 ground truth 的数据
- **WHEN** 运行合成数据生成函数
- **THEN** MUST 返回 `C`、随机加权 ground-truth adjacency `A_true` 和二值结构矩阵 `A_structure_true`
- **AND** `C` 的形状 MUST 为 `[B, N, 64]`
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
- **THEN** 脚本 MUST 使用 `.venv` 中已有依赖在 CPU 上完成训练检查
- **AND** 默认训练配置 MUST 使用 `n_nodes=116` 和 `epochs=2000`，用于进行较长的全脑节点规模检查
- **AND** MUST 不要求用户先运行 `S-DeCI` 或主训练脚本

#### Scenario: 同时对比两种 DAG 方法
- **WHEN** 用户使用 `--dag-methods both` 运行训练脚本
- **THEN** 脚本 MUST 在同一份合成数据上分别训练 NOTEARS 与 Analytic DAG 方法
- **AND** MUST 分别保存两种方法的矩阵、差值矩阵和 heatmap
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
- **AND** 可视化输出路径 MUST 位于 `module_2_test/outputs/` 或用户通过参数指定的位置

#### Scenario: 验证因果学习有效性
- **WHEN** 使用默认 `116` 节点合成数据和默认长训练配置运行训练脚本
- **THEN** 训练后的 edge F1 MUST 高于随机猜测基线
- **AND** 学到的邻接矩阵 SHOULD 能恢复 `A_true` 中的主要有向边
