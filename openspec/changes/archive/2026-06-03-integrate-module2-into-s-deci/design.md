## Context

当前 `S-DeCI` 已在 `models/S_DeCI.py` 中实现模块 1：输入 fMRI 时间序列后先得到 `[B, N, d_model]` 的节点特征，再经过若干 `DeCI_Block` 分解出 trend / seasonal，其中分类结果来自每个 block 的 `cls_seasonal` 累加。`DeCI_Block` 内部已经计算了真实的 seasonal feature，但当前 `forward` 只返回 seasonal 分类 logits，没有把 `[B, N, d_model]` 的 seasonal feature 暴露给上层。

模块 2 已在 `module_2_test/` 中独立验证，核心组件 `CausalGraphLearner` 的输入约定是 `C: [B, N, F]`，输出共享邻接矩阵 `A: [N, N]`、重构特征 `C_hat` 和 DAG penalty。该目录目前更像实验验证空间，不适合让正式模型长期依赖测试目录。

本次变更需要把模块 2 接入 `S-DeCI` 的模块 1 输出，让正式训练路径能够学习因果图；但分类仍只使用 Cycle/seasonal 特征，不引入模块 3、HGCN 或因果图卷积分类结果。可视化需要显式开启，并复用 `utils.tensor_visualization.visualize_tensors`。

## Goals / Non-Goals

**Goals:**

- 在不修改 `models/DeCI.py` 的前提下，为 `S-DeCI` 接入模块 2 因果图学习。
- 以模块 1 的 Cycle/seasonal feature 作为模块 2 输入，保持张量语义为 `[B, N, d_model]`。
- 训练时将模块 2 的 reconstruction、DAG acyclicity、L1 sparsity loss 纳入总 loss，使因果图可被反向传播优化。
- 保持 `S-DeCI` 的最终分类输出只来自 seasonal 分类分支，保证当前阶段不提前引入模块 3 的分类逻辑。
- 提供可选的中间量可视化，包括 Cycle feature、`C_hat`、`A_learned`、阈值化邻接矩阵和重构误差等。
- 在 `S-DeCI` 新增或改动的关键逻辑处添加中文注释，必要的英文关键词如 `Cycle`、`seasonal`、`causal graph`、`DAG` 可保留，方便后续阅读和维护。
- 保持现有模型和训练流程的兼容性：其他模型不需要实现额外接口也能照常训练。

**Non-Goals:**

- 不实现模块 3、HGCN 或基于因果图卷积的分类。
- 不在本次变更中重新设计 `DeCI_Block` 的分解算法。
- 不让训练依赖真实因果矩阵，因为真实数据训练阶段不可获得 ground truth graph。
- 不把可视化作为默认训练行为，避免常规训练变慢或生成大量图片。
- 不直接修改原始 `DeCI` 模型。

## Decisions

### 1. 将模块 2 核心迁移为可复用组件

将 `module_2_test/causal_graph_learner.py` 中的核心因果学习逻辑迁移或复制到正式可复用位置，例如 `layers/causal_graph_learner.py`。`module_2_test` 保留独立实验代码，但正式 `S-DeCI` 不从测试目录 import。

选择原因：`module_2_test` 是验证目录，正式模型依赖它会让工程边界混乱；迁移到 `layers/` 后，模块 2 可以同时服务正式模型和后续模块 3。

备选方案：直接从 `module_2_test` import。该方案改动更少，但会让生产模型依赖测试目录，后续维护成本更高。

### 2. 在 `S-DeCI` 内捕获 seasonal feature，而不是使用 seasonal logits

模块 2 输入应为 `DeCI_Block` 内部的 seasonal feature，形状为 `[B, N, d_model]`。由于当前 `DeCI_Block.forward()` 只返回 `cls_seasonal`，实现时有两种可接受方式：

- 优先方案：在 `layers/DeCI_Layer.py` 中为 `DeCI_Block` 增加可选返回 feature 的能力，例如 `return_features=False`，默认行为保持不变；`S-DeCI` 调用时开启并拿到 `seasonal_feature`。
- 备选方案：在 `S_DeCI.py` 中按 `DeCI_Block` 的内部顺序显式调用 `trend_ext` 和 `seasonal_ext`，自行得到 `seasonal_feature` 和 logits。

选择优先方案的原因：默认接口不变，其他使用 `DeCI_Block` 的代码不受影响；同时避免在 `S_DeCI.py` 重复 block 内部逻辑。

### 3. 使用聚合后的 Cycle feature 学习共享因果图

若 `S-DeCI` 有多层 `DeCI_Block`，模块 2 默认使用所有 block 的 seasonal feature 求和或取最后一层作为 `C`，并通过配置控制。默认建议使用与分类 logits 一致的聚合方式，即对各层 seasonal feature 求和，得到 `cycle_features: [B, N, d_model]`。

选择原因：当前分类逻辑是对多层 seasonal logits 求和，使用聚合后的 seasonal feature 能更贴近当前分类分支使用的信息。

备选方案：只用最后一层 seasonal feature。该方案更简单，但和现有 logits 聚合语义不完全一致。

### 4. 分类输出保持原接口，辅助损失通过模型属性暴露

`S-DeCI.forward(x_enc)` 继续返回 `y_hat`，保持训练、验证和测试代码对输出的兼容。模块 2 产生的中间结果与 loss 通过模型属性或方法暴露，例如：

- `model.latest_causal_output`
- `model.latest_aux_losses`
- `model.get_aux_loss()`

训练循环在得到 `y_hat` 后，如果模型存在可用的辅助 loss，并且配置开启模块 2，就把辅助 loss 加到分类 loss 上。

选择原因：现有 `exp/exp_classification_CV.py` 默认假设模型只返回 logits/probability。保持 `forward` 返回不变，可以避免影响其他模型。

备选方案：让 `forward` 返回 `(y_hat, aux)`。该方案语义清晰，但会要求训练、验证和所有下游调用都适配 tuple，风险更大。

### 5. 辅助损失使用无监督因果学习项

模块 2 总损失由以下项组成：

- `causal_recon_loss`: `MSE(C_hat, C)`，约束因果图能重构 Cycle feature。
- `causal_dag_loss`: 归一化 DAG penalty，约束邻接矩阵接近无环。
- `causal_l1_loss`: 归一化 L1 sparsity，鼓励图稀疏。

不使用 `structure_loss` 或任何真实因果矩阵监督项。权重通过配置控制，例如 `lambda_causal_recon`、`lambda_causal_dag`、`lambda_causal_l1`，并提供默认值以便 S-DeCI 训练脚本可直接运行。

### 6. 可视化显式触发并落盘

`S-DeCI` 提供显式可视化方法或在 forward 后按配置触发一次性可视化，例如：

- `model.visualize_causal_intermediates(save_path=...)`
- 或配置 `visualize_causal=1`、`causal_vis_dir=...`、`visualize_every=...`

可视化内容复用 `visualize_tensors`，至少包括：

- `cycle_features`
- `causal_reconstruction`
- `causal_reconstruction_error`
- `causal_adjacency`
- `causal_adjacency_binary`

3D 张量只显示 Batch0，由已有工具在副标题中展示 shape 和 Batch 提示。默认不开启可视化。

### 7. 训练流程只对支持辅助损失的模型叠加 loss

在 `exp/exp_classification_CV.py` 中新增通用辅助 loss 读取逻辑。伪流程如下：

```python
y_hat = self.model(x_enc)
loss = criterion(y_hat, label)
aux_loss = getattr(self.model, "get_aux_loss", None)
if callable(aux_loss):
    loss = loss + aux_loss()
```

如果使用 `DataParallel`，需要从 `self.model.module` 读取同名方法或属性。验证阶段默认只统计分类 loss 和分类指标，避免辅助 loss 改变现有早停逻辑；如需诊断，可额外打印 causal loss，但不作为主指标。

### 8. `S-DeCI` 代码注释使用中文

`models/S_DeCI.py` 中本次新增的模块 2 接入、Cycle feature 聚合、辅助 loss 缓存和可视化调用等关键逻辑需要添加简洁中文注释。注释重点解释“为什么这样接”和“当前阶段不做什么”，避免逐行翻译代码。已有英文术语如 `Cycle`、`seasonal`、`causal graph`、`DAG`、`adjacency` 可以保留。

## Risks / Trade-offs

- [Risk] `DeCI_Block` 接口变化可能影响 `DeCI` 或其他模型 → 通过可选参数保持默认返回值不变，并只在 `S-DeCI` 中启用 feature 返回。
- [Risk] 因果重构 loss 过大可能压制分类目标 → 为辅助 loss 设置较小默认权重，并允许通过命令行调整或关闭模块 2。
- [Risk] DAG penalty 数值在 116 节点上可能较大或不稳定 → 使用归一化 DAG loss，并保留 analytic / NOTEARS 方法配置，默认采用已有验证更稳定的方法。
- [Risk] 可视化在训练中频繁调用会明显拖慢速度 → 默认关闭，并通过 `visualize_every` 或手动方法限制保存频率。
- [Risk] 多层 seasonal feature 聚合方式会影响因果图语义 → 默认与分类分支一致使用求和聚合，同时保留 `causal_feature_source` 配置支持切换到 last。
- [Risk] `DataParallel` 下辅助 loss 读取位置不同 → 训练循环读取时兼容 `model.module`。

## Migration Plan

1. 新建正式可复用的模块 2 文件，例如 `layers/causal_graph_learner.py`，迁移 `CausalGraphLearner`、DAG penalty 和阈值化工具。
2. 扩展 `DeCI_Block` 的可选 feature 返回能力，默认行为保持完全兼容。
3. 修改 `models/S_DeCI.py`，初始化模块 2，forward 中生成 `cycle_features`、运行因果学习，并缓存中间量与辅助 loss。
4. 在 `models/S_DeCI.py` 的新增关键逻辑处补充中文注释，说明模块 2 输入、loss 缓存和可视化触发方式。
5. 修改 `exp/exp_classification_CV.py`，在训练阶段通用读取并叠加模型辅助 loss。
6. 在 `run_cv.py` 或测试脚本中补充模块 2 相关参数默认值，保证 IDE 直接运行和命令行验证都可用。
7. 增加或更新测试：至少覆盖 S-DeCI forward shape、辅助 loss 可反传、可视化文件生成、短训练能跑通。
8. 若需要回滚，关闭配置中的模块 2 或移除 `S_DeCI.py` 的因果模块初始化与训练循环中的辅助 loss 读取即可，原 `DeCI` 不受影响。
