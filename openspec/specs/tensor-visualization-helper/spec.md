## Purpose

定义项目中可复用的 tensor/matrix 可视化 helper。该 helper 使用 `matplotlib` 将 1D、2D、3D tensor 以 heatmap 展示，支持多个输入自动排版，并可被 DeCI、S-DeCI、模块 2 调试和其他实验代码显式调用。

## Requirements

### Requirement: 可复用 tensor 可视化 helper

系统 SHALL 提供一个可被多个模块 import 的独立 Python helper 文件，用于使用 `matplotlib` 可视化模型或数据处理流程中的中间 tensor/matrix。

#### Scenario: 从多个模块调用 helper

- **WHEN** `models/`、`layers/`、`data_provider/` 或实验调试代码需要可视化中间量
- **THEN** 调用方 MUST 能从 `utils.tensor_visualization` import 主可视化函数
- **AND** 调用 helper MUST 不改变训练、验证或推理的默认行为

### Requirement: 输入类型兼容

helper SHALL 支持 `torch.Tensor`、`numpy.ndarray` 和可被 `numpy.asarray` 转换的数组类输入。

#### Scenario: 输入 torch Tensor

- **WHEN** 调用方传入 `torch.Tensor`
- **THEN** helper MUST 使用 `detach().cpu().numpy()` 将其转换为 numpy 数组
- **AND** MUST 不保留 autograd graph

#### Scenario: 输入 numpy 或数组类对象

- **WHEN** 调用方传入 `numpy.ndarray` 或可转换数组类对象
- **THEN** helper MUST 将输入转换为 numpy 数组并继续执行可视化流程

### Requirement: 维度可视化规则

helper SHALL 对 1D、2D、3D 输入使用一致的 heatmap 规则，并清晰拒绝不支持的维度。

#### Scenario: 可视化 1D tensor

- **WHEN** 输入数组维度为 1D
- **THEN** helper MUST 将其转换为单行二维数组
- **AND** MUST 以 heatmap 形式展示
- **AND** subplot MUST 显示原始输入维度

#### Scenario: 可视化 2D tensor

- **WHEN** 输入数组维度为 2D
- **THEN** helper MUST 直接以 heatmap 形式展示
- **AND** subplot MUST 显示原始输入维度

#### Scenario: 可视化 3D tensor

- **WHEN** 输入数组维度为 3D
- **THEN** helper MUST 默认只取第 0 个 batch 样本
- **AND** MUST 将该样本按二维 heatmap 展示
- **AND** subplot MUST 显示原始输入维度
- **AND** subplot MUST 明确提示当前展示的是 Batch0 或调用方指定的 batch index

#### Scenario: 拒绝高维 tensor

- **WHEN** 输入数组维度大于 3D 或小于 1D
- **THEN** helper MUST 抛出清晰的 `ValueError`

### Requirement: 多输入同图自动排版

helper SHALL 支持一次输入多个 tensor/matrix，并自动将它们排布在同一张 `matplotlib` figure 中。

#### Scenario: 可视化多个输入

- **WHEN** 调用方一次传入多个 tensor/matrix
- **THEN** helper MUST 为每个输入创建一个 subplot
- **AND** MUST 根据输入数量自动计算合理的行列布局

#### Scenario: 标题数量匹配输入

- **WHEN** 调用方提供 titles
- **THEN** helper MUST 将每个 title 应用到对应 subplot
- **AND** 当 titles 数量与输入数量不匹配时 MUST 给出清晰错误

### Requirement: matplotlib 输出控制

helper SHALL 提供常用 matplotlib 输出控制选项，支持交互查看和文件保存。

#### Scenario: 保存 figure

- **WHEN** 调用方传入 `save_path`
- **THEN** helper MUST 将生成的 figure 保存到指定路径

#### Scenario: 控制是否 show

- **WHEN** 调用方设置 `show=False`
- **THEN** helper MUST 不调用 `plt.show()`

#### Scenario: 自定义样式

- **WHEN** 调用方传入 `cmap`、`figsize` 或 `colorbar` 选项
- **THEN** helper MUST 将这些选项应用到生成的 heatmap figure

### Requirement: DeCI 中间量可视化示例

系统 SHALL 在 DeCI 相关代码中提供调用 helper 查看关键中间量的示例，且该示例必须是非默认执行的调试用代码。

#### Scenario: 查看 DeCI 中间量示例

- **WHEN** 开发者查看 DeCI 模型相关代码
- **THEN** 代码中 MUST 提供如何使用 helper 可视化 `x_enc`、`x_embed`、`trend`、`seasonal` 或 `res` 等中间量的示例
- **AND** 示例 MUST 默认不执行
- **AND** 示例 MUST 不改变 DeCI 的 forward 返回值或训练行为

### Requirement: S-DeCI 因果学习中间量可视化

系统 SHALL 支持在 `S-DeCI` 接入模块 2/3 后显式可视化关键中间量，并且默认不改变训练、验证或推理行为。

#### Scenario: 可视化 Cycle 和因果矩阵

- **WHEN** 用户显式开启 `S-DeCI` 因果学习可视化
- **THEN** 系统 MUST 调用 `utils.tensor_visualization.visualize_tensors`
- **AND** MUST 能保存 Cycle/seasonal feature、模块 2 temporal prediction、temporal prediction error、$A_{\mathrm{lag}}$、$A_0$、$A_{\mathrm{cls}}$、连续邻接矩阵 `A_learned` 和阈值化邻接矩阵
- **AND** 在启用模块 3 时 MUST 能保存 `C_clipped`、Poincare 投影结果或 `H0`、`H_gcn`、`z_global` 或 `logmap0(z_global)`
- **AND** 设计原因 MUST 写明：默认模块 2 已从静态 `C_hat` 重构改为历史时间窗预测未来时间点，因此可视化应重点显示预测值、预测误差和时序因果图，而不是把静态重构写成默认中间量

#### Scenario: 区分训练集和测试集中间量

- **WHEN** 用户显式开启 `S-DeCI` 因果学习可视化
- **THEN** 每个 fold 训练结束后 SHOULD 分别保存训练集 batch 和测试集 batch 的中间量 heatmap
- **AND** 文件名 SHOULD 能区分 `train` 和 `test`
- **AND** 测试集 forward MUST NOT 将 label 输入模型，label 仅可在 forward 后用于可视化标注

#### Scenario: 默认不执行可视化

- **WHEN** 用户未显式开启可视化配置或手动调用可视化方法
- **THEN** `S-DeCI` MUST NOT 在常规训练、验证或推理中保存图片
- **AND** 可视化逻辑 MUST NOT 改变模型 forward 返回值

#### Scenario: 三维张量显示 Batch0 提示

- **WHEN** 可视化 Cycle/seasonal feature、temporal prediction、prediction error 或 HGCN 中间表示等 3D 张量
- **THEN** helper MUST 默认只显示 Batch0
- **AND** subplot 标题或副标题 MUST 显示原始维度并提示当前展示的 batch index

### Requirement: S-DeCI 可视化输出可定位

系统 SHALL 允许用户配置或指定 `S-DeCI` 因果学习可视化输出位置。

#### Scenario: 保存到指定目录

- **WHEN** 用户提供可视化输出目录或保存路径
- **THEN** 系统 MUST 将 heatmap 图片保存到该位置
- **AND** 文件名 MUST 能区分 Cycle feature、temporal prediction、prediction error、$A_{\mathrm{lag}}$、$A_0$、$A_{\mathrm{cls}}$、因果矩阵和方向差异矩阵

#### Scenario: 训练中限制保存频率

- **WHEN** 用户在训练中启用可视化
- **THEN** 系统 MUST 支持通过配置或手动调用限制保存频率
- **AND** 默认配置 MUST 避免每个 batch 都保存图片

### Requirement: 最终 epoch 的 train/test t-SNE 可视化

系统 SHALL 支持在最后一个 epoch 训练完成后，将训练集和测试集的模型表示投影到二维 t-SNE 图中，用于观察类别与数据划分的分布关系。

#### Scenario: 生成 train/test 联合 t-SNE

- **WHEN** 用户显式开启 `S-DeCI` 可视化
- **THEN** 系统 SHOULD 收集训练集和测试集样本在最后 epoch 后的模型表示
- **AND** 启用模块 3 时 SHOULD 优先使用 `logmap0(z_global)` 或等价缓存表示作为 t-SNE 输入
- **AND** MUST 将训练集和测试集点绘制在同一张二维图中
- **AND** MUST 用不同 marker 或样式区分 train/test
- **AND** MUST 用颜色区分真实 label
- **AND** 测试集 label MUST 仅用于绘图上色，不得作为模型 forward 输入

### Requirement: HPEC 中间量可视化

系统 SHALL 使用既有 tensor 可视化 helper 显示模块 4 HPEC 的关键中间量。

#### Scenario: 保存 HPEC heatmap

- **GIVEN** 用户显式开启 `S-DeCI` 可视化
- **WHEN** 启用模块 4并完成一个 fold 的训练
- **THEN** 系统 SHOULD 保存 HPEC prototype、angle matrix、aperture 或 `psi`、energy matrix、prediction 和 label 对照 heatmap
- **AND** 文件名 SHOULD 能区分 `train`、`test`、fold 和模块 4 内容

#### Scenario: 显示 label 与预测

- **WHEN** 可视化 HPEC prediction 和 label
- **THEN** 可视化结果 MUST 能显示真实 label
- **AND** 测试集真实 label MUST 仅用于 forward 之后的绘图标注，不得作为模型输入

#### Scenario: HPEC 可视化默认不影响训练

- **WHEN** 用户未显式开启可视化配置
- **THEN** 系统 MUST NOT 自动保存 HPEC 图片
- **AND** 可视化缓存和保存逻辑 MUST NOT 改变模型输出、loss 计算或反向传播结果

### Requirement: 可视化样本相关矩阵 adjacency

S-DeCI 中间量可视化 SHALL 在模块 2 关闭且模块 3 使用样本相关矩阵时显示该 adjacency。

#### Scenario: 保存 sample correlation heatmap

- **GIVEN** 用户显式开启 `visualize_causal`
- **AND** `S-DeCI` 使用 sample correlation adjacency 进入模块 3
- **WHEN** 每个 fold 训练结束后保存中间量 heatmap
- **THEN** 可视化 MUST 包含 sample correlation adjacency
- **AND** 文件名 MUST 能区分 train/test 和 fold

#### Scenario: 与模块 3 中间量对照

- **GIVEN** sample correlation adjacency 已缓存
- **WHEN** 可视化模块 3 中间量
- **THEN** 系统 MUST 能同时显示 sample correlation adjacency、Module3 normalized adjacency、`H_gcn` 和 `z_global`
- **AND** 三维张量仍 MUST 按现有规则默认显示 Batch0 并提示维度

#### Scenario: 模块 2 开启路径保持原可视化

- **GIVEN** `use_causal_module2 == 1`
- **WHEN** 用户保存 S-DeCI 中间量可视化
- **THEN** 系统 MUST 继续显示 `A_learned`、`A_learned_binary`、$A_{\mathrm{lag}}$、$A_0$、$A_{\mathrm{cls}}$ 和模块 2 temporal prediction 相关中间量
- **AND** sample correlation adjacency MUST NOT 替代模块 2 可视化内容

### Requirement: 多 prototype heatmap 可视化

S-DeCI 中间量可视化 SHALL 支持展示多 prototype HPEC 的 prototype 张量和 prototype-level energy。

#### Scenario: 展示多 prototype 张量

- **GIVEN** HPEC prototype 张量形状为 `[classes, K, hidden_dim]`
- **WHEN** 用户显式保存 S-DeCI 中间量 heatmap
- **THEN** 系统 MUST 将 prototype 张量转换为可读的二维或三维 heatmap 输入
- **AND** 标题 MUST 标明类别数、每类 prototype 数和 hidden 维度

#### Scenario: 展示 prototype-level energy

- **GIVEN** HPEC forward 产生 prototype-level energy
- **WHEN** 用户显式保存 S-DeCI 中间量 heatmap
- **THEN** 系统 SHOULD 保存 prototype-level energy 或 similarity heatmap
- **AND** 标题 MUST 能区分类别级 energy 和 prototype-level energy

### Requirement: t-SNE 显示每类多个 prototype

最终 epoch 的 train/test t-SNE 可视化 SHALL 显示每个类别下的多个 prototype，并与样本点使用同一 t-SNE 投影。

#### Scenario: 绘制多 prototype 点

- **GIVEN** 用户显式开启 `visualize_causal`
- **AND** `hpec_prototypes_per_class > 1`
- **WHEN** 系统生成最终 epoch t-SNE
- **THEN** 图中 MUST 绘制所有 prototype 点
- **AND** prototype 点 MUST 使用与 train/test 不同的 marker
- **AND** prototype 点颜色 MUST 与其类别 label 对应

#### Scenario: prototype 与样本同坐标系

- **GIVEN** 训练集样本、测试集样本和 prototype 都需要显示
- **WHEN** 系统执行 t-SNE 降维
- **THEN** 系统 MUST 将样本 embedding 和 prototype embedding 拼接后一起执行 t-SNE
- **AND** 系统 MUST NOT 将 prototype 用单独 t-SNE 或不同坐标系投影后叠加到图上

#### Scenario: 单 prototype t-SNE 兼容

- **GIVEN** `hpec_prototypes_per_class == 1`
- **WHEN** 系统生成最终 epoch t-SNE
- **THEN** 图中 MUST 仍能显示每个类别的单 prototype 点
- **AND** 图例或标题 MUST 不因 prototype 数量为 1 而报错
