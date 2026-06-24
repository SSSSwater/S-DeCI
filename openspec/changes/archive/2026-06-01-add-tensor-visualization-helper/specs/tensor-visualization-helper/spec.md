## ADDED Requirements

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
