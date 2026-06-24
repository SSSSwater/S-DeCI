## Why

模型开发和调试过程中，经常需要查看中间 tensor、矩阵或特征图的形态；目前项目缺少一个可在多个模块中直接调用的轻量可视化工具，导致调试时容易重复写临时 matplotlib 代码。

新增一个独立 Python helper 可以把 1D/2D/3D 张量统一转成热力图查看，并支持一次输入多个矩阵自动排版，方便快速观察 DeCI block、baseline 模型或数据处理流程中的中间量。

## What Changes

- 新增一个独立 `.py` 工具文件，提供可被多个模块 import 的可视化函数。
- 函数使用 `matplotlib` 绘图，支持 `torch.Tensor`、`numpy.ndarray` 或可转换为数组的输入。
- 支持同时输入多个矩阵/张量，并在同一张 figure 中自动计算 subplot 排版。
- 对 3D tensor，默认只可视化 Batch 维度的第 0 个样本；其余维度按二维热力图展示。
- 对 2D tensor，直接按 heatmap 可视化。
- 对 1D tensor，转换为单行 heatmap 可视化。
- 支持 title、cmap、是否显示 colorbar、保存路径、是否 `show()` 等常用调试选项。
- 在 DeCI 模型相关代码中添加调用该工具查看若干关键中间量的示例，方便开发者理解如何可视化输入、嵌入、trend/seasonal/residual 等调试对象。
- 不引入除现有 `matplotlib`/`numpy`/可选 `torch` 以外的新强制依赖。

## Capabilities

### New Capabilities
- `tensor-visualization-helper`: 可复用的 tensor/matrix 中间量可视化 helper，支持 1D/2D/3D 输入、多图自动排版、Batch0 展示和 matplotlib 输出控制。

### Modified Capabilities

## Impact

- 影响文件：新增一个独立 Python 工具文件，预计放在 `utils/` 或根目录附近的通用工具位置，具体路径在 design 阶段确定；DeCI 模型相关文件中会新增非默认执行的调用示例或注释示例。
- 影响系统：开发调试流程、模型内部中间量观察、数据处理可视化；不改变训练、验证或模型推理行为。
- 依赖：使用 `matplotlib` 和 `numpy`；兼容 `torch.Tensor` 但不要求调用方必须传入 torch 对象。
- 回滚方案：删除新增 helper 文件以及任何新增文档/示例引用；现有训练和模型代码无需回滚。
