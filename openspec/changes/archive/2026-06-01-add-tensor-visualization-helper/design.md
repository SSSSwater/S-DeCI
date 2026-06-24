## Context

项目目前已有 `utils/` 目录用于训练辅助函数和指标工具，但缺少一个统一的中间量可视化 helper。模型调试时，开发者经常需要观察输入时间序列、embedding、trend/seasonal/residual 等 tensor 的形态；如果每次都在模型内部临时写 matplotlib 代码，会造成重复、侵入训练路径，也容易忘记关闭绘图。

本变更新增一个独立工具文件，并在 DeCI 相关代码中提供非默认执行的调用示例。工具应当服务于开发调试，不改变训练、验证或推理的默认行为。

## Goals / Non-Goals

**Goals:**
- 在 `utils/` 下新增可复用的 tensor/matrix 可视化 helper。
- 支持 1D、2D、3D 输入，其中 3D 输入默认只展示 Batch 维度第 0 个样本。
- 支持一次传入多个 tensor/matrix，并自动在同一张 figure 中排版为多个 subplot。
- 使用 `matplotlib` 绘制 heatmap，支持 `cmap`、`title`、`colorbar`、`save_path`、`show` 等调试选项。
- 兼容 `torch.Tensor`、`numpy.ndarray` 和可被 `numpy.asarray` 转换的对象。
- 在 DeCI 代码中加入如何调用该 helper 查看中间量的示例，但示例必须是 opt-in，不得默认执行。

**Non-Goals:**
- 不在训练过程中默认弹窗、保存图片或引入额外耗时。
- 不改变 DeCI 的 forward 输出、模型结构、训练指标或 checkpoint 行为。
- 不实现复杂交互式可视化、动态图或 dashboard。
- 不为高维 tensor 做自动降维；超过 3D 的输入应清晰报错。

## Decisions

1. 工具文件放在 `utils/tensor_visualization.py`。

   原因：该能力是跨模块调试工具，放在 `utils/` 与现有辅助函数组织方式一致，也方便 `models/`、`layers/`、`data_provider/` 等模块按需 import。

   备选方案：放在根目录。根目录更醒目，但会增加顶层文件噪音，也不如 `utils/` 语义清晰。

2. 提供单一主函数 `visualize_tensors(*items, ...)`。

   原因：调用方可以直接传入一个或多个 tensor/matrix；函数内部统一做转换、维度处理、自动排版和绘图输出，降低调试门槛。

   备选方案：分别提供 `visualize_1d`、`visualize_2d`、`visualize_3d`。这样 API 更细，但调用者需要先判断维度，重复心智负担更高。

3. 3D tensor 默认展示 Batch0。

   原因：项目中常见中间量包含 batch 维度，例如 `[B, T, N]` 或 `[B, N, D]`。调试时通常只需要先观察一个样本，默认取 Batch0 能避免一次绘制过多内容。

   备选方案：把 3D tensor 每个 batch 都画出来。这样信息更全，但容易生成过大的 figure，不适合快速调试。

4. 1D tensor 转为单行 heatmap，2D tensor 直接 heatmap。

   原因：用户明确要求二维和一维按热力图可视化。单行 heatmap 可以保持 1D/2D 输出形式一致，也便于与多个 subplot 一起排版。

   备选方案：1D 使用折线图。折线图适合趋势查看，但不符合本次需求的 heatmap 统一展示。

5. DeCI 中使用非默认执行示例，而不是在 forward 内直接绘图。

   原因：forward 默认绘图会破坏训练性能、批处理和无界面环境；示例应当指导开发者在需要时复制/启用，而不影响正常实验。

   备选方案：增加 `configs.visualize_debug` 并在 forward 内条件绘图。这样更自动，但会把调试行为耦合到模型配置和训练路径中，风险更高。

## Risks / Trade-offs

- [Risk] matplotlib 在无 GUI 环境中 `show()` 可能不可用 -> 默认允许 `show=False` 和 `save_path`，调用方可保存图片而不弹窗。
- [Risk] 传入 GPU tensor 或 requires_grad tensor 可能导致转换错误或保留计算图 -> helper 在转换时应使用 `detach().cpu().numpy()`。
- [Risk] 多个输入 shape 差异较大时，subplot 尺寸可能不完美 -> 自动排版优先保证可用，调用方可通过 `figsize` 覆盖。
- [Risk] DeCI 示例被误认为默认逻辑 -> 示例必须明确标注为调试用，并保持注释或 opt-in 函数形式。

## Migration Plan

1. 新增 `utils/tensor_visualization.py`，实现主函数和内部转换/排版逻辑。
2. 在 DeCI 相关文件中添加非默认执行的调用示例，展示如何观察 `x_enc`、`x_embed`、`trend`、`seasonal`、`res`。
3. 添加轻量验证脚本或命令，确认 helper 可处理 1D、2D、3D、多输入和保存图片路径。
4. 回滚时删除新增 helper 文件和 DeCI 示例代码即可，不需要修改训练数据或模型权重。

## Open Questions

- 无。
