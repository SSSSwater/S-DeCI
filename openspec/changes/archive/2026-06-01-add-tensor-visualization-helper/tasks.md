## 1. 可视化 helper 实现

- [x] 1.1 新增 `utils/tensor_visualization.py`，提供可 import 的主函数 `visualize_tensors(*items, ...)`。
- [x] 1.2 实现输入转换逻辑，支持 `torch.Tensor`、`numpy.ndarray` 和 array-like 对象，并对 torch tensor 使用 `detach().cpu().numpy()`。
- [x] 1.3 实现 1D、2D、3D 维度处理规则：1D 转单行 heatmap，2D 直接 heatmap，3D 默认取 Batch0。
- [x] 1.4 对 0D、超过 3D、空输入、titles 数量不匹配等情况抛出清晰 `ValueError`。

## 2. 多输入排版与 matplotlib 输出

- [x] 2.1 实现多个输入的自动 subplot 行列排版，并为每个输入绘制 heatmap。
- [x] 2.2 支持 `titles`、`cmap`、`figsize`、`colorbar`、`save_path`、`show` 等常用参数。
- [x] 2.3 确保 `show=False` 时不调用 `plt.show()`，并在传入 `save_path` 时保存 figure。
- [x] 2.4 返回 `matplotlib` figure 和 axes，方便调用方继续定制或测试。

## 3. DeCI 调试示例

- [x] 3.1 在 DeCI 相关代码中添加非默认执行的可视化调用示例。
- [x] 3.2 示例应展示如何查看 `x_enc`、`x_embed`、`trend`、`seasonal` 或 `res` 等关键中间量。
- [x] 3.3 确保示例不会在默认 forward、训练、验证或推理路径中执行。

## 4. 验证

- [x] 4.1 使用 `.venv` Python 编译检查新增 helper 和修改过的 DeCI 文件。
- [x] 4.2 运行轻量脚本验证 1D、2D、3D、多输入、`save_path` 和 `show=False` 行为。
- [x] 4.3 验证错误输入会抛出清晰 `ValueError`。
- [x] 4.4 记录验证命令和结果，便于后续归档前检查。

## 5. 验证记录

- `.\.venv\Scripts\python.exe -m py_compile utils\tensor_visualization.py models\DeCI.py` 通过。
- 轻量脚本验证通过：1D、2D、3D 输入可同图自动排版，`show=False` 不弹窗，`save_path` 成功保存 `.tmp_visualization_tests\multi_heatmap.png`。
- DeCI 示例函数验证通过：`visualize_deci_intermediates_example(...)` 成功保存 `.tmp_visualization_tests\deci_example.png`，且未修改 `Model.forward()` 默认路径。
- 复核时将 DeCI 示例函数默认值修正为 `show=False`，验证通过：默认调用不会触发 `plt.show()`，仍可通过参数显式 `show=True`。
- 错误输入验证通过：空输入、0D scalar、4D 输入、titles 数量不匹配均抛出清晰 `ValueError`。
- 补充验证通过：每个 subplot 标题会显示原始输入维度；3D 输入会额外显示 `showing Batch0`，指定 `batch_index=2` 时显示 `showing Batch2`。
