## Why

当前 `S-DeCI` 的模块 3 默认依赖模块 2 学到的因果邻接矩阵 `A_learned`。当用户关闭模块 2 时，模块 3 缺少图结构输入，导致无法退化为“直接使用样本自身相关系数矩阵做 HGCN 图传播”的原始模块 3 设计。

## What Changes

- 在数据加载流程中支持为每个时间序列样本同时读取对应的相关系数矩阵 `.mat` 文件。
- 当 `use_causal_module2=0` 且 `use_hgcn_module3=1` 时，`S-DeCI` 不再要求模块 2 输出，而是将该样本 batch 的相关系数矩阵传入模块 3 作为 adjacency。
- 相关矩阵文件解析以“同一 subject、同一 protocol 的 correlation matrix”为准，需支持用户描述的 `sub-xxx_xxx_features_sub_correlation_matrix.mat` 命名模式，并兼容当前数据集中已有的 `sub-xxx_<protocol>_correlation_matrix.mat` 命名模式。
- 模块 2 开启时行为保持不变：模块 3 继续使用 `A_learned`，模块 2 auxiliary loss 继续参与训练。
- 模块 2 关闭时不计算 reconstruction、DAG、L1 auxiliary loss；分类 loss 仍按模块 3/4 当前配置计算。
- 训练脚本和根目录测试脚本增加显式参数，用于控制是否在模块 2 关闭时加载并传递样本相关矩阵。
- 可视化中在模块 2 关闭路径下显示 sample correlation adjacency，便于与模块 2 学到的 `A_learned` 路径对比。
- **BREAKING**：无。默认 `use_causal_module2=1` 的当前训练路径保持不变。
- 回滚方案：关闭新增的相关矩阵回退开关，或保持 `use_causal_module2=1`；必要时移除数据加载中的额外返回字段和 `S-DeCI.forward(..., correlation_matrix=...)` 兼容参数。

## Capabilities

### New Capabilities

- `sample-correlation-matrix-provider`: 定义如何为时间序列样本解析、加载和 batch 化对应相关系数矩阵。

### Modified Capabilities

- `s-deci-model`: `S-DeCI` 在模块 2 关闭但模块 3 开启时，使用样本相关系数矩阵作为模块 3 adjacency。
- `module3-hgcn-readout`: 模块 3 的 adjacency 输入从单个全局 `[N, N]` 扩展为支持 batch 级 `[B, N, N]`，用于每个样本不同图结构。
- `training-test-scripts`: 训练入口和测试脚本需要支持模块 2 关闭时加载相关矩阵并传递给模型。
- `tensor-visualization-helper`: S-DeCI 可视化需要支持显示 sample correlation adjacency。

## Impact

- 影响代码范围：
  - `data_provider/data_loader_CV.py`
  - `data_provider/data_factory_CV.py`
  - `models/S_DeCI.py`
  - `layers/hyperbolic_gcn_layer.py`
  - `exp/exp_classification_CV.py`
  - `run_cv.py`
  - `test_training_smoke.py`
  - `test_matai_small_sample.py`
  - `utils.tensor_visualization` 的调用侧
  - 新增实现说明文档，原始 `docs/新模块设计.md` 不修改
- 行为影响：
  - 模块 2 开启：继续使用学习到的因果图。
  - 模块 2 关闭且模块 3 开启：使用每个样本对应的相关系数矩阵作为模块 3 图结构。
  - 模块 2 关闭且相关矩阵缺失：训练应清晰失败并提示缺失文件。
