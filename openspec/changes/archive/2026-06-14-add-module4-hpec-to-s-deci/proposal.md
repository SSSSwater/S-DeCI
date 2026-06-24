## Why

当前 `S-DeCI` 已经完成模块 1、模块 2 和模块 3，能够得到 128 维可配置的双曲中心点 `z_global`，但分类仍然使用普通线性分类头。按照 `docs/新模块设计.md` 的模块 4 设计，下一步需要引入 HPEC 能量分类，使双曲中心点与类别原型之间的几何关系直接成为分类目标。

## What Changes

- 在 `S-DeCI` 中新增模块 4 HPEC 分类能力：使用模块 3 输出的 `z_global` 或 `logmap0(z_global)` 与类别原型计算能量函数。
- 将当前模块 3 的普通分类损失替换为 HPEC energy loss，训练目标继续保留模块 2 的 reconstruction、DAG 和 L1 sparsity 辅助项。
- 参考 `docs/新模块设计.md` 和 `reference/HPEC-main/` 的原型初始化、Poincare Ball、角度/能量计算和训练方式，但正式运行代码不得直接依赖 `reference/` 路径。
- 在 `layers/` 或合适位置新增可复用 HPEC 原型/能量计算组件；模块 4 的装配、缓存和训练入口保留在 `models/S_DeCI.py`。
- 为模块 4 新增中文注释，并缓存原型、角度或能量矩阵、预测能量、`z_global` 等诊断量。
- 扩展可视化：显式开启可视化时展示 HPEC energy、prototype、prediction energy、label/prediction 对照，以及最终 epoch train/test t-SNE 中的 HPEC 表示。
- 更新训练入口和测试脚本参数，使模块 4 可开关、原型维度/半径/温度/边界等关键超参数可配置。
- **BREAKING**：当 `use_hpec_module4=1` 时，`S-DeCI` 的分类 loss 不再由普通线性分类头产生，而由 HPEC energy loss 产生；模块 4 关闭时应保留模块 3 线性分类头回退路径。
- 回滚方案：设置 `use_hpec_module4=0`，回退到当前模块 3 `logmap0(z_global)` 线性分类路径；必要时移除新增 HPEC 层、参数和可视化缓存，不影响原始 `models/DeCI.py`。

## Capabilities

### New Capabilities

- `module4-hpec-classification`: 定义模块 4 HPEC 原型、能量函数、HPEC energy loss、预测规则和诊断缓存要求。

### Modified Capabilities

- `s-deci-model`: `S-DeCI` 将从模块 3 线性分类扩展为可配置地使用模块 4 HPEC energy 分类，并在启用模块 4 时以 HPEC energy loss 替换普通分类 loss。
- `module3-hgcn-readout`: 模块 3 输出的 `z_global`/`logmap0(z_global)` 将作为模块 4 HPEC 输入，需保持维度、流形语义和缓存稳定。
- `tensor-visualization-helper`: S-DeCI 可视化要求扩展到模块 4 的原型、energy matrix、预测能量和 HPEC 表示。
- `training-test-scripts`: 训练入口和根目录测试脚本需要暴露模块 4/HPEC 参数，并支持低预算验证 HPEC loss 能跑通。

## Impact

- 影响代码范围：
  - `models/S_DeCI.py`
  - `layers/` 下新增或扩展 HPEC energy/prototype 相关组件
  - `exp/exp_classification_CV.py`
  - `run_cv.py`
  - `test_training_smoke.py`
  - `test_matai_small_sample.py`
  - `utils.tensor_visualization` 或现有可视化调用处
  - `docs/` 下新增模块 4 实现说明文档，原始 `docs/新模块设计.md` 不修改
- 参考范围：
  - `reference/HPEC-main/` 中 HPEC 原型、Poincare Ball 和 energy/entailment loss 相关实现
  - 现有 `layers/hyperbolic_gcn_layer.py` 与 `S-DeCI` 模块 3 缓存输出
- 行为影响：
  - 启用模块 4 后，分类依据从线性分类头切换为 HPEC energy 最小化。
  - 总 loss 变为 `Loss_HPEC(z_global, label) + alpha*Loss_Recon + lambda*Loss_DAG + gamma*L1`。
  - 模块 2 因果图仍应通过模块 3/4 分类目标接收梯度。
