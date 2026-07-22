## Why

当前 S-DeCI 的模块 3/4 已具备有向因果图上的双曲传播与多原型 HPEC 分类，但 prototype 更新仍可能受到低置信度训练样本影响，且图级表征可能过度依赖少数高显著 ROI，导致小样本测试泛化不稳定。BrainCL 提出的可靠样本 EMA prototype 与显著性遮挡互补学习为这一问题提供了可验证的训练机制，但必须保持模块 2 时序因果学习和模块 3/4 双曲叙事不被替代。

## What Changes

- 将模块 4 的默认慢更新路径从 Sinkhorn 均衡分配替换为可靠 true-positive (TP) 样本驱动的多 prototype EMA 更新：仅使用预测正确、置信度达标且标准/互补视图一致的训练样本；该更新独立于总 loss，不通过反向传播移动 prototype。
- 新增因果显著性互补视图：保留标准样本的模块 2 时序因果图，仅对送入模块 3 的节点特征进行动态 ROI 遮挡，并以共享模块 3/4 生成互补双曲表征。
- 新增可选 Poincare 双视图一致性损失，约束同一受试者的标准与遮挡图表征在双曲空间保持稳定；默认关闭，避免未经完整实验验证的新增监督成为主路线。
- 新增可选多阶因果可达性编码：由模块 2 的有向分类图构建小 hop 的有向传播先验，在进入 HGCN 前增强节点特征；不改变模块 2 的因果图学习目标。
- 为每项新增机制提供独立开关、参数、TensorBoard 诊断、可视化和消融入口；保留旧 Sinkhorn 路径作为明确的 legacy 对照，支持回滚。
- 新增一份修改后设计说明，不修改 `docs/` 中现有参考设计文档。

## Capabilities

### New Capabilities

- `causal-salience-complementary-learning`: 基于时序因果信息流和节点表征活跃度构建动态遮挡互补视图，并在双曲空间执行可选一致性训练。
- `reliable-ema-prototype-updates`: 使用可靠 TP 样本和视图一致性权重更新每类多个 HPEC prototype，替代默认 Sinkhorn 更新。
- `multi-hop-causal-reachability-encoding`: 用小范围有向因果可达性编码增强模块 3 的节点输入并支持消融。

### Modified Capabilities

- `multi-prototype-hpec-classification`: 原型更新机制增加可靠 TP EMA 默认路径、Sinkhorn legacy 回退与更新诊断。
- `module3-hgcn-readout`: 模块 3 支持共享参数互补视图传播及可选多阶有向因果拓扑编码。
- `module4-hpec-classification`: 模块 4 支持标准/互补视图的 HPEC evidence 诊断和可靠 prototype 更新控制。
- `s-deci-model`: 模型 forward、损失聚合和训练配置支持上述可选机制，且保持默认四模块主路线兼容。
- `training-test-scripts`: 训练入口与超参数扫描支持新开关、完整消融、TensorBoard 及最终结果记录。

## Impact

- 影响 `models/S_DeCI.py`、`layers/hyperbolic_gcn_layer.py`、`layers/hpec_energy_layer.py`、`exp/exp_classification_CV.py`、`run_cv.py`、`test_*_best_config.py` 和 `sweep_hparam.py`。
- 不新增外部依赖；新增少量 PyTorch 模块与训练期共享分支，推理默认只运行标准分支，不增加部署期双分支开销。
- 默认所有新增开关关闭或保留当前行为，除了明确切换后的可靠 EMA 路径；发现完整 5-fold/50+ epoch 无稳定收益时可回退为 `hpec_prototype_update_mode=sinkhorn_ema`、关闭互补视图、多阶编码和一致性损失。
