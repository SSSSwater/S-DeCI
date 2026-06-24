## Why

当前 `S-DeCI` 已经逐步接入模块 1、模块 2、模块 3 与模块 4，但不同模块之间的启用/禁用语义还不够统一。为了便于做消融实验、定位训练问题，并比较 DeCI、因果图、HGCN/HPEC 与普通 GCN 路径的贡献，需要为各模块建立明确的开关和退化路径。

## What Changes

- 为 `S-DeCI` 增加模块级启用/禁用参数，使模块 1、模块 2、模块 3/4 的组合实验可以通过 CLI 直接配置。
- 当模块 1 禁用时，不执行 DeCI 高频/周期分解，直接将原始输入时间序列投影为 `d_model` 维节点特征，作为后续模块输入。
- 保留模块 2 禁用时使用样本级相关矩阵作为 adjacency 的既有行为。
- 将模块 3 与模块 4 作为一组联合启用/禁用：启用时继续使用 HGCN + HPEC 路径；禁用时退化为普通 GCN，将模块 2 输出的因果矩阵或模块 2 禁用后的相关矩阵作为 adjacency，并使用模块 1 输出的节点特征进行图学习和分类。
- 训练入口与测试脚本暴露对应参数，并用中文 help 描述各开关含义。
- 可视化中间量需要能区分 DeCI Cycle feature、raw projected feature、HGCN/HPEC 表征和 GCN fallback 表征。
- 回滚方案：保留当前默认配置为现有全模块路径；若新开关或 GCN fallback 影响训练，可将默认值恢复为模块 1/2/3/4 全启用，并移除新增 fallback 分支而不影响已有 `S-DeCI` 主路径。

## Capabilities

### New Capabilities

- `s-deci-gcn-fallback`: 定义模块 3/4 禁用时的普通 GCN 退化分类路径。

### Modified Capabilities

- `s-deci-model`: 增加模块 1、模块 2、模块 3/4 的开关语义，以及各模块禁用时的输入输出约束。
- `module3-hgcn-readout`: 明确 HGCN/HPEC 路径与 GCN fallback 路径之间的 adjacency 与节点特征来源关系。
- `training-test-scripts`: 训练入口和测试脚本需要暴露新的模块开关参数，并覆盖关键组合的训练验证。

## Impact

- 影响 `models/S_DeCI.py` 中的模块初始化、forward 路径、辅助 loss 暴露与中间量缓存。
- 影响 `layers/` 中可能新增的普通 GCN fallback 层，或复用已有图卷积实现。
- 影响 `run_cv.py`、根目录测试脚本和 experiment 训练流程中的 CLI 参数、batch 调用、loss 汇总与指标打印。
- 不修改 `docs/` 下作为初始参考的设计文档；如需记录修改后的设计说明，应在 OpenSpec 或新的项目文档中维护。
