## Why

当前 ABIDE-120 测试中，绕开模块 2/3/4 的 GCN fallback 可以作为对照基线，但不符合本项目的核心设计。真正需要解决的问题是：在保持 `S-DeCI` 四模块主框架完整启用的前提下，降低 ABIDE 小信噪比时序上的训练集记忆与测试集不可分现象。

该变更用于把抗过拟合能力放回原始设计链路中：模块 1 做去噪，模块 2 学稳定因果关系，模块 3 在双曲空间投影，模块 4 使用 HPEC 多原型能量损失完成分类。

## What Changes

- 强制 ABIDE 主测试配置默认启用 `S-DeCI` 四模块，不再把 GCN fallback 作为 ABIDE 最佳配置默认路径。
- 在模块 1 中加入面向 ABIDE 短时序的去噪机制，包括训练时随机时间窗、temporal dropout / ROI dropout，以及可选 denoising auxiliary loss。
- 调整模块 2，使其从模块 1 去噪后的时序或特征学习预测式 SEM 因果图，并加入图稳定性、稀疏性与低自由度样本残差约束，避免样本图过度自由。
- 调整模块 3，使双曲投影接收模块 2 的因果图，并支持基于因果图的 edge dropout、`z_global` 半径/范数诊断与正则。
- 调整模块 4，使 HPEC 多原型初始化与损失更适合 ABIDE：使用训练表示 warm-start / 聚类式初始化、类内紧凑、类间分离、prototype diversity 与 margin 约束。
- 更新 `test_abide_best_config.py` 默认参数，使其用于完整四模块 S-DeCI 的 ABIDE-120 抗过拟合测试。
- 新增实现说明文档，记录修改后的 ABIDE 四模块抗过拟合设计；不修改 `docs/新模块设计.md` 原始参考文档。
- 回滚方案：保留现有模块开关与 GCN fallback，可通过关闭新增正则参数、恢复旧默认参数，回到当前已验证的 S-DeCI 行为或 GCN fallback 对照。

## Capabilities

### New Capabilities

- `abide-full-sdeci-anti-overfit`: 定义 ABIDE-120 上完整四模块 S-DeCI 的抗过拟合训练能力，包括模块 1 去噪、模块 2 稳定因果学习、模块 3 双曲投影正则、模块 4 多原型正则与测试脚本默认配置。

### Modified Capabilities

- `s-deci-model`: 明确 ABIDE 主路径必须保持模块 1/2/3/4 连续启用，GCN fallback 仅用于消融对照。
- `module2-causal-learning`: 模块 2 需要支持从模块 1 去噪输出进行预测式 SEM 因果学习，并限制样本残差图自由度。
- `module3-hgcn-readout`: 模块 3 需要支持因果图 edge dropout 与双曲表示半径/范数诊断，降低双曲空间过拟合。
- `module4-hpec-classification`: 模块 4 需要支持更稳的 HPEC energy、prototype warm-start、margin 与 prototype 正则。
- `multi-prototype-hpec-classification`: 多 prototype 需要支持类内多样性与类间分离的抗 collapse 约束。
- `training-test-scripts`: ABIDE 专用测试脚本默认参数需要回到完整四模块路径，并提供能复现实验的抗过拟合参数。

## Impact

- 影响模型文件：`models/S_DeCI.py`。
- 影响层文件：`layers/DeCI_Layer.py`、`layers/causal_graph_layer.py`、`layers/hyperbolic_gcn_layer.py`、`layers/hpec_energy_layer.py` 或对应现有层实现。
- 影响训练流程：`exp/exp_classification_CV.py` 中的 loss 汇总、日志、早停指标与可视化。
- 影响数据/脚本：`data_provider/data_factory_CV.py` 的训练时随机时间窗策略，以及 `test_abide_best_config.py`、`run_cv.py` 的参数入口。
- 影响文档：新增一份中文实现说明文档，描述 ABIDE 完整四模块抗过拟合路径与回滚方式。
