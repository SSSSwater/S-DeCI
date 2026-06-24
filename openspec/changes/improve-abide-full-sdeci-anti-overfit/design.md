## Context

当前 ABIDE-120 实验显示，完整四模块 S-DeCI 在训练集上很快拟合，但验证集 `z_global` 与类别分离不足。此前作为对照测试的 GCN fallback 可在早期得到略高指标，但它绕开了模块 2 因果学习和模块 3/4 双曲原型分类，不符合本项目的核心模型设计。

本设计以 `docs/新模块设计.md` 为初始参考，但不修改该文档。实现后的工程说明将新建中文文档记录。设计边界是：ABIDE 主路径必须保留模块 1、模块 2、模块 3、模块 4，所有抗过拟合策略都放在这些模块内部或训练调度中完成。

## Goals / Non-Goals

**Goals:**

- 让 ABIDE 默认测试脚本启用完整 S-DeCI 四模块，而不是退化为 GCN fallback。
- 让模块 1 从“提取特征”进一步承担“去噪”职责，输出更稳定的时序或节点特征给模块 2。
- 让模块 2 在去噪输出上学习更稳定的预测式 SEM 因果图，并减少样本级残差图的自由度。
- 让模块 3 在双曲空间投影时保留因果图结构，同时通过 edge dropout 与半径诊断抑制过拟合。
- 让模块 4 的 HPEC 多原型损失更稳定，避免 prototype collapse 和训练样本记忆。
- 在日志和可视化中清楚标注四模块是否启用、主要正则项、因果图稳定性、`z_global` 半径与 prototype 匹配情况。

**Non-Goals:**

- 不把 GCN fallback、DeCI 单模型或其他对照模型作为 ABIDE 主方案。
- 不引入与当前设计无关的 population graph 或表型图模型。
- 不修改 `docs/新模块设计.md` 原始参考文档。
- 不使用测试集信息做 prototype 初始化、因果图约束或 denoising 统计。

## Decisions

### Decision 1: ABIDE 默认路径必须完整启用模块 1/2/3/4

`test_abide_best_config.py` 将默认使用 `use_deci_module1=1`、`use_causal_module2=1`、`use_hyperbolic_modules34=1`、`use_hpec_module4=1`。GCN fallback 保留为消融开关，但不再作为 ABIDE 默认“最佳配置”。

理由：用户明确要求基本框架不能偏离。fallback 的作用是判断模块贡献，而不是替代模块。

替代方案：继续使用 correlation GCN fallback 作为默认。该方案虽然早期指标略稳，但绕开因果图与双曲原型，不符合本变更目标。

### Decision 2: 模块 1 增加训练期去噪增强

模块 1 增加以下可配置机制：

- `module1_random_crop`：训练时从长于 `seq_len` 的时序中随机裁剪窗口；验证/测试保持确定性裁剪。
- `module1_temporal_dropout`：训练时随机置零部分时间点或时间段。
- `module1_roi_dropout`：训练时随机置零部分 ROI 输入。
- `module1_denoise_loss_weight`：可选 denoising auxiliary loss，用被扰动输入的模块 1 输出贴近干净输入输出。

理由：ABIDE-120 中模型快速记忆训练集，说明输入噪声和短时序局部模式容易被模型捕获。去噪约束可以让后续因果图学习更依赖稳定成分。

替代方案：只增加 dropout。普通 feature dropout 不一定作用在时序噪声源头，且不能显式约束模块 1 输出稳定性。

### Decision 3: 模块 2 使用去噪输出进行预测式 SEM，并限制样本图自由度

模块 2 在 `causal_learning_target=temporal_sem` 下优先使用模块 1 的去噪时序或去噪节点特征构造预测式 SEM loss。全局共享因果图仍是主图；样本级 residual graph 需要低秩、低幅度、强正则，默认更保守。

新增或调整约束：

- `lambda_causal_stability`：同一 batch 或增强视图下因果图的一致性损失。
- `sample_graph_delta_scale` 默认降低，避免样本图吞掉共享因果图作用。
- `lambda_sample_graph_l1` 与 `lambda_sample_graph_deviation` 在 ABIDE 默认中非零。
- DAG、sparsity 与 temporal prediction loss 使用 warmup，避免训练初期因果图塌缩。

理由：模块 2 的目标是学习稳定因果关系，而不是为每个样本自由拟合一张图。ABIDE 短时序下必须增强共享图与稳定图的约束。

替代方案：直接用 correlation matrix 替代因果图。该方案属于 fallback，不符合主框架。

### Decision 4: 模块 3 对因果图做 edge dropout，并诊断双曲半径

模块 3 继续使用模块 2 输出的因果图作为 HGCN adjacency。训练时可对因果图边做 `causal_edge_dropout`，但测试时必须使用完整 learned graph。模块 3 需要记录：

- `z_global` 在 Poincare Ball 中的半径分布。
- `z_tangent` 的均值、方差或范数。
- HGCN 输入/输出的图读出稳定性。

理由：当前可视化中 `z_global` 区分度弱，可能是双曲投影空间没有形成稳定层级结构，或少量强边导致训练记忆。edge dropout 可以迫使表示不依赖单一训练边。

替代方案：提高 HGCN 维度或层数。该方案更容易过拟合，不作为默认改进。

### Decision 5: 模块 4 增强 HPEC 多原型初始化与正则

模块 4 保留 HPEC energy 作为分类主损失。多原型默认开启，但需要更稳：

- 使用训练 fold 内 `z_global` warm-start 或 mini-batch 累积初始化 prototype，不能使用验证/测试标签。
- 增加 prototype diversity，避免同类多个 prototype collapse。
- 增加类间 margin，拉开不同类别 prototype。
- 限制 prototype 半径范围，避免 prototype 贴近边界造成 energy 不稳定。
- 保留 `L_mle`、`L_pcl`、`L_pal`，并给 ABIDE 默认更温和权重。

理由：ABIDE 类内异质性较强，多 prototype 有必要，但如果原型自由度过强会记忆训练样本。初始化和正则需要服务于泛化。

替代方案：退回单 prototype。该方案简单但无法表达 ASD/HC 内部多样性，只作为消融。

## Risks / Trade-offs

- [Risk] 新增正则过多导致训练难以收敛。  
  Mitigation：所有新增项提供独立权重，默认从小权重开始，并在日志中单独打印。

- [Risk] denoising auxiliary loss 需要额外 forward，增加训练时间。  
  Mitigation：默认可只对部分 batch 或使用轻量扰动；提供权重为 0 的关闭方式。

- [Risk] 因果图被过度稀疏化后分类信号不足。  
  Mitigation：保留 warmup 与阈值/稀疏权重可调，输出 graph mass 与 edge stability 诊断。

- [Risk] prototype warm-start 在前几个 epoch 表示不稳定。  
  Mitigation：允许延迟初始化或在 warm-start 后冻结若干步，再逐渐开放训练。

- [Risk] 单 fold ABIDE 指标波动较大。  
  Mitigation：默认脚本保留 `max_folds` 与 `iterations` 参数；变更验证至少跑 1 fold，最终参数建议再跑多 fold。

## Migration Plan

1. 增加参数入口，默认保持非 ABIDE 数据集旧行为不变。
2. 在 `test_abide_best_config.py` 中改为完整四模块默认配置，并填入 ABIDE 专用抗过拟合默认值。
3. 实现模块 1 去噪增强与可选 denoising loss。
4. 实现模块 2 稳定因果图约束和更保守的样本残差图默认。
5. 实现模块 3 edge dropout 与半径诊断。
6. 实现模块 4 prototype warm-start、diversity、margin 与半径约束。
7. 运行 ABIDE 1 fold 训练检查完整四模块可跑通，并记录指标、loss、可视化输出。

回滚方式：

- 将新增 loss 权重全部设为 0。
- 将 `module1_random_crop=0`、`module1_temporal_dropout=0`、`module1_roi_dropout=0`、`causal_edge_dropout=0`。
- 将 `test_abide_best_config.py` 参数改回当前已验证的 GCN fallback 对照配置。

## Open Questions

- ABIDE 本地数据是否有可解析站点或表型文件；当前文件名只显示 `unknown`，本变更不依赖站点信息。
- denoising auxiliary loss 默认是否每个 batch 都计算，还是按比例采样计算，需要通过训练时间与效果决定。
- prototype warm-start 使用首个 epoch 后初始化，还是使用前若干 batch 的 moving average，需要实现时根据现有训练流程选择。
