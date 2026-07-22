## ADDED Requirements

### Requirement: ABIDE 主路径不得绕开四模块框架

`S-DeCI` SHALL 支持 ABIDE 主实验在模块 1、模块 2、模块 3 和模块 4 全部启用时完成训练、验证和诊断。

#### Scenario: ABIDE 完整路径训练

- **GIVEN** 数据集为 `Abide`
- **AND** `seq_len == 120`
- **AND** `use_deci_module1 == 1`
- **AND** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 1`
- **WHEN** `S-DeCI.forward()` 执行
- **THEN** 模型 MUST 依次执行模块 1 去噪特征提取、模块 2 因果图学习、模块 3 HGCN 双曲 readout 和模块 4 HPEC energy 分类
- **AND** 训练流程 MUST 能完成至少一个 fold

#### Scenario: fallback 不作为默认主路径

- **GIVEN** 数据集为 `Abide`
- **WHEN** 构造默认测试配置
- **THEN** 模型配置 MUST NOT 默认关闭模块 2
- **AND** 模型配置 MUST NOT 默认关闭模块 3/4
- **AND** GCN fallback MUST 仅在用户显式关闭模块时使用

### Requirement: 模块 1 输出去噪后的节点表示

`S-DeCI` SHALL 允许模块 1 在保持 DeCI/Cycle 语义的同时输出去噪后的时序或节点特征，供模块 2 和后续模块使用。

#### Scenario: 缓存去噪输出

- **GIVEN** `use_deci_module1 == 1`
- **WHEN** 模块 1 完成 forward
- **THEN** 模型 MUST 能缓存模块 1 输出的去噪节点表示
- **AND** 该表示 MUST 保持 `[B, N, d_model]` 或可被模块 2 使用的等价形状

#### Scenario: 去噪 loss 可关闭

- **WHEN** `module1_denoise_loss_weight == 0`
- **THEN** 模块 1 denoising auxiliary loss MUST 不影响总 loss
- **AND** 模型 MUST 保持当前模块 1 行为兼容

### Requirement: 完整四模块 loss 可联合回传

`S-DeCI` SHALL 在完整四模块路径中把 HPEC 分类 loss、模块 1 去噪 loss、模块 2 因果辅助 loss、模块 3 正则和模块 4 prototype loss 合并为一次反向传播。

#### Scenario: 合并完整 loss

- **GIVEN** 四个模块均已启用
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 HPEC primary loss
- **AND** 当权重大于 0 时 MUST 包含模块 1 denoising loss
- **AND** 当权重大于 0 时 MUST 包含模块 2 reconstruction、DAG、L1 和 stability loss
- **AND** 当权重大于 0 时 MUST 包含模块 3 双曲正则 loss
- **AND** 当权重大于 0 时 MUST 包含模块 4 prototype auxiliary loss

#### Scenario: 不阻断分类梯度

- **GIVEN** 四个模块均已启用
- **WHEN** 总 loss 执行 `backward()`
- **THEN** HPEC 分类梯度 MUST 能通过模块 4、模块 3 和模块 2 的 adjacency 回传
- **AND** 默认配置 MUST NOT detach 模块 2 的因果图
