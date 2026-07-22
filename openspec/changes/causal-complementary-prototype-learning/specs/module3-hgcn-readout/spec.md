## ADDED Requirements

### Requirement: 模块 3 支持共享互补视图传播

模块 3 SHALL 能够在不新建第二套参数的情况下，对标准节点特征和遮挡节点特征分别执行双曲图传播与 readout。

#### Scenario: 共享参数执行两个节点特征视图
- **GIVEN** 标准与互补节点特征均为 `[B, N, D]`
- **AND** 两个视图共享同一 `A_cls`
- **WHEN** `S-DeCI` 在训练期调用模块 3 两次
- **THEN** 两次调用 MUST 使用同一个模块 3 参数对象
- **AND** 两次调用 MUST 输出兼容的 `H_gcn`、`z_global` 和 `z_tangent`
- **AND** 推理期 MUST 只调用标准视图

### Requirement: 模块 3 接入多阶因果输入编码

模块 3 SHALL 在配置启用后使用有向多阶因果可达性编码后的节点特征，同时保持其现有 Poincare HGCN 路径和 `mean_std` 默认 readout 不变。

#### Scenario: 使用增强后的节点输入
- **GIVEN** `use_multi_hop_causal_encoding == 1`
- **WHEN** 模块 3 执行 HGCN
- **THEN** HGCN MUST 接收多阶编码增强后的节点特征
- **AND** 默认 `hgcn_readout_mode == "mean_std"` MUST 继续可用
