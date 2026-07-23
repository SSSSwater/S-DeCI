## MODIFIED Requirements

### Requirement: 模块 3 可选接入多阶因果输入编码

模块 3 SHALL 在显式启用时接收由 `A_cls[parent,child]` 构造的小 hop 有向可达性编码，同时保持 Poincare HGCN 和默认 `mean_std` readout 不变。

#### Scenario: 使用多阶增强节点输入
- **GIVEN** `use_multi_hop_causal_encoding == 1`
- **WHEN** 模块 3 执行 HGCN
- **THEN** child MUST 聚合 1 至 `causal_reachability_hops` 阶 causal parent 特征
- **AND** 各 hop MUST 由可学习 gate 融合
- **AND** 增强量 MUST 通过 `causal_reachability_scale` 残差加入原节点特征
- **AND** 系统 MUST NOT 改写 `A_cls` 的数值或方向

#### Scenario: 默认关闭多阶编码
- **GIVEN** `use_multi_hop_causal_encoding == 0`
- **WHEN** 模块 3 执行 forward
- **THEN** 系统 MUST 直接使用模块 1 节点特征
- **AND** MUST 不计算额外矩阵幂或 hop projection

### Requirement: 模块 3 只执行标准视图

当前正式模块 3 SHALL 每个 batch 只执行标准节点特征视图，不提供已退役的因果显著性遮挡互补前向。

#### Scenario: 训练与推理路径一致
- **WHEN** 模块 3 在训练、验证或测试阶段执行
- **THEN** 系统 MUST 使用同一标准 HGCN/readout 路径
- **AND** 训练阶段 MUST NOT 因互补学习再次调用模块 3
