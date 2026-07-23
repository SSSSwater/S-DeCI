## ADDED Requirements

### Requirement: 有向多阶因果可达性编码

系统 SHALL 支持以模块 2 输出的有向 `A_cls[parent, child]` 构造小 hop 可达性编码，并以残差方式增强模块 3 输入特征。

#### Scenario: 构造前向因果可达性
- **GIVEN** `use_multi_hop_causal_encoding == 1`
- **AND** `A_cls` 的形状为 `[N, N]` 或 `[B, N, N]`，语义为 parent 到 child
- **WHEN** 模块 3 准备节点输入
- **THEN** 系统 MUST 从 `abs(A_cls)` 构造行归一化有向转移矩阵 `P`
- **AND** MUST 计算从 parent 到 child 的 1 至 `causal_reachability_hops` 阶可达性
- **AND** 每一阶的节点编码 MUST 使用转置传播，使 child 聚合其 causal parent 的特征
- **AND** 系统 MUST 不修改模块 2 的 `A_cls` 数值或方向语义

#### Scenario: 残差注入 HGCN 输入
- **GIVEN** 节点特征 `C` 的形状为 `[B, N, D]`
- **WHEN** 多阶编码完成
- **THEN** 系统 MUST 以可学习或可配置 hop gate 融合各阶编码
- **AND** MUST 通过 `causal_reachability_scale` 以残差形式得到同形状 `C'`
- **AND** 原有 HGCN MUST 接收 `C'`，并保持输出形状和后续 HPEC 接口兼容

#### Scenario: 关闭多阶编码
- **GIVEN** `use_multi_hop_causal_encoding == 0`
- **WHEN** 模块 3 执行 forward
- **THEN** 系统 MUST 将原始 `C` 直接送入既有 HGCN 路径
- **AND** MUST 不计算矩阵幂或新增 projection

### Requirement: 多阶因果编码诊断

系统 SHALL 记录多阶因果编码的规模与 gate，避免其成为未观察的过平滑来源。

#### Scenario: 记录多阶诊断
- **GIVEN** 多阶因果编码已启用
- **WHEN** 训练或验证执行 forward
- **THEN** 系统 MUST 记录每个 hop 的 gate、编码范数和最终残差范数
- **AND** MUST 将这些诊断写入 TensorBoard
