## Purpose

定义由模块 2 时序因果图导出的轻量多阶有向可达性编码：在不改写因果图方向和边权的前提下，以小 hop 残差增强模块 3 对多步信息流的建模，并保留可独立关闭的消融入口。

## Requirements

### Requirement: 前向因果可达性残差编码

系统 SHALL 在显式启用时从 `A_cls[parent, child]` 构造最多 L 阶的有向转移，并残差增强 HGCN 输入。

#### Scenario: child 聚合 parent 的多阶信息
- **GIVEN** `use_multi_hop_causal_encoding == 1`
- **WHEN** 模块 3 准备节点特征 `C[B,N,D]`
- **THEN** 系统 MUST 从 `abs(A_cls)` 构造行归一化 `P`
- **AND** MUST 以 `(P^l)^T` 聚合 parent 到 child 的第 l 阶特征
- **AND** MUST 用可学习 hop gate 和 `causal_reachability_scale` 形成残差输入
- **AND** MUST 不改写 `A_cls` 或其方向语义

#### Scenario: 关闭编码保持现有 HGCN
- **GIVEN** `use_multi_hop_causal_encoding == 0`
- **WHEN** 模块 3 执行 forward
- **THEN** 系统 MUST 不计算矩阵幂或新增编码 projection
- **AND** HGCN 输入输出接口 MUST 保持兼容
