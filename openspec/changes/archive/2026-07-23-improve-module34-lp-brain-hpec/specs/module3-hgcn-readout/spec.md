## MODIFIED Requirements

### Requirement: 模块 3 正式路线使用 Poincare HGCN 与 mean_std readout

模块 3 SHALL 使用模块 2 产生的有向分类图执行 Poincare HGCN 图传播，并在原点切空间以 `mean_std` readout 生成图级双曲表示；当前正式实现不得要求 LP-Brain-HPEC Lorentz 路径存在。

#### Scenario: 默认模块 3 前向传播
- **GIVEN** 模块 3 已启用，节点特征为 `[B,N,D]`，分类图为 `[N,N]` 或 `[B,N,N]`
- **WHEN** 模块 3 执行 forward
- **THEN** 系统 MUST 保持 `A[parent,child]` 的有向语义执行图传播
- **AND** MUST 将节点双曲表示映射到原点切空间
- **AND** MUST 计算节点维度的均值和标准差，并生成形状为 `[B,H]` 的 `z_tangent`
- **AND** MUST 通过 Poincare 指数映射得到形状为 `[B,H]` 的 `z_global`

#### Scenario: LP 路径退出正式能力
- **WHEN** 用户查看模型架构选择和模块 3 layer
- **THEN** 系统 MUST NOT 把 `lp_brain_hpec` 暴露为可执行正式路径
- **AND** MUST NOT 依赖已退役的 Lorentz lifting、Lorentz tangent readout 或 LP 专用 layer
- **AND** 历史 LP 公式和失败原因 MAY 保留在实验文档中，但 MUST 标注为负向实验记录

### Requirement: 模块 3 不接受坐标级 FC 注入

模块 3 SHALL 保持双曲坐标由节点特征和图传播产生，样本 FC 只能用于构造分类图或并行欧氏证据，不得直接平移 `z_tangent` 或 `z_global`。

#### Scenario: 融合 FC 与双曲证据
- **GIVEN** 样本 FC 与模块 3 双曲表示均可用
- **WHEN** 模型组合两种信息
- **THEN** FC MAY 在图构造阶段连续门控 adjacency
- **AND** FC MAY 在独立欧氏分支生成分类证据
- **AND** MUST NOT 将原始 FC embedding 直接加到双曲切向量后重新投影
