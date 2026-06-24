## ADDED Requirements

### Requirement: HGCN/HPEC 路径服从联合开关

模块 3 HGCN readout SHALL 仅在 `S-DeCI` 的模块 3/4 联合开关启用时参与 forward 和训练。

#### Scenario: 联合开关启用时执行 HGCN readout
- **GIVEN** `use_hyperbolic_modules34 == 1`
- **WHEN** `S-DeCI.forward()` 已获得节点特征和 adjacency
- **THEN** 模块 3 MUST 执行 HGCN readout
- **AND** 模块 4 MUST 使用模块 3 输出的 `z_global` 或等价双曲中心表示进行 HPEC energy/prototype 分类

#### Scenario: 联合开关禁用时跳过 HGCN readout
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** `S-DeCI.forward()` 已获得节点特征和 adjacency
- **THEN** 模型 MUST NOT 调用模块 3 HGCN readout
- **AND** 模型 MUST NOT 生成 HPEC energy 或 prototype loss
- **AND** 模型 MUST 将节点特征和 adjacency 交给 GCN fallback 路径

### Requirement: 图路径统一使用当前节点特征和 adjacency

系统 SHALL 在 HGCN/HPEC 路径和 GCN fallback 路径中统一使用当前模块开关产生的节点特征和 adjacency。

#### Scenario: 模块 1 关闭且 HGCN/HPEC 启用
- **GIVEN** `use_deci_module1 == 0`
- **AND** `use_hyperbolic_modules34 == 1`
- **WHEN** 模块 3 执行 HGCN readout
- **THEN** 模块 3 MUST 使用 raw projected feature 作为节点特征
- **AND** 模块 3 MUST 使用模块 2 因果矩阵或样本相关矩阵作为 adjacency

#### Scenario: 模块 1 关闭且 GCN fallback 启用
- **GIVEN** `use_deci_module1 == 0`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** GCN fallback 执行图学习
- **THEN** GCN fallback MUST 使用 raw projected feature 作为节点特征
- **AND** GCN fallback MUST 使用模块 2 因果矩阵或样本相关矩阵作为 adjacency

#### Scenario: 模块 2 关闭时 adjacency 来源保持样本相关矩阵
- **GIVEN** `use_causal_module2 == 0`
- **WHEN** HGCN/HPEC 路径或 GCN fallback 路径需要 adjacency
- **THEN** 系统 MUST 使用 batch 中的 sample correlation matrix
- **AND** 系统 MUST NOT 用空矩阵、单位矩阵或随机矩阵静默替代缺失的 correlation matrix
