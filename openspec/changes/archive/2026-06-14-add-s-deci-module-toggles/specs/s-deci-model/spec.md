## ADDED Requirements

### Requirement: S-DeCI 模块 1 可禁用

`S-DeCI` SHALL 支持通过配置禁用模块 1 的 DeCI/Cycle 分解，并在禁用后直接从原始时间序列生成节点特征。

#### Scenario: 模块 1 启用时保持现有 Cycle 路径
- **GIVEN** `use_deci_module1 == 1`
- **WHEN** `S-DeCI.forward()` 接收形状为 `[B, T, N]` 的输入
- **THEN** 模型 MUST 执行现有 DeCI block 流程
- **AND** 模型 MUST 继续以 Cycle/seasonal feature 作为模块 2、模块 3/4 或 fallback 分类路径的节点特征来源

#### Scenario: 模块 1 禁用时使用 raw projection
- **GIVEN** `use_deci_module1 == 0`
- **WHEN** `S-DeCI.forward()` 接收形状为 `[B, T, N]` 的输入
- **THEN** 模型 MUST NOT 调用 DeCI block
- **AND** 模型 MUST NOT 提取高频、trend、seasonal 或 residual
- **AND** 模型 MUST 将原始时间序列转为 `[B, N, T]` 后投影为 `[B, N, d_model]`
- **AND** 投影后的 raw feature MUST 能作为模块 2、模块 3/4 或 GCN fallback 的节点特征输入

#### Scenario: 模块 1 禁用时可视化 raw feature
- **GIVEN** `use_deci_module1 == 0`
- **AND** 显式启用中间量可视化
- **WHEN** 模型完成 forward
- **THEN** 模型 MUST 缓存 raw projected feature
- **AND** 可视化标题或文件名 MUST 表明该特征不是 Cycle/seasonal feature

### Requirement: S-DeCI 模块开关组合约束

`S-DeCI` SHALL 对模块 1、模块 2、模块 3/4 的开关组合进行归一化与校验，使训练路径明确且可复现。

#### Scenario: 模块 3 和模块 4 联合启用
- **GIVEN** `use_hyperbolic_modules34 == 1`
- **WHEN** 模型初始化
- **THEN** 模型 MUST 使用 HGCN readout 与 HPEC energy/prototype 分类路径
- **AND** 若实现仍保留 `use_hgcn_module3` 和 `use_hpec_module4`，二者 MUST 被设置为一致的启用状态

#### Scenario: 模块 3 和模块 4 联合禁用
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** 模型初始化
- **THEN** 模型 MUST NOT 初始化 HGCN readout
- **AND** 模型 MUST NOT 初始化 HPEC energy/prototype 分类器
- **AND** 模型 MUST 初始化 GCN fallback 分类路径

#### Scenario: 拒绝不一致的旧参数组合
- **GIVEN** 用户同时传入 `use_hyperbolic_modules34`、`use_hgcn_module3` 或 `use_hpec_module4`
- **WHEN** 参数组合表达出 HGCN 与 HPEC 不一致的状态
- **THEN** 系统 MUST 归一化为 `use_hyperbolic_modules34` 的值或清晰失败
- **AND** 错误信息 MUST 说明模块 3 与模块 4 在本设计中需要联合启用或联合禁用

### Requirement: S-DeCI 根据模块开关选择 loss

`S-DeCI` 训练流程 SHALL 根据当前模块开关组合选择分类 loss 与 auxiliary loss。

#### Scenario: 全模块启用时使用 HPEC 联合 loss
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 1`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 HPEC primary loss
- **AND** 总 loss MUST 包含模块 2 reconstruction、DAG 和 L1 auxiliary loss
- **AND** 若多 prototype loss 权重大于 0，总 loss MUST 包含对应 prototype auxiliary loss

#### Scenario: GCN fallback 且模块 2 启用时使用分类 loss 加因果辅助项
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 GCN fallback 分类 loss
- **AND** 总 loss MUST 包含模块 2 reconstruction、DAG 和 L1 auxiliary loss
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss

#### Scenario: GCN fallback 且模块 2 禁用时只使用分类 loss
- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 GCN fallback 分类 loss
- **AND** 总 loss MUST NOT 包含模块 2 auxiliary loss
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss
