## MODIFIED Requirements

### Requirement: S-DeCI 使用单一正式模块 3/4 几何路径

`S-DeCI` SHALL 将 `hgcn_hpec` 作为当前唯一正式模块 3/4 路径，并保留关闭模块 3/4 后的 GCN fallback 作为消融对照。

#### Scenario: 初始化默认四模块路线
- **GIVEN** `use_hgcn_module3 == 1` 且 `use_hpec_module4 == 1`
- **WHEN** `S-DeCI` 初始化
- **THEN** 系统 MUST 创建 Poincare HGCN 模块和 HPEC 多原型模块
- **AND** MUST NOT 要求 `module34_arch` 才能选择默认路线
- **AND** MUST NOT 初始化已退役的 LP-Brain-HPEC 模块

#### Scenario: 模块 3/4 消融
- **GIVEN** 模块 3/4 被关闭
- **WHEN** 模型执行分类
- **THEN** 系统 MUST 使用现有 GCN fallback
- **AND** MUST 保持模块 1/2 与分类图输入接口兼容

### Requirement: S-DeCI 保持双视角证据融合

`S-DeCI` SHALL 将模块 3/4 的 HPEC 双曲证据与并行欧氏局部结构证据在 logits 层融合，且分类损失能够反向传播到模块 3、模块 4以及分类图依赖的模块 2 参数。

#### Scenario: 四模块联合训练
- **GIVEN** 模块 1 至模块 4 全部启用
- **WHEN** 系统计算最终 logits 与分类 loss
- **THEN** 最终 logits MUST 包含 HPEC energy 产生的双曲证据
- **AND** MUST 包含基于同一节点特征和分类图产生的欧氏局部结构证据
- **AND** 分类 loss MUST 能沿双曲路径更新模块 3/4
- **AND** 当分类图保持可微时，分类 loss MUST 能更新模块 2 图参数
