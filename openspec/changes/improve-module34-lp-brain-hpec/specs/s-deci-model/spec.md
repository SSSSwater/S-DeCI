## ADDED Requirements

### Requirement: S-DeCI 支持 module34_arch 路径选择

`S-DeCI` SHALL 支持通过 `module34_arch` 选择现有 HGCN/HPEC 路径或新增 LP-Brain-HPEC 路径。

#### Scenario: 选择 LP-Brain-HPEC 路径
- **GIVEN** `use_hyperbolic_modules34 == 1`
- **AND** `module34_arch == "lp_brain_hpec"`
- **WHEN** `S-DeCI` 初始化
- **THEN** 模型 MUST 初始化 Lorentz lifting、Directed Lorentz GCN、Lorentz readout、bridge、MAC/HBR 和 HPEC energy 组件
- **AND** 模型 MUST 保持模块 1 和模块 2 的现有输入输出语义

#### Scenario: 回退现有 HGCN/HPEC 路径
- **GIVEN** `module34_arch == "hgcn_hpec"`
- **WHEN** `S-DeCI` 初始化
- **THEN** 模型 MUST 使用现有 HGCN readout 与 HPEC energy 路径
- **AND** 现有默认训练行为 MUST 不被 LP-Brain-HPEC 破坏

#### Scenario: 模块 3/4 关闭时不初始化 LP 路径
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** `S-DeCI` 初始化
- **THEN** 模型 MUST NOT 初始化 LP-Brain-HPEC 模块
- **AND** 模型 MUST 使用 GCN fallback 路径完成分类

### Requirement: S-DeCI 将模块 2 有向图传入 LP-Brain-HPEC

`S-DeCI` SHALL 将模块 2 输出的 `A_effective`、`A_lag.mean(dim=0)` 或样本相关矩阵作为 LP-Brain-HPEC 的有向 adjacency，并保持 `A[parent, child]` 方向语义。

#### Scenario: 模块 2 启用时使用 learned causal graph
- **GIVEN** `use_causal_module2 == 1`
- **AND** `module34_arch == "lp_brain_hpec"`
- **WHEN** 模块 3/4 需要 adjacency
- **THEN** 系统 MUST 使用模块 2 解析后的 learned/effective causal graph
- **AND** 系统 MUST NOT 默认对该图做对称化

#### Scenario: 模块 2 关闭时使用样本相关矩阵
- **GIVEN** `use_causal_module2 == 0`
- **AND** `module34_arch == "lp_brain_hpec"`
- **WHEN** 模块 3/4 需要 adjacency
- **THEN** 系统 MUST 使用 batch 对应的 sample correlation matrix
- **AND** 若缺少 sample correlation matrix，系统 MUST 以清晰错误失败

### Requirement: S-DeCI 缓存 LP-Brain-HPEC 中间量

`S-DeCI` SHALL 缓存 LP-Brain-HPEC 的关键中间量，用于可视化、TensorBoard 和性能诊断。

#### Scenario: 缓存几何和能量中间量
- **WHEN** LP-Brain-HPEC 路径完成 forward
- **THEN** 系统 MUST 缓存 Lorentz 节点表示、Lorentz graph embedding、Poincare bridge embedding、MAC 后 embedding、HPEC energy matrix 和 prototype assignment
- **AND** 缓存 MUST 不改变 `S-DeCI.forward()` 的主返回值

#### Scenario: 可视化不泄漏测试标签
- **WHEN** 测试集可视化 LP-Brain-HPEC 中间量
- **THEN** 测试 label MUST NOT 作为模型 forward 输入
- **AND** label 只能用于训练后图例、t-SNE 着色或指标计算

### Requirement: S-DeCI 对 LP-Brain-HPEC 提供中文注释和文档

`S-DeCI` SHALL 对新增 LP-Brain-HPEC 路径提供简洁中文注释，并新增项目实现说明文档。

#### Scenario: 代码注释说明数据流
- **WHEN** 开发者查看新增模块和 `models/S_DeCI.py`
- **THEN** 关键逻辑 MUST 有中文注释说明 Lorentz lifting、有向入/出边聚合、bridge、MAC/HBR 和 HPEC energy
- **AND** 必要英文关键词 MAY 保留

#### Scenario: 新增说明文档
- **WHEN** 变更实现完成
- **THEN** 项目 MUST 新增中文说明文档
- **AND** 文档 MUST 说明 `修改方案.md` 中哪些内容被采纳、哪些内容被项目化适配
- **AND** 文档 MUST 不覆盖原始 `docs/新模块设计.md`

