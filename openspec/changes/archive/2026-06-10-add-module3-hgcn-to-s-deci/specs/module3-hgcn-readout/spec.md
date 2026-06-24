## ADDED Requirements

### Requirement: 模块 3 提供 HGCN 双曲 readout 组件

系统 SHALL 提供模块 3 HGCN 双曲 readout 组件，用于将模块 1 的 Cycle/seasonal feature 和模块 2 的因果邻接矩阵转换为全脑双曲中心点 `z_global`。

#### Scenario: 创建可复用 HGCN 层

- **GIVEN** 项目需要在 `S-DeCI` 中接入模块 3
- **WHEN** 开发者查看 `layers/`
- **THEN** MUST 能找到模块 3 使用的 HGCN/双曲 readout 层文件
- **AND** 该层文件 MUST NOT 依赖 `reference/` 目录作为运行时 import 路径

#### Scenario: 模块 3 输入输出形状

- **GIVEN** Cycle feature `C` 的形状为 `[B, N, d_model]`
- **AND** 因果邻接矩阵 `A_learned` 的形状为 `[N, N]`
- **WHEN** 模块 3 执行 forward
- **THEN** MUST 输出节点级双曲表示 `H_gcn`
- **AND** `H_gcn` 的形状 MUST 为 `[B, N, hgcn_hidden_dim]`
- **AND** MUST 输出全脑中心点 `z_global`
- **AND** `z_global` 的形状 MUST 为 `[B, hgcn_hidden_dim]`

### Requirement: 模块 3 按新模块设计执行双曲映射与图传播

模块 3 SHALL 按 `docs/新模块设计.md` 的模块 3 描述实现 Backclip、Poincare Ball 投影、HGCN 图传播和 Fréchet readout。

#### Scenario: 执行 Backclip 与 Poincare 投影

- **GIVEN** 模块 3 接收 Cycle feature `C`
- **WHEN** 模块 3 开始双曲图传播
- **THEN** MUST 先执行 Backclip 或等价限幅逻辑得到 `C_clipped`
- **AND** MUST 使用 Poincare Ball 的 `expmap0` 或等价接口将 `C_clipped` 投影到双曲空间

#### Scenario: 使用因果图执行 HGCN

- **GIVEN** 模块 3 已得到双曲节点表示
- **AND** 模块 2 已输出 `A_learned`
- **WHEN** 模块 3 执行图传播
- **THEN** MUST 使用 `A_learned` 作为图卷积拓扑
- **AND** MUST 使用 Mobius 运算或设计中声明的可微切空间近似完成邻居信息聚合

#### Scenario: 读取全脑 Fréchet 中心

- **GIVEN** HGCN 输出节点级双曲表示 `H_gcn`
- **WHEN** 模块 3 执行 readout
- **THEN** MUST 使用可微 Fréchet mean 或设计中声明的可微切空间均值 readout 得到 `z_global`
- **AND** `z_global` MUST 位于 Poincare Ball 或对应切空间映射可投影回 Poincare Ball

### Requirement: 模块 3 双曲中心维度可配置

系统 SHALL 将模块 3 输出的双曲中心维度设为可配置超参数，并默认使用 `128`。

#### Scenario: 使用默认双曲中心维度

- **GIVEN** 用户未显式指定模块 3 hidden/readout 维度
- **WHEN** 初始化 `S-DeCI` 模块 3
- **THEN** `hgcn_hidden_dim` MUST 默认为 `128`
- **AND** `z_global` 的最后一维 MUST 为 `128`

#### Scenario: 使用自定义双曲中心维度

- **GIVEN** 用户通过配置指定 `hgcn_hidden_dim`
- **WHEN** 初始化 `S-DeCI` 模块 3
- **THEN** HGCN 输出和 `z_global` MUST 使用该维度

### Requirement: 模块 3 当前阶段不实现模块 4

系统 SHALL 在当前阶段直接使用 `z_global` 作为分类依据，不得实现模块 4/HPEC 原型角度损失或能量分类器。

#### Scenario: 不创建模块 4 分类路径

- **GIVEN** `S-DeCI` 启用模块 3
- **WHEN** 模型执行 forward 并产生分类输出
- **THEN** 分类输出 MUST 来自 `z_global` 对应的分类头
- **AND** MUST NOT 使用 HPEC 原型角度损失
- **AND** MUST NOT 使用模块 4 能量分类器

### Requirement: 模块 3 中间量可缓存

模块 3 SHALL 缓存关键中间量，供训练诊断和可视化使用。

#### Scenario: 缓存双曲图传播中间量

- **GIVEN** `S-DeCI` 启用模块 3
- **WHEN** 模型完成一次 forward
- **THEN** MUST 能读取 `C_clipped`、`H0` 或 Poincare 投影结果、`H_gcn`、`z_global`
- **AND** 缓存中间量 MUST 不改变 `S-DeCI.forward()` 的主返回值

