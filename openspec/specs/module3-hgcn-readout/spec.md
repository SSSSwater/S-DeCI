## Purpose

定义 `S-DeCI` 模块 3 的 HGCN 双曲 readout 能力：将模块 1 的 Cycle/seasonal feature `C` 与图结构结合，得到节点级双曲表示 `H_gcn` 和全脑中心点 `z_global`。图结构可以来自模块 2 学到的因果邻接矩阵 `A_learned`，也可以在模块 2 关闭时来自样本级相关系数矩阵。模块 3 输出可供线性分类回退路径或模块 4 HPEC 原型能量分类使用。
## Requirements
### Requirement: 模块 3 提供 HGCN 双曲 readout 组件

系统 SHALL 提供模块 3 HGCN 双曲 readout 组件，用于将模块 1 的 Cycle/seasonal feature 和模块 2 的因果邻接矩阵转换为全脑双曲中心点 `z_global`。

#### Scenario: 创建可复用 HGCN 层

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

### Requirement: 模块 3 按设计执行双曲映射与图传播

模块 3 SHALL 实现 Backclip、Poincare Ball 投影、HGCN 图传播和可微 readout。

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

#### Scenario: 读取全脑中心

- **GIVEN** HGCN 输出节点级双曲表示 `H_gcn`
- **WHEN** 模块 3 执行 readout
- **THEN** MUST 使用可微 Fréchet mean 或可微切空间均值 readout 得到 `z_global`
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

### Requirement: 模块 3 支持模块 4 HPEC 接入

模块 3 SHALL 向模块 4 提供稳定的 `z_global` 和 `logmap0(z_global)` 缓存，使 HPEC energy 分类可以复用模块 3 的双曲 readout。

#### Scenario: 暴露 z_global

- **GIVEN** `S-DeCI` 启用模块 3 和模块 4
- **WHEN** 模块 3 完成 HGCN readout
- **THEN** 模块 3 MUST 输出形状为 `[B, hgcn_hidden_dim]` 的 `z_global`
- **AND** `z_global` MUST 位于 Poincare Ball 或可被投影回 Poincare Ball
- **AND** 模块 4 MUST 能直接读取该表示作为默认输入

#### Scenario: 暴露切空间表示

- **WHEN** 模块 3 缓存诊断量
- **THEN** 模块 3 MUST 缓存 `logmap0(z_global)`
- **AND** 该缓存 MUST 可用于线性分类回退、t-SNE 可视化和 HPEC 调试对照

#### Scenario: HPEC loss 经 HGCN 回传

- **GIVEN** `Loss_HPEC` 已由模块 4 计算
- **WHEN** 训练流程执行一次联合 `backward()`
- **THEN** HPEC 分类梯度 MUST 能经过 `z_global`、`H_gcn` 和 `A_learned` 回传
- **AND** 系统 MUST NOT 在模块 3 与模块 2 之间新增阻断该梯度的默认逻辑

### Requirement: 模块 3 中间量可缓存

模块 3 SHALL 缓存关键中间量，供训练诊断、heatmap 可视化和 t-SNE 可视化使用。

#### Scenario: 缓存双曲图传播中间量

- **GIVEN** `S-DeCI` 启用模块 3
- **WHEN** 模型完成一次 forward
- **THEN** MUST 能读取 `C_clipped`、`H0` 或 Poincare 投影结果、`H_gcn`、`z_global` 和 `logmap0(z_global)`
- **AND** 缓存中间量 MUST 不改变 `S-DeCI.forward()` 的主返回值

#### Scenario: 提供 HGCN 与 HPEC 对照中间量

- **WHEN** 用户显式开启 S-DeCI 中间量可视化
- **THEN** 系统 MUST 能读取 `C_clipped`、`H0` 或 Poincare 投影结果、`H_gcn`、`z_global` 和 `logmap0(z_global)`
- **AND** 这些中间量 MUST 能与模块 4 的 prototype、angle 和 energy 在同一批次诊断中对照保存

### Requirement: 模块 3 支持 batch 级 adjacency

模块 3 SHALL 同时支持全局 adjacency `[N, N]` 和样本级 adjacency `[B, N, N]`。

#### Scenario: 使用全局 adjacency

- **GIVEN** `adjacency` 的形状为 `[N, N]`
- **WHEN** 模块 3 执行 HGCN forward
- **THEN** 模块 3 MUST 将同一 adjacency 用于 batch 内所有样本
- **AND** 该路径 MUST 与当前模块 2 `A_learned` 输入兼容

#### Scenario: 使用 batch adjacency

- **GIVEN** `cycle_features` 的形状为 `[B, N, D]`
- **AND** `adjacency` 的形状为 `[B, N, N]`
- **WHEN** 模块 3 执行 HGCN forward
- **THEN** 模块 3 MUST 对每个样本使用其对应的 adjacency
- **AND** 输出 `H_gcn` 的形状 MUST 保持 `[B, N, hgcn_hidden_dim]`
- **AND** 输出 `z_global` 的形状 MUST 保持 `[B, hgcn_hidden_dim]`

### Requirement: 模块 3 规范化样本相关矩阵图

模块 3 SHALL 在使用 sample correlation adjacency 时执行数值清理和图归一化。

#### Scenario: 清理相关矩阵

- **GIVEN** 输入 adjacency 来自样本相关系数矩阵
- **WHEN** 模块 3 归一化 adjacency
- **THEN** 模块 3 MUST 将 NaN 或 Inf 替换为有限值
- **AND** MUST 支持按配置处理负相关，至少包括 `abs`、`positive` 和 `raw`
- **AND** 默认模式 MUST 为 `abs`

#### Scenario: 加入 self-loop 并归一化

- **GIVEN** 模块 3 配置 `hgcn_add_self_loop == 1`
- **WHEN** adjacency 进入图传播
- **THEN** 模块 3 MUST 对 `[N, N]` 和 `[B, N, N]` 两种 adjacency 都支持添加 self-loop
- **AND** MUST 对两种 adjacency 都支持 `row`、`sym` 和 `none` 归一化方式

#### Scenario: 拒绝错误形状

- **GIVEN** adjacency 既不是 `[N, N]` 也不是 `[B, N, N]`
- **WHEN** 模块 3 forward 被调用
- **THEN** 模块 3 MUST 以清晰错误失败

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

### Requirement: ?? 3 ??????? readout ?? `z_global`

?? 3 SHALL ?????? ROI ?????????????????HGCN ??????? `z_global` ? SHOULD ??????????????? mean/max ?????????????????

#### Scenario: ????? readout

- **GIVEN** HGCN ?????????? `h_gcn`
- **WHEN** ?? 3 ????????? `z_global`
- **THEN** ?? SHOULD ? `logmap0(h_gcn)` ???????? ROI ???? attention weight
- **AND** `z_global` SHOULD ? attention-weighted mean ?????????????? Poincare Ball
- **AND** ??? MUST ?? `z_global` ? `z_tangent`???? 4 HPEC ? t-SNE ??

#### Scenario: ???????

- **WHEN** ?? 3 ?? forward
- **THEN** ?? SHOULD ????? `[B, N]` ? `node_attention`
- **AND** ?????? SHOULD ?????????????????? label ???? ROI ????

### Requirement: ?? 3 ????????? readout

?? 3 SHOULD ??????????? readout ?? `z_global`?????????????????????????????

#### Scenario: ?? mean/std/max ??

- **GIVEN** HGCN ?????????? `h_gcn`
- **WHEN** ?? 3 ????????? `z_global`
- **THEN** ?? SHOULD ? `logmap0(h_gcn)` ??????????? `mean`?`std` ? `max` ??
- **AND** ?? SHOULD ???? MLP ?????????????????
- **AND** ??????? SHOULD ???? Poincare Ball ?????????????????????????

#### Scenario: ???????????

- **WHEN** ?? 3 ?? forward
- **THEN** ?? SHOULD ???????????????? `node_attention` ????????
- **AND** ??? SHOULD ?????????????????????????????

### Requirement: ?? 3 ?? MDD ??????? readout

?? 3 SHOULD ??? AAL116 ??? MDD ???????????????? `z_global` ?????????????????????????????????

#### Scenario: AAL116 ?????

- **GIVEN** ????? AAL116 ? 116 ? ROI
- **WHEN** `use_brain_network_prior == 1`
- **THEN** ?? 3 SHOULD ????????? mask
- **AND** ?? SHOULD ???? `DMN`?`fronto-limbic/affective`?`cognitive control/frontoparietal`?`salience`?`subcortical-thalamic/striatal`?`sensorimotor`?`visual` ? `cerebellum`
- **AND** `DMN`?`fronto-limbic/affective`?`salience`?`subcortical-thalamic/striatal` SHOULD ??????????

#### Scenario: ????????

- **GIVEN** HGCN ?????????? `h_gcn`
- **WHEN** ?? 3 ?? `z_global`
- **THEN** ?? SHOULD ? `logmap0(h_gcn)` ????????????????
- **AND** ?? SHOULD ??????????????????? readout fusion
- **AND** ?? MUST ???? mean/std/max ???????????????????

#### Scenario: ???????????

- **WHEN** ????????
- **THEN** ?? SHOULD ????? summary ???? attention/weight
- **AND** ????? SHOULD ????????????? MDD ????????????

