## Purpose

定义 `S-DeCI` 模块 3 的 HGCN 双曲 readout 能力：将模块 1 的 Cycle/seasonal feature `C` 与图结构结合，得到节点级双曲表示 `H_gcn` 和全脑中心点 `z_global`。图结构可以来自模块 2 学到的因果邻接矩阵 `A_learned`，也可以在模块 2 关闭时来自样本级相关系数矩阵。模块 3 输出可供线性分类回退路径或模块 4 HPEC 原型能量分类使用。
## Requirements
### Requirement: 模块 3 的 LP-Brain-HPEC 历史实验边界

模块 3 的 LP-Brain-HPEC Lorentz 有向 readout 已完成实验并退出当前代码。下列公式仅记录历史候选设计和失败诊断，不构成当前可执行 `module34_arch` 能力。默认主路线 MUST 使用 Poincare HGCN 和 `hgcn_readout_mode == "mean_std"`，因为完整测试中该路径更快、更稳定，且不会引入 Lorentz-to-Poincare bridge 的额外数值敏感性。

#### Scenario: Lorentz lifting 输入输出形状
- **GIVEN** 节点特征 `node_features` 的形状为 `[B, N, d_in]`
- **WHEN** `module34_arch == "lp_brain_hpec"` 且模块 3 执行 forward
- **THEN** 系统 MAY 先将节点特征投影到 Lorentz 切空间
- **AND** MAY 通过指数映射得到 `x_lorentz_0`
- **AND** 该投影过程 MUST 在文档中写成 LaTeX 公式，例如：

$$
u_{b,i}
=
W_{\mathrm{lift}}c_{b,i}
+b_{\mathrm{lift}},
\qquad
u\in\mathbb{R}^{B\times N\times D}.
$$

$$
\bar{u}_{b,i}
=
u_{b,i}
\cdot
\min
\left(
1,
\frac{R_{\mathrm{lor}}}{\|u_{b,i}\|_2+\epsilon}
\right).
$$

$$
x^{\mathrm{lor}}_{b,i}
=
\operatorname{Exp}^{\mathbb{L}}_{o}
\left(
\bar{u}_{b,i}
\right)
\in
\mathbb{L}^{D}.
$$

- **AND** `x_lorentz_0` 的形状 SHOULD 为 `[B, N, D + 1]`
- **AND** Lorentz time-like 维度 SHOULD 位于最后一维的第 0 个位置
- **AND** 文档 MUST 说明该步骤来源于 Lorentz hyperboloid 模型中的切空间指数映射，设计原因是先把欧氏 ROI 特征转换到 Lorentz manifold，方便后续实验性有向双曲图卷积；同时 MUST 说明该路径不是默认主路线

#### Scenario: 使用有向因果图区分入边和出边
- **GIVEN** 模块 2 输出 adjacency，语义为 `A[parent, child]`
- **WHEN** Directed Lorentz GCN 聚合节点 `i`
- **THEN** 入边聚合 MUST 使用 `A[j, i]`
- **AND** 出边聚合 MUST 使用 `A[i, k]`
- **AND** 入边和出边 MUST 使用独立的 attention 或等价权重计算
- **AND** 系统 MUST 提供可学习或可配置的出边平衡系数
- **AND** 文档 MUST 写出入边和出边聚合公式，例如：

$$
m^{\mathrm{in}}_{b,i}
=
\sum_{j=1}^{N}
\alpha^{\mathrm{in}}_{b,j\rightarrow i}
A_{b,j,i}
\operatorname{Log}^{\mathbb{L}}_{x_{b,i}}
\left(
x_{b,j}
\right).
$$

$$
m^{\mathrm{out}}_{b,i}
=
\sum_{k=1}^{N}
\alpha^{\mathrm{out}}_{b,i\rightarrow k}
A_{b,i,k}
\operatorname{Log}^{\mathbb{L}}_{x_{b,i}}
\left(
x_{b,k}
\right).
$$

$$
m_{b,i}
=
m^{\mathrm{in}}_{b,i}
+
\lambda_{\mathrm{out}}m^{\mathrm{out}}_{b,i}.
$$

$$
x'_{b,i}
=
\operatorname{Exp}^{\mathbb{L}}_{x_{b,i}}
\left(
\sigma
\left(
W_m m_{b,i}
+b_m
\right)
\right).
$$

- **AND** 文档 MUST 说明入边表示“其他 ROI 对当前 ROI 的影响”，出边表示“当前 ROI 对其他 ROI 的影响”，该设计来源于有向图神经网络和 Lorentz HGCN 的图消息传递思想；设计原因是模块 2 的 $A[parent,child]$ 本身有方向，若只做无向聚合会损失因果方向语义

#### Scenario: 支持全局图和样本级图
- **GIVEN** adjacency 的形状为 `[N, N]` 或 `[B, N, N]`
- **WHEN** LP-Brain-HPEC 模块 3 执行 forward
- **THEN** 系统 MUST 对 `[N, N]` 图广播到 batch
- **AND** 系统 MUST 对 `[B, N, N]` 图逐样本使用对应 adjacency
- **AND** 输出节点 Lorentz 表示形状 MUST 为 `[B, N, D + 1]`

### Requirement: 模块 3 的 Lorentz tangent readout 历史记录

该 Lorentz 原点切空间 readout 只记录已完成 LP-Brain-HPEC 实验的计算契约，不属于当前可执行能力。默认主路线 SHALL 使用 Poincare HGCN 的 `mean_std` readout 得到 `z_global`。

#### Scenario: 切空间均值 readout
- **GIVEN** 最后一层 Lorentz 节点表示形状为 `[B, N, D + 1]`
- **WHEN** 模块 3 执行图级 readout
- **THEN** 系统 MUST 将节点表示映射到 Lorentz 原点切空间
- **AND** MUST 在节点维度执行 mean 或 attention-weighted mean
- **AND** MUST 将均值向量映射回 Lorentz manifold
- **AND** 输出 `graph_lorentz` 的形状 MUST 为 `[B, D + 1]`
- **AND** 文档 MUST 写出完整 readout 公式：

$$
u_{b,i}
=
\operatorname{Log}^{\mathbb{L}}_{o}
\left(
x^{(R)}_{b,i}
\right)
\in
\mathbb{R}^{D}.
$$

$$
\bar{u}_{b}
=
\frac{1}{N}
\sum_{i=1}^{N}
u_{b,i}.
$$

$$
g^{\mathrm{lor}}_{b}
=
\operatorname{Exp}^{\mathbb{L}}_{o}
\left(
\bar{u}_{b}
\right)
\in
\mathbb{L}^{D}.
$$

- **AND** 若使用 attention-weighted mean，文档 MUST 写出：

$$
\alpha_{b,i}
=
\frac{
\exp(a^\top u_{b,i})
}{
\sum_{j=1}^{N}\exp(a^\top u_{b,j})
},
\qquad
\bar{u}_{b}
=
\sum_{i=1}^{N}
\alpha_{b,i}u_{b,i}.
$$

- **AND** 文档 MUST 说明该 readout 来源于 Lorentz manifold 上“先映射到切空间再做欧氏聚合”的近似 Fréchet mean 思想；设计原因是直接在线性空间平均 Lorentz 坐标会破坏流形约束，因此必须写清 log map、聚合和 exp map 三步

#### Scenario: 缓存 Lorentz 诊断
- **WHEN** LP-Brain-HPEC 模块 3 完成 forward
- **THEN** 系统 MUST 缓存 Lorentz 节点表示、图级 Lorentz embedding 和 tangent readout 表示
- **AND** 诊断量 MUST 至少包含 Lorentz norm 或 manifold constraint error
- **AND** 诊断量 SHOULD 包含入边聚合强度和出边聚合强度
- **AND** Lorentz 约束误差 MUST 在文档中写成：

$$
\delta_{\mathbb{L}}
=
\frac{1}{BN}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\left|
\langle x_{b,i},x_{b,i}\rangle_{\mathbb{L}}
+
\frac{1}{c}
\right|.
$$

- **AND** 入边和出边强度诊断 SHOULD 写成：

$$
M_{\mathrm{in}}
=
\frac{1}{BN}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\|m^{\mathrm{in}}_{b,i}\|_2,
\qquad
M_{\mathrm{out}}
=
\frac{1}{BN}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\|m^{\mathrm{out}}_{b,i}\|_2.
$$

- **AND** 文档 MUST 说明这些量只用于 TensorBoard 或控制台诊断，不参与默认训练 loss；设计原因是判断 Lorentz 实验路径是否发生流形约束漂移或出入边消息失衡

### Requirement: 模块 3 的 Lorentz 几何 dtype 历史记录

`module34_geo_dtype` 只记录已退役 LP 实验的数值策略，当前 HGCN-HPEC 主路线 SHALL 不依赖该配置。

#### Scenario: 使用 auto 几何 dtype
- **GIVEN** `module34_geo_dtype == "auto"`
- **WHEN** 模块 3 执行 Lorentz 几何计算
- **THEN** 系统 MAY 在 Lorentz inner product、指数映射或对数映射中临时使用更高精度
- **AND** 输出 MUST 转回与主模型兼容的 dtype
- **AND** 训练流程 MUST 不因 dtype 不一致而失败

#### Scenario: 拒绝非法 dtype 配置
- **GIVEN** 用户传入不支持的 `module34_geo_dtype`
- **WHEN** 初始化 S-DeCI
- **THEN** 系统 MUST 以清晰错误失败

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
- **THEN** MUST 先将节点特征线性投影到切空间：

$$
u_{b,n}
=
W_c C_{b,n}+b_c,
\qquad
C\in\mathbb{R}^{B\times N\times D},
\quad
u\in\mathbb{R}^{B\times N\times H}.
$$

- **AND** MUST 执行 Backclip 或等价限幅逻辑：

$$
\operatorname{Backclip}_R(u_{b,n})
=
u_{b,n}
\cdot
\min
\left(
1,
\frac{R}{\|u_{b,n}\|_2+\epsilon}
\right).
$$

- **AND** MUST 使用 Poincare Ball 原点指数映射将限幅后的切向量投影到双曲空间：

$$
h^{(0)}_{b,n}
=
\exp_0^c(\bar{u}_{b,n}),
\qquad
\bar{u}_{b,n}=\operatorname{Backclip}_R(u_{b,n}),
$$

$$
\exp_0^c(v)
=
\tanh(\sqrt{c}\|v\|_2)
\frac{v}{\sqrt{c}\|v\|_2+\epsilon}.
$$

- **AND** MUST 使用投影算子保证输出位于 Poincare Ball 内：

$$
\operatorname{proj}_c(x)
=
\begin{cases}
x, & \sqrt{c}\|x\|_2<1-\epsilon,\\
\dfrac{1-\epsilon}{\sqrt{c}}\dfrac{x}{\|x\|_2+\epsilon}, & \sqrt{c}\|x\|_2\ge 1-\epsilon.
\end{cases}
$$

- **AND** 文档 MUST 说明 Backclip 的原因是限制指数映射前的切空间范数，避免样本过早贴近球边界导致梯度和距离计算不稳定
- **AND** 文档 MUST 说明该步骤来源于 HGCN / Poincare Ball 中“先在切空间做欧氏投影，再映射到双曲空间”的常用做法

#### Scenario: 使用因果图执行 HGCN

- **GIVEN** 模块 3 已得到双曲节点表示
- **AND** 模块 2 已输出 `A_learned`
- **WHEN** 模块 3 执行图传播
- **THEN** MUST 使用 `A_learned` 或最终分类图 $A_{\mathrm{cls},b}$ 作为图卷积拓扑，并保持方向语义 `A[parent, child]`
- **AND** MUST 将双曲节点映回原点切空间：

$$
v^{(r)}_{b,n}
=
\log_0^c(h^{(r)}_{b,n}),
$$

$$
\log_0^c(x)
=
\frac{\operatorname{arctanh}(\sqrt{c}\|x\|_2)}
{\sqrt{c}\|x\|_2+\epsilon}x.
$$

- **AND** MUST 使用 Mobius 线性变换或其切空间等价形式计算节点消息：

$$
\bar{h}^{(r)}_{b,i}
=
W_r\otimes_c h^{(r)}_{b,i},
\qquad
\bar{v}^{(r)}_{b,i}
=
\log_0^c(\bar{h}^{(r)}_{b,i})+b_r.
$$

- **AND** MUST 按 parent 指向 child 聚合：

$$
m^{(r)}_{b,j}
=
\sum_{i=1}^{N}
A_{\mathrm{cls},b,i,j}\bar{v}^{(r)}_{b,i}.
$$

- **AND** MUST 更新并映射回 Poincare Ball：

$$
\tilde{v}^{(r+1)}_{b,j}
=
\sigma(W_m m^{(r)}_{b,j}+b_m),
\qquad
\tilde{h}^{(r+1)}_{b,j}
=
\exp_0^c(\tilde{v}^{(r+1)}_{b,j}).
$$

- **AND** MAY 使用残差混合减少过度平滑：

$$
h^{(r+1)}_{b,j}
=
\operatorname{proj}_c
\left(
(1-\alpha_r)\tilde{h}^{(r+1)}_{b,j}
+\alpha_r h^{(r)}_{b,j}
\right).
$$

- **AND** 文档 MUST 说明图传播来源于 HGCN 的切空间聚合思想，设计原因是 Poincare Ball 中直接线性加权不稳定，而切空间聚合可以保留可微性和数值稳定性

#### Scenario: 读取全脑中心

- **GIVEN** HGCN 输出节点级双曲表示 `H_gcn`
- **WHEN** 模块 3 执行 readout
- **THEN** 默认主路线 MUST 使用 `mean_std` readout，而不是省略公式地写作 Fréchet mean
- **AND** MUST 先执行：

$$
U_{b,n}
=
\log_0^c(H_{\mathrm{gcn},b,n}),
\qquad
U\in\mathbb{R}^{B\times N\times H}.
$$

- **AND** MUST 计算节点维统计：

$$
\mu_b=\frac{1}{N}\sum_{n=1}^{N}U_{b,n},
\qquad
\sigma_b=
\sqrt{
\frac{1}{N}\sum_{n=1}^{N}(U_{b,n}-\mu_b)^2
},
\qquad
\text{默认不计算坐标级 }\max\text{ 统计。}
$$

- **AND** MUST 得到图级切空间向量并映回 Poincare Ball：

$$
z^{\mathrm{tan}}_b
=
f_{\mathrm{readout}}([\mu_b;\sigma_b]),
\qquad
z_{\mathrm{global},b}
=
\exp_0^c(z^{\mathrm{tan}}_b).
$$

- **AND** `z_global` MUST 位于 Poincare Ball 或对应切空间映射可投影回 Poincare Ball
- **AND** 文档 MUST 说明 mean/std 分别对应全脑平均状态和脑区异质性，设计原因是该低自由度读出比 attention readout 更不易在小样本 fMRI 中过拟合
- **AND** 文档 MUST 说明 `node_stats` 的坐标级 max 仅作消融，因为它依赖切空间坐标轴方向，几何意义弱且容易放大单 ROI 噪声

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

### Requirement: 模块 3 使用可解释 readout 得到 `z_global`

模块 3 SHALL 从 ROI 节点级双曲表示中得到图级全脑表示 `z_global`。默认主路线 SHOULD 使用 `mean_std` readout，即在原点切空间中计算节点表示的 mean 和 std，再用 MLP 得到图级切空间向量。`node_stats` 中包含的坐标级 max 仅作为消融入口。

#### Scenario: 计算 mean_std readout

- **GIVEN** HGCN 输出节点表示 $H_{\mathrm{gcn}}\in\mathbb{B}_c^{B\times N\times H}$
- **WHEN** 模块 3 需要得到 `z_global`
- **THEN** 系统 MUST 先执行 Poincare log map：

$$
U_{b,i}
=
\log_0^c
\left(
H_{\mathrm{gcn},b,i}
\right)
\in\mathbb{R}^{H}.
$$

- **AND** 系统 MUST 计算节点统计：

$$
s_b
=
\left[
\operatorname{mean}_{i}(U_{b,i}),
\operatorname{std}_{i}(U_{b,i})
\right].
$$

- **AND** 系统 MUST 用 MLP 得到切空间图表示：

$$
z_{\mathrm{tan},b}
=
f_{\mathrm{readout}}(s_b).
$$

- **AND** 系统 MUST 将其映射回 Poincare Ball：

$$
z_{\mathrm{global},b}
=
\exp_0^c
\left(
z_{\mathrm{tan},b}
\right).
$$

- **AND** 缓存中 MUST 同时提供 `z_global` 与 `z_tangent`，供模块 4 HPEC、t-SNE 和 TensorBoard 诊断使用

#### Scenario: readout 设计来源与原因

- **WHEN** 开发者阅读模块 3 规范
- **THEN** 文档 MUST 说明该 readout 来源于 HGCN 的“切空间中做欧氏计算、再映射回双曲空间”原则
- **AND** 文档 MUST 说明 mean/std 分别保留全脑平均状态和节点离散程度
- **AND** 文档 MUST 说明它比注意力 readout 或复杂网络先验 readout 更适合作为默认路线，因为完整测试中速度更快且稳定性更好
- **AND** 文档 MUST 说明坐标级 max 不是默认项，因为切空间坐标轴没有固定脑区物理含义

#### Scenario: 默认 readout 公式必须完整

- **WHEN** 文档描述默认模块 3 readout
- **THEN** MUST 写出 Poincare log map、mean/std 统计、MLP readout 和 exp map 的 LaTeX 公式
- **AND** MUST 写清 $H_{\mathrm{gcn}}\in\mathbb{B}_c^{B\times N\times H}$、$U\in\mathbb{R}^{B\times N\times H}$、$z_{\mathrm{tan}}\in\mathbb{R}^{B\times H}$ 和 $z_{\mathrm{global}}\in\mathbb{B}_c^{B\times H}$ 的维度
- **AND** MUST 写明原理来源是 HGCN 中“在切空间进行欧氏计算，再映射回双曲空间”的常用做法

### Requirement: 模块 3 可选因果子网络 readout

模块 3 SHALL 保持默认 `mean_std` 主路线，并 MAY 支持 `causal_subnetwork` readout 作为消融或论文补充实验，用于根据模块 2 因果图抽取若干重要 ROI 子网络并影响 `z_global`。

#### Scenario: 计算因果子网络摘要

- **GIVEN** 分类图 $A_{\mathrm{cls}}\in\mathbb{R}^{B\times N\times N}$ 与切空间节点表示 $U\in\mathbb{R}^{B\times N\times H}$
- **WHEN** `hgcn_readout_mode == "causal_subnetwork"`
- **THEN** 系统 SHOULD 根据节点入度和出度计算重要性：

$$
r_{b,i}
=
\sum_{j=1}^{N}
\left|A_{\mathrm{cls},b,j,i}\right|
+
\sum_{j=1}^{N}
\left|A_{\mathrm{cls},b,i,j}\right|.
$$

- **AND** 系统 SHOULD 保留 top-$K$ ROI 形成子网络摘要：

$$
u_{\mathrm{sub},b}
=
\sum_{i\in\operatorname{TopK}(r_b)}
\alpha_{b,i}U_{b,i},
\qquad
\sum_i\alpha_{b,i}=1.
$$

- **AND** 子网络摘要只能以轻量残差方式影响 `z_tangent`，避免覆盖全脑 readout

#### Scenario: 因果子网络定位

- **WHEN** 文档描述 `causal_subnetwork`
- **THEN** MUST 明确它是可选弱结构先验
- **AND** MUST 说明它的设计原因是让模块 2 学到的重要有向边能在模块 3 readout 中被观察和消融
- **AND** MUST 说明它训练更慢，默认主路线仍使用 `mean_std`

### Requirement: 模块 3 支持可选多阶因果输入编码

模块 3 SHALL 可选接收多阶因果可达性增强后的节点特征；当前正式训练只运行标准节点特征视图。

#### Scenario: 保持默认 readout 兼容
- **GIVEN** 多阶编码被启用
- **WHEN** 模块 3 执行 HGCN
- **THEN** 默认 `mean_std` readout MUST 保持可用
- **AND** 训练、验证与推理 MUST 只运行标准视图

