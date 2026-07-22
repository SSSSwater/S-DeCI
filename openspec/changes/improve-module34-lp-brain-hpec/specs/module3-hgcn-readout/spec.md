## ADDED Requirements

### Requirement: 模块 3 支持 LP-Brain-HPEC Lorentz 有向 readout

模块 3 SHALL 支持 `lp_brain_hpec` 路径，将模块 1 节点特征提升到 Lorentz manifold，并使用模块 2 的有向因果图执行入边/出边分离的 Lorentz 图卷积。该路径属于实验对照路径；默认主路线仍以 `docs/新模块设计.md` 为准。

#### Scenario: Lorentz lifting 输入输出形状与公式

- **GIVEN** 节点特征形状为：

$$
C\in\mathbb{R}^{B\times N\times d_{\mathrm{in}}}
$$

- **WHEN** `module34_arch == "lp_brain_hpec"` 且模块 3 执行 forward
- **THEN** 系统 MUST 先将节点特征投影到 Lorentz 原点切空间：

$$
u_{b,i}
=
W_{\mathrm{lift}}C_{b,i}+b_{\mathrm{lift}},
\qquad
u_{b,i}\in\mathbb{R}^{D}
$$

- **AND** MUST 对切空间向量做范数限幅：

$$
\tilde{u}_{b,i}
=
u_{b,i}
\cdot
\min
\left(
1,
\frac{R_{\mathrm{tan}}}{\|u_{b,i}\|_2+\epsilon}
\right)
$$

- **AND** MUST 将其扩展为 Lorentz tangent vector：

$$
v_{b,i}
=
\left[
0,\tilde{u}_{b,i}
\right]
\in\mathbb{R}^{D+1}
$$

- **AND** MUST 通过指数映射得到初始 Lorentz 节点：

$$
h^{(0)}_{b,i}
=
\exp_o^c(v_{b,i})
$$

- **AND** 输出形状 MUST 为：

$$
h^{(0)}
\in
\mathbb{R}^{B\times N\times(D+1)}
$$

- **AND** Lorentz time-like 维度 MUST 位于最后一维的第 0 个位置
- **AND** 文档 MUST 说明该步骤来源于 Lorentz / Hyperboloid model 的切空间指数映射
- **AND** 文档 MUST 说明这样设计的原因是欧氏节点特征不能直接当作 Lorentz manifold 点使用

#### Scenario: 使用有向因果图区分入边和出边

- **GIVEN** 模块 2 输出 adjacency，语义为：

$$
A_{\mathrm{cls},b,i,j}
\equiv
\text{parent ROI }i
\rightarrow
\text{child ROI }j
$$

- **WHEN** Directed Lorentz GCN 聚合节点
- **THEN** 系统 MUST 先将 Lorentz 节点映射到原点切空间：

$$
v^{(r)}_{b,i}
=
\log_o^c(h^{(r)}_{b,i})
$$

- **AND** 入边聚合 MUST 使用 $A_{\mathrm{cls},b,i,j}$：

$$
m^{\mathrm{in}}_{b,j}
=
\sum_{i=1}^{N}
A_{\mathrm{cls},b,i,j}
\left(
W_{\mathrm{in}}v^{(r)}_{b,i}
+
b_{\mathrm{in}}
\right)
$$

- **AND** 出边聚合 MUST 使用 $A_{\mathrm{cls},b,i,j}$ 的出边方向：

$$
m^{\mathrm{out}}_{b,i}
=
\sum_{j=1}^{N}
A_{\mathrm{cls},b,i,j}
\left(
W_{\mathrm{out}}v^{(r)}_{b,j}
+
b_{\mathrm{out}}
\right)
$$

- **AND** 出边平衡系数 MUST 为可学习或可配置：

$$
\alpha_{\mathrm{out}}
=
\sigma(a_{\mathrm{out}})
$$

- **AND** 入边和出边 MUST 融合为：

$$
m^{(r)}_{b,i}
=
(1-\alpha_{\mathrm{out}})m^{\mathrm{in}}_{b,i}
+
\alpha_{\mathrm{out}}m^{\mathrm{out}}_{b,i}
$$

- **AND** 下一层 Lorentz 节点表示 MUST 通过指数映射或等价稳定映射得到：

$$
h^{(r+1)}_{b,i}
=
\operatorname{proj}_{\mathcal{L}}
\left(
\exp_o^c
\left(
\sigma(m^{(r)}_{b,i})
\right)
\right)
$$

- **AND** 输出形状 MUST 为：

$$
h^{(r+1)}
\in
\mathbb{R}^{B\times N\times(D+1)}
$$

- **AND** 文档 MUST 说明该步骤来源于 HGCN 的切空间图传播和 directed graph 的入边/出边分离
- **AND** 文档 MUST 说明这样设计的原因是保留模块 2 时序因果图的方向语义，而不是把有向图退化成无向 FC

#### Scenario: 支持全局图和样本级图

- **GIVEN** adjacency 的形状为：

$$
A\in\mathbb{R}^{N\times N}
\quad
\text{or}
\quad
A\in\mathbb{R}^{B\times N\times N}
$$

- **WHEN** LP-Brain-HPEC 模块 3 执行 forward
- **THEN** 系统 MUST 对全局图执行 batch 广播：

$$
A^{\mathrm{batch}}_{b,i,j}
=
A_{i,j}
$$

- **AND** 系统 MUST 对样本级图逐样本使用：

$$
A^{\mathrm{batch}}_{b,i,j}
=
A_{b,i,j}
$$

- **AND** 输出节点 Lorentz 表示形状 MUST 为：

$$
h^{(R)}
\in
\mathbb{R}^{B\times N\times(D+1)}
$$

- **AND** 若输入形状不满足上述两种情况，系统 MUST 以清晰错误失败

### Requirement: 模块 3 提供 Lorentz tangent readout

模块 3 SHALL 使用 Lorentz 原点切空间 readout 生成图级 Lorentz embedding，并缓存可用于 t-SNE 与诊断的切空间表示。

#### Scenario: 切空间均值或注意力 readout

- **GIVEN** 最后一层 Lorentz 节点表示形状为：

$$
h^{(R)}
\in
\mathbb{R}^{B\times N\times(D+1)}
$$

- **WHEN** 模块 3 执行图级 readout
- **THEN** 系统 MUST 将节点表示映射到 Lorentz 原点切空间：

$$
v^{(R)}_{b,i}
=
\log_o^c(h^{(R)}_{b,i})
$$

- **AND** 若使用均值 readout，MUST 计算：

$$
\bar{v}_b
=
\frac{1}{N}
\sum_{i=1}^{N}
v^{(R)}_{b,i}
$$

- **AND** 若使用 attention readout，MUST 计算：

$$
s_{b,i}
=
w_{\mathrm{read}}^\top v^{(R)}_{b,i},
\qquad
\omega_{b,i}
=
\frac{\exp(s_{b,i})}
{\sum_{j=1}^{N}\exp(s_{b,j})}
$$

$$
\bar{v}_b
=
\sum_{i=1}^{N}
\omega_{b,i}v^{(R)}_{b,i}
$$

- **AND** MUST 将 readout 向量映射回 Lorentz manifold：

$$
g^{\mathcal{L}}_b
=
\exp_o^c(\bar{v}_b)
$$

- **AND** 输出 shape MUST 为：

$$
g^{\mathcal{L}}
\in
\mathbb{R}^{B\times(D+1)}
$$

- **AND** 文档 MUST 说明该步骤来源于 HGCN 的 tangent-space pooling
- **AND** 文档 MUST 说明它是速度与几何一致性的折中；严格 Fréchet mean 可作为未来对照，但不是当前默认实现

#### Scenario: 缓存 Lorentz 诊断

- **WHEN** LP-Brain-HPEC 模块 3 完成 forward
- **THEN** 系统 MUST 缓存 Lorentz 节点表示、图级 Lorentz embedding 和 tangent readout 表示
- **AND** Lorentz constraint error SHOULD 计算为：

$$
\mathrm{constraint\_error}
=
\frac{1}{BN}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\left|
\langle h_{b,i},h_{b,i}\rangle_{\mathcal{L}}
+
\frac{1}{c}
\right|
$$

- **AND** 入边聚合强度 SHOULD 计算为：

$$
\mathrm{in\_aggregation\_norm}
=
\frac{1}{BN}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\|m^{\mathrm{in}}_{b,i}\|_2
$$

- **AND** 出边聚合强度 SHOULD 计算为：

$$
\mathrm{out\_aggregation\_norm}
=
\frac{1}{BN}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\|m^{\mathrm{out}}_{b,i}\|_2
$$

- **AND** 这些诊断量 MUST 标注为“不参与训练”，除非另有明确 loss 公式把它们加入 $\mathcal{L}_{\mathrm{total}}$

### Requirement: 模块 3 几何 dtype 可配置

模块 3 SHALL 支持配置关键几何计算 dtype，避免将完整训练强制为 `float64`。

#### Scenario: 使用 auto 几何 dtype

- **GIVEN** `module34_geo_dtype == "auto"`
- **WHEN** 模块 3 执行 Lorentz 几何计算
- **THEN** 系统 MAY 在 Lorentz inner product、指数映射或对数映射中临时使用更高精度：

$$
\tilde{x}
=
\operatorname{cast}(x,d_{\mathrm{geo}}),
\qquad
y
=
f_{\mathrm{geo}}(\tilde{x}),
\qquad
y_{\mathrm{out}}
=
\operatorname{cast}(y,d_{\mathrm{model}})
$$

- **AND** 输出 MUST 转回与主模型兼容的 dtype
- **AND** 训练流程 MUST 不因 dtype 不一致而失败
- **AND** 文档 MUST 说明该步骤来源于双曲几何数值稳定性，而不是模型语义改动

#### Scenario: 拒绝非法 dtype 配置

- **GIVEN** 用户传入不支持的 `module34_geo_dtype`
- **WHEN** 初始化 S-DeCI
- **THEN** 系统 MUST 以清晰错误失败
