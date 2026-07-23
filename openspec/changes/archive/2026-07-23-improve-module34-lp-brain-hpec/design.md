## Context

当前 S-DeCI 已具备四个默认模块：

$$
X
\rightarrow
C,\;X_{\mathrm{temp}}
\rightarrow
A_{\mathrm{lag}},A_0
\rightarrow
A_{\mathrm{cls}}
\rightarrow
z_{\mathrm{global}}
\rightarrow
E_{\mathrm{HPEC}}
\rightarrow
\hat{Y}.
$$

其中 $X\in\mathbb{R}^{B\times T\times N}$ 是 BOLD 时间序列，$C\in\mathbb{R}^{B\times N\times d_{\mathrm{in}}}$ 是模块 1 输出的节点特征，$A_{\mathrm{cls}}\in\mathbb{R}^{B\times N\times N}$ 是模块 2 和样本 FC 融合后的分类图。

完整实验显示：模块 3 单独启用时能接近 GCN fallback，但模块 4 HPEC final head 容易出现训练集继续变好、测试集下降。TensorBoard 中也能看到 $z_{\mathrm{global}}$ 半径和 prototype 相似度不稳定。因此本变更参考 `修改方案.md`，新增 `module34_arch=lp_brain_hpec` 路径，让模块 3/4 更直接地使用模块 2 学到的有向因果图。

来源或原理：

- Lorentz manifold / Hyperboloid model：双曲空间的一种数值表达，适合建模层级结构。
- HGCN：在双曲空间中进行图传播。
- Poincare Ball：HPEC 原型能量所在空间。
- Hyperbolic entailment cone / HPEC：用 cone energy 描述类别原型区域。
- 工程稳定性原则：跨流形映射必须显式处理半径，避免中心塌缩、边界 NaN 和 dtype 不稳定。

设计原因：该方案不绕开模块 3/4，也不把模块 4 权重调到近似无效；它让“模块 2 有向因果图 $\rightarrow$ 双曲图传播 $\rightarrow$ 原型能量分类”的故事更连贯。

## Goals / Non-Goals

**Goals:**

- 新增 Lorentz lifting、Directed Lorentz GCN、Lorentz tangent readout、Lorentz-to-Poincare bridge、MAC/HBR-HPEC 组件。
- 模块 3/4 启用时，明确使用模块 2 的有向因果图 $A[parent, child]$，同时建模入边和出边。
- 模块 4 继续使用 HPEC energy/prototype classifier，不通过降低模块 4 权重绕开问题。
- 输出可解释诊断量：Lorentz constraint error、入边/出边聚合强度、Poincare 半径、MAC clip 比例、HBR loss、prototype energy。
- 提供 MDD 完整 5-fold / 50 epoch 对比，并写入 `result.xlsx` 与 TensorBoard。

**Non-Goals:**

- 不重新设计模块 1 或模块 2 的时序因果学习目标。
- 不引入除 `geoopt` 外的新大型几何库。
- 不把 GCN fallback 作为 `lp_brain_hpec` 的默认最终分类器。
- 不默认修改所有数据集最佳参数；先以 MDD AAL116 为主要验证对象。

## Detailed Design

### Step 1: Lorentz lifting

输入：

$$
C=\{C_{b,i}\}
\in
\mathbb{R}^{B\times N\times d_{\mathrm{in}}},
$$

其中 $B$ 是 batch size，$N$ 是 ROI 数，$d_{\mathrm{in}}$ 是模块 1 输出的节点特征维度。

计算过程：

先把节点特征投影到 Lorentz 原点切空间的空间分量：

$$
u_{b,i}
=
W_{\mathrm{lift}}C_{b,i}+b_{\mathrm{lift}},
\qquad
u_{b,i}\in\mathbb{R}^{D}.
$$

为避免指数映射前向量范数过大，进行切空间限幅：

$$
\tilde{u}_{b,i}
=
u_{b,i}
\cdot
\min
\left(
1,\;
\frac{R_{\mathrm{tan}}}{\|u_{b,i}\|_2+\epsilon}
\right).
$$

这里的乘法表示向量整体缩放。若代码中使用 `clamp` 或等价写法，其数学语义必须等价于：

$$
\|\tilde{u}_{b,i}\|_2
\le
R_{\mathrm{tan}}.
$$

在曲率 $c>0$ 的 Lorentz 模型中，原点为：

$$
o=
\left[
\frac{1}{\sqrt{c}},
0,
\ldots,
0
\right]
\in\mathbb{R}^{D+1}.
$$

将切空间向量写成 time-like 分量为 $0$ 的 Lorentz tangent vector：

$$
v_{b,i}
=
\left[
0,\tilde{u}_{b,i}
\right]
\in\mathbb{R}^{D+1}.
$$

指数映射得到 Lorentz 节点表示：

$$
h^{(0)}_{b,i}
=
\exp_o^c(v_{b,i})
=
\cosh(\sqrt{c}\|v_{b,i}\|_{\mathcal{L}})
o
+
\frac{
\sinh(\sqrt{c}\|v_{b,i}\|_{\mathcal{L}})
}{
\sqrt{c}\|v_{b,i}\|_{\mathcal{L}}+\epsilon
}
v_{b,i}.
$$

Lorentz 内积为：

$$
\langle x,y\rangle_{\mathcal{L}}
=
-x_0y_0
+
\sum_{r=1}^{D}x_ry_r.
$$

输出：

$$
h^{(0)}
\in
\mathbb{R}^{B\times N\times(D+1)}.
$$

其中最后一维第 $0$ 个位置是 time-like 分量。

来源或原理：该步骤来自 Hyperboloid / Lorentz model 的指数映射。HGCN 类方法通常先把欧氏特征放入切空间，再映射到双曲流形。

设计原因：模块 1 输出仍是欧氏节点特征，不能直接当作 Lorentz 点。显式 lifting 让后续图传播在同一个双曲几何空间中进行，避免“欧氏特征直接混入双曲 prototype”的语义断裂。

### Step 2: Directed Lorentz GCN

输入：

$$
h^{(r)}
\in
\mathbb{R}^{B\times N\times(D+1)},
\qquad
A_{\mathrm{cls}}
\in
\mathbb{R}^{B\times N\times N}.
$$

图语义保持：

$$
A_{\mathrm{cls},b,i,j}
\equiv
\text{parent ROI }i
\rightarrow
\text{child ROI }j.
$$

计算过程：

先将 Lorentz 节点拉回原点切空间：

$$
v^{(r)}_{b,i}
=
\log_o^c(h^{(r)}_{b,i})
\in
\mathbb{R}^{D+1}.
$$

入边聚合读取所有 parent 指向当前 child 的边：

$$
m^{\mathrm{in}}_{b,j}
=
\sum_{i=1}^{N}
A_{\mathrm{cls},b,i,j}
\left(
W_{\mathrm{in}}v^{(r)}_{b,i}
+
b_{\mathrm{in}}
\right).
$$

出边聚合读取当前节点指向其他 child 的边：

$$
m^{\mathrm{out}}_{b,i}
=
\sum_{j=1}^{N}
A_{\mathrm{cls},b,i,j}
\left(
W_{\mathrm{out}}v^{(r)}_{b,j}
+
b_{\mathrm{out}}
\right).
$$

出边权重用可学习或可配置参数控制：

$$
\alpha_{\mathrm{out}}
=
\sigma(a_{\mathrm{out}}),
\qquad
0<\alpha_{\mathrm{out}}<1.
$$

入边和出边融合为：

$$
m^{(r)}_{b,i}
=
(1-\alpha_{\mathrm{out}})m^{\mathrm{in}}_{b,i}
+
\alpha_{\mathrm{out}}m^{\mathrm{out}}_{b,i}.
$$

经过非线性和残差后映回 Lorentz manifold：

$$
\tilde{v}^{(r+1)}_{b,i}
=
\sigma(m^{(r)}_{b,i}),
$$

$$
h^{(r+1)}_{b,i}
=
\operatorname{proj}_{\mathcal{L}}
\left(
\exp_o^c
\left(
(1-\rho_r)\tilde{v}^{(r+1)}_{b,i}
+
\rho_r v^{(r)}_{b,i}
\right)
\right).
$$

输出：

$$
h^{(r+1)}
\in
\mathbb{R}^{B\times N\times(D+1)}.
$$

来源或原理：该步骤来自 HGCN 的“切空间线性变换 + 图聚合 + 映回双曲空间”思想，并结合 directed graph neural network 的入边/出边分离建模。

设计原因：模块 2 的 $A_{\mathrm{lag}}$ 是有向图。如果在模块 3 中直接对称化，会抹掉“过去 ROI $i$ 影响未来 ROI $j$”的方向解释。入边聚合表示当前脑区被哪些脑区驱动，出边聚合表示当前脑区对哪些脑区有输出影响，两者对疾病分类可能有不同含义。

### Step 3: Lorentz tangent readout

输入：

$$
h^{(R)}
\in
\mathbb{R}^{B\times N\times(D+1)}.
$$

计算过程：

先回到原点切空间：

$$
v^{(R)}_{b,i}
=
\log_o^c(h^{(R)}_{b,i}).
$$

若使用普通均值 readout：

$$
\bar{v}_b
=
\frac{1}{N}
\sum_{i=1}^{N}
v^{(R)}_{b,i}.
$$

若使用 attention-weighted readout，则节点权重为：

$$
s_{b,i}
=
w_{\mathrm{read}}^\top v^{(R)}_{b,i},
\qquad
\omega_{b,i}
=
\frac{\exp(s_{b,i})}
{\sum_{j=1}^{N}\exp(s_{b,j})},
$$

$$
\bar{v}_b
=
\sum_{i=1}^{N}
\omega_{b,i}v^{(R)}_{b,i}.
$$

图级 Lorentz 表示为：

$$
g^{\mathcal{L}}_b
=
\exp_o^c(\bar{v}_b).
$$

输出：

$$
g^{\mathcal{L}}
\in
\mathbb{R}^{B\times(D+1)}.
$$

来源或原理：该步骤来自 HGCN 中常用的 tangent-space readout。严格 Fréchet mean 更符合流形几何，但计算更慢。

设计原因：当前完整 5-fold 训练需要可接受速度。切空间 readout 是速度和几何一致性之间的折中：聚合在切空间完成，最终再映回 Lorentz manifold。

### Step 4: Lorentz-to-Poincare bridge

输入：

$$
g^{\mathcal{L}}_b
=
\left[
g_{b,0}^{\mathcal{L}},
g_{b,1:D}^{\mathcal{L}}
\right]
\in
\mathbb{R}^{D+1}.
$$

计算过程：

使用 stereographic projection 将 Lorentz 点映射到 Poincare Ball：

$$
z^{\mathrm{bridge}}_b
=
\frac{
g_{b,1:D}^{\mathcal{L}}
}{
g_{b,0}^{\mathcal{L}}+\frac{1}{\sqrt{c}}+\epsilon
}.
$$

输出：

$$
z^{\mathrm{bridge}}
\in
\mathbb{R}^{B\times D},
\qquad
\|z^{\mathrm{bridge}}_b\|_2
<
\frac{1}{\sqrt{c}}.
$$

来源或原理：该公式是 Lorentz hyperboloid 到 Poincare Ball 的标准 stereographic bridge。

设计原因：模块 4 HPEC energy 在 Poincare Ball 中定义，Lorentz 图级 embedding 不能直接送入 HPEC。显式 bridge 保证模块 3 与模块 4 的流形空间一致。

### Step 5: MAC 半径裁剪

输入：

$$
z^{\mathrm{bridge}}
\in
\mathbb{R}^{B\times D}.
$$

计算过程：

半径为：

$$
r_b
=
\|z^{\mathrm{bridge}}_b\|_2.
$$

安全半径下界和上界为：

$$
r_{\min}=\mathrm{mac\_min\_radius},
\qquad
r_{\max}=\mathrm{mac\_max\_radius}
<
\frac{1}{\sqrt{c}}.
$$

裁剪后半径为：

$$
\tilde{r}_b
=
\min
\left(
r_{\max},
\max(r_{\min},r_b)
\right).
$$

裁剪后的 Poincare 表示为：

$$
z^{\mathrm{mac}}_b
=
z^{\mathrm{bridge}}_b
\cdot
\frac{\tilde{r}_b}{r_b+\epsilon}.
$$

低半径和高半径诊断量为：

$$
\mathrm{low\_clip\_ratio}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}[r_b<r_{\min}],
$$

$$
\mathrm{high\_clip\_ratio}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}[r_b>r_{\max}].
$$

输出：

$$
z^{\mathrm{mac}}
\in
\mathbb{B}_c^{B\times D}.
$$

来源或原理：该步骤来自 Mobius Annulus Clipping 的工程稳定思想，即把双曲表示限制在一个可训练、可区分且远离数值边界的半径环带。

设计原因：若 $z$ 全部靠近原点，HPEC cone aperture 过宽，类别难以分开；若 $z$ 贴近边界，距离和 arctanh 容易数值爆炸。MAC 用显式半径约束减少这两种问题。

### Step 6: HBR 半径惩罚

输入：

$$
z^{\mathrm{mac}}
\in
\mathbb{B}_c^{B\times D}.
$$

计算过程：

双曲半径近似为：

$$
\rho_b
=
\operatorname{arctanh}
\left(
\sqrt{c}\|z^{\mathrm{mac}}_b\|_2
\right).
$$

未加权 HBR loss 为：

$$
\mathcal{L}^{0}_{\mathrm{HBR}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\operatorname{ReLU}
\left(
\rho_b-R_{\mathrm{safe}}
\right)^2.
$$

进入总损失时使用：

$$
\mathcal{L}_{\mathrm{HBR}}
=
\lambda_{\mathrm{HBR}}
\cdot
\mathcal{L}^{0}_{\mathrm{HBR}}.
$$

输出：

$$
\mathcal{L}^{0}_{\mathrm{HBR}}\in\mathbb{R},
\qquad
\mathcal{L}_{\mathrm{HBR}}\in\mathbb{R}.
$$

来源或原理：该步骤来自双曲空间边界正则化思想。Poincare Ball 边界附近距离梯度更敏感，因此需要软惩罚。

设计原因：MAC 是硬裁剪，HBR 是软约束。二者结合可以记录并限制表示是否长期冲向边界，避免 HPEC energy 被半径爆炸主导。

### Step 7: HPEC energy 分类

输入：

$$
z_b=z^{\mathrm{mac}}_b,
\qquad
p_{k,m}\in\mathbb{B}_c^D.
$$

计算过程：

prototype cone aperture 为：

$$
\psi(p_{k,m})
=
\arcsin
\left(
\frac{
K_{\mathrm{cone}}(1-c\|p_{k,m}\|_2^2)
}{
\sqrt{c}\|p_{k,m}\|_2+\epsilon
}
\right).
$$

样本相对 prototype 的 cone angle 为：

$$
\Xi(p_{k,m},z_b)
=
\arccos
\left(
\operatorname{clip}
\left[
\frac{
\langle p_{k,m},z_b\rangle(1+c\|p_{k,m}\|_2^2)
-\|p_{k,m}\|_2^2(1+c\|z_b\|_2^2)
}{
\|p_{k,m}\|_2
\|p_{k,m}-z_b\|_2
\sqrt{
1+c^2\|p_{k,m}\|_2^2\|z_b\|_2^2
-2c\langle p_{k,m},z_b\rangle
}
+\epsilon
}
\right]
\right).
$$

prototype-level energy 为：

$$
E_{b,k,m}
=
\operatorname{ReLU}
\left(
\Xi(p_{k,m},z_b)-\psi(p_{k,m})
\right)
+
\lambda_d d_c(z_b,p_{k,m}).
$$

类别能量使用 softmin 聚合：

$$
\pi_{b,k,m}
=
\frac{
\exp(-E_{b,k,m}/\tau_p)
}{
\sum_{r=1}^{P}
\exp(-E_{b,k,r}/\tau_p)
},
\qquad
E_{b,k}
=
\sum_{m=1}^{P}
\pi_{b,k,m}E_{b,k,m}.
$$

energy-based logits 为：

$$
\ell^{\mathrm{energy}}_{b,k}
=
-E_{b,k}.
$$

输出：

$$
E\in\mathbb{R}^{B\times K},
\qquad
\ell^{\mathrm{energy}}\in\mathbb{R}^{B\times K}.
$$

来源或原理：该步骤来自 HPEC / hyperbolic entailment cone 和 prototype learning。

设计原因：模块 4 的故事必须是“样本是否落入类别 prototype 的双曲 cone 区域”。因此 `lp_brain_hpec` 路径不能默认绕回普通线性分类头。

## Decisions

### Decision 1: 作为 S-DeCI 模块 3/4 可切换路径

输入：

$$
\mathrm{module34\_arch}
\in
\{
\mathrm{hgcn\_hpec},
\mathrm{lp\_brain\_hpec}
\}.
$$

计算过程：

若：

$$
\mathrm{module34\_arch}
=
\mathrm{lp\_brain\_hpec},
$$

则启用：

$$
C,A_{\mathrm{cls}}
\rightarrow
h^{(0)}
\rightarrow
h^{(R)}
\rightarrow
g^{\mathcal{L}}
\rightarrow
z^{\mathrm{bridge}}
\rightarrow
z^{\mathrm{mac}}
\rightarrow
E.
$$

否则保留现有：

$$
C,A_{\mathrm{cls}}
\rightarrow
z_{\mathrm{global}}
\rightarrow
E_{\mathrm{HPEC}}.
$$

输出：两条路径都必须输出可用于 HPEC 的 Poincare 表示和分类 logits。

来源或原理：这是模块化消融设计原则。

设计原因：当前训练、可视化、result 和 TensorBoard 都围绕 `S-DeCI` 工作。独立模型会破坏消融可比性，也让模块 1/2 的现有调参与诊断无法复用。

### Decision 2: 几何 dtype 可配置

输入：

$$
d_{\mathrm{geo}}
\in
\{\mathrm{auto},\mathrm{float32},\mathrm{float64}\}.
$$

计算过程：

对关键几何算子临时转换 dtype：

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
\operatorname{cast}(y,d_{\mathrm{model}}).
$$

输出：

$$
y_{\mathrm{out}}
\text{ 与主模型 dtype 兼容。}
$$

来源或原理：双曲几何计算中 `arctanh`、`arccosh`、指数映射和对数映射对数值精度更敏感。

设计原因：全程 `float64` 会显著拖慢 MDD/ABIDE 完整训练；只在关键几何运算中提升精度可以折中稳定性和速度。

## Risks / Trade-offs

- Lorentz 路径比现有 HGCN 更慢。缓解方式：记录训练用时，提供层数、hidden dim、dtype 和 attention 开关，先做 1-fold smoke，再做完整 5-fold。
- MAC 外推低半径点可能改变表示分布。缓解方式：记录 $\mathrm{low\_clip\_ratio}$、$\mathrm{high\_clip\_ratio}$ 和半径均值。
- Lorentz 与 Poincare 两套流形可能出现 shape 或 dtype 错误。缓解方式：新增 forward/backward 检查，覆盖 $[B,N,D]$、$[N,N]$ 和 $[B,N,N]$。
- 新路径不一定优于当前默认路径。缓解方式：完整记录 5-fold 结果；如果效果不好，保留为消融或负结果，不替换默认最佳参数。

## Migration Plan

1. 新增 `layers/lp_brain_hpec_layer.py`，实现 Lorentz lifting、Directed Lorentz GCN、readout、bridge、MAC 和 HBR。
2. 在 `models/S_DeCI.py` 增加 `module34_arch` 分支，并缓存 LP-Brain-HPEC 中间量。
3. 在训练脚本中暴露 `module34_arch`、Lorentz、MAC 和 HBR 参数。
4. 更新 TensorBoard 和 `result.xlsx`，记录训练用时、几何诊断和最终指标。
5. 先通过 `py_compile` 和 smoke test，再执行 MDD 5-fold / 50 epoch 对比。

回滚公式化描述为：

$$
\mathrm{module34\_arch}
\leftarrow
\mathrm{hgcn\_hpec},
$$

或：

$$
\mathrm{use\_hyperbolic\_modules34}
\leftarrow
0.
$$

## 最终实验决策（2026-07-23）

上述内容记录 LP-Brain-HPEC 的历史候选设计，不再代表当前正式实现。MDD/AAL116 完整 5-fold、50 epoch 的 final-epoch 结果为 Accuracy 62.63%、Macro-F1 59.43%、AUC 62.50%，低于 Poincare HGCN-HPEC 主线且训练耗时更高。几何诊断还显示 Lorentz 消息范数衰减、bridge 后半径压缩以及 MAC 对自然分布的强制干预。

因此最终采用以下决策：

1. 当前正式模块 3 只保留 Poincare HGCN 与原点切空间 `mean_std` readout。
2. 当前正式模块 4 直接接收 Poincare `z_global`，不经过 Lorentz-to-Poincare bridge、MAC 或 HBR。
3. FC 只能用于分类图构造或独立欧氏证据，不允许直接加到双曲切向量中。
4. 最终分类采用欧氏局部结构证据与 HPEC 双曲原型证据的 dual-view evidence fusion。
5. `lp_brain_hpec`、专用 layer、架构开关和专用训练参数退出当前正式代码；历史公式只用于解释负向实验。
6. 未来若重启 Lorentz 路线，必须新建 change，并采用流形原生距离注意力、动态基点或 Lorentz centroid readout，且重新通过完整五折比较。
