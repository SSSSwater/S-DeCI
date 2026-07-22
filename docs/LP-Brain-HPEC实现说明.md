# LP-Brain-HPEC 实现说明

本文档说明本项目如何把 `修改方案.md` 中的 Lorentz-to-Poincare HPEC 思路接入 S-DeCI。该实现不是新建独立主模型，而是作为 `S-DeCI` 的模块 3/4 可切换路径：

$$
\mathrm{module34\_arch}
=
\mathrm{lp\_brain\_hpec}.
$$

该路径目前属于实验对照路径，不替代 `docs/新模块设计.md` 中的默认 `hgcn_hpec` 主路线。若完整 5-fold / 50+ epoch 结果证明其稳定优于默认路径，再同步升级默认文档。

## 一、总体数据流

输入：

$$
C\in\mathbb{R}^{B\times N\times d_{\mathrm{in}}},
\qquad
A_{\mathrm{cls}}\in\mathbb{R}^{B\times N\times N}.
$$

其中 $C$ 是模块 1 输出的节点特征，$A_{\mathrm{cls}}$ 是模块 2 时序因果图和样本 FC 融合后的分类图。图的方向语义为：

$$
A_{\mathrm{cls},b,i,j}
\equiv
\text{ROI }i
\rightarrow
\text{ROI }j.
$$

完整数据流为：

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
E_{\mathrm{HPEC}}
\rightarrow
\hat{Y}.
$$

输出：

$$
z^{\mathrm{mac}}\in\mathbb{B}_c^{B\times D},
\qquad
E_{\mathrm{HPEC}}\in\mathbb{R}^{B\times K},
\qquad
\hat{Y}\in\mathbb{R}^{B\times K}.
$$

来源或原理：该路线结合 Lorentz hyperboloid、HGCN、Poincare Ball 和 HPEC。设计原因是让模块 2 的有向因果图直接驱动双曲图传播，再把图级表示送入 HPEC prototype energy。

## 二、Lorentz lifting

输入：

$$
C_{b,i}\in\mathbb{R}^{d_{\mathrm{in}}},
\qquad
b=1,\ldots,B,\quad i=1,\ldots,N.
$$

计算过程：

欧氏节点特征先投影到 Lorentz 原点切空间：

$$
u_{b,i}
=
W_{\mathrm{lift}}C_{b,i}+b_{\mathrm{lift}},
\qquad
u_{b,i}\in\mathbb{R}^{D}.
$$

为了避免指数映射过大，执行切空间范数限幅：

$$
\tilde{u}_{b,i}
=
u_{b,i}
\cdot
\min
\left(
1,
\frac{R_{\mathrm{tan}}}{\|u_{b,i}\|_2+\epsilon}
\right).
$$

将空间向量扩展为 Lorentz tangent vector：

$$
v_{b,i}
=
\left[
0,\tilde{u}_{b,i}
\right]
\in\mathbb{R}^{D+1}.
$$

Lorentz 原点为：

$$
o=
\left[
\frac{1}{\sqrt{c}},
0,
\ldots,
0
\right].
$$

指数映射得到初始 Lorentz 节点：

$$
h^{(0)}_{b,i}
=
\exp_o^c(v_{b,i}).
$$

输出：

$$
h^{(0)}\in\mathbb{R}^{B\times N\times(D+1)}.
$$

来源或原理：Lorentz model 中欧氏特征不能直接视为流形点，通常先放入切空间再通过指数映射进入双曲面。

设计原因：这样可以保持模块 3 的几何空间一致，避免把普通欧氏节点特征直接送入双曲图卷积。

## 三、Directed Lorentz GCN

输入：

$$
h^{(r)}\in\mathbb{R}^{B\times N\times(D+1)},
\qquad
A_{\mathrm{cls}}\in\mathbb{R}^{B\times N\times N}.
$$

计算过程：

先映射回原点切空间：

$$
v^{(r)}_{b,i}
=
\log_o^c(h^{(r)}_{b,i}).
$$

入边聚合使用 parent 指向当前 child 的边：

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

出边聚合使用当前节点指向其他节点的边：

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

出边权重为：

$$
\alpha_{\mathrm{out}}
=
\sigma(a_{\mathrm{out}}).
$$

入边和出边融合：

$$
m^{(r)}_{b,i}
=
(1-\alpha_{\mathrm{out}})m^{\mathrm{in}}_{b,i}
+
\alpha_{\mathrm{out}}m^{\mathrm{out}}_{b,i}.
$$

映回 Lorentz manifold：

$$
h^{(r+1)}_{b,i}
=
\operatorname{proj}_{\mathcal{L}}
\left(
\exp_o^c
\left(
\sigma(m^{(r)}_{b,i})
\right)
\right).
$$

输出：

$$
h^{(r+1)}
\in
\mathbb{R}^{B\times N\times(D+1)}.
$$

来源或原理：HGCN 通常在切空间做线性图传播，再映回双曲空间；有向图卷积则区分 incoming 和 outgoing relations。

设计原因：模块 2 学到的是有向时序因果关系。如果模块 3 把图对称化，会丢失方向性。入边表示某脑区被哪些脑区驱动，出边表示某脑区驱动哪些脑区，两者都可能与疾病状态有关。

## 四、Lorentz tangent readout

输入：

$$
h^{(R)}
\in
\mathbb{R}^{B\times N\times(D+1)}.
$$

计算过程：

节点表示回到切空间：

$$
v^{(R)}_{b,i}
=
\log_o^c(h^{(R)}_{b,i}).
$$

普通均值 readout 为：

$$
\bar{v}_{b}
=
\frac{1}{N}
\sum_{i=1}^{N}
v^{(R)}_{b,i}.
$$

若启用注意力 readout，则：

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
\bar{v}_{b}
=
\sum_{i=1}^{N}
\omega_{b,i}v^{(R)}_{b,i}.
$$

图级 Lorentz embedding：

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

来源或原理：该 readout 来自 HGCN 的切空间池化近似。严格 Fréchet mean 更几何精确，但计算更慢。

设计原因：当前 ROI 数量为 116 或更大，完整 5-fold 训练需要效率。切空间 readout 可微、速度较快，并保留了双曲映射的基本语义。

## 五、Lorentz-to-Poincare bridge

输入：

$$
g^{\mathcal{L}}_b
=
\left[
g^{\mathcal{L}}_{b,0},
g^{\mathcal{L}}_{b,1:D}
\right]
\in
\mathbb{R}^{D+1}.
$$

计算过程：

使用 stereographic projection：

$$
z^{\mathrm{bridge}}_b
=
\frac{
g^{\mathcal{L}}_{b,1:D}
}{
g^{\mathcal{L}}_{b,0}+\frac{1}{\sqrt{c}}+\epsilon
}.
$$

输出：

$$
z^{\mathrm{bridge}}
\in
\mathbb{R}^{B\times D}.
$$

来源或原理：这是 Lorentz hyperboloid 到 Poincare Ball 的标准投影公式。

设计原因：模块 4 的 HPEC energy 定义在 Poincare Ball 中，不能直接使用 Lorentz 坐标。bridge 解决了 Lorentz 和 Poincare 流形空间不一致的问题。

## 六、MAC 半径裁剪

输入：

$$
z^{\mathrm{bridge}}_b\in\mathbb{R}^{D}.
$$

计算过程：

半径：

$$
r_b
=
\|z^{\mathrm{bridge}}_b\|_2.
$$

安全环带：

$$
r_{\min}
=
\mathrm{mac\_min\_radius},
\qquad
r_{\max}
=
\mathrm{mac\_max\_radius}.
$$

裁剪半径：

$$
\tilde{r}_b
=
\min
\left(
r_{\max},
\max(r_{\min},r_b)
\right).
$$

裁剪后的表示：

$$
z^{\mathrm{mac}}_b
=
z^{\mathrm{bridge}}_b
\cdot
\frac{\tilde{r}_b}{r_b+\epsilon}.
$$

诊断量：

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
z^{\mathrm{mac}}\in\mathbb{B}_c^{B\times D}.
$$

来源或原理：MAC 是双曲表示的半径安全控制，用于避免表示全部塌到中心或贴近球边界。

设计原因：若样本全在原点附近，HPEC cone aperture 过宽，类别区域不可分；若样本贴近边界，距离和梯度容易爆炸。

## 七、HBR 半径惩罚

输入：

$$
z^{\mathrm{mac}}\in\mathbb{B}_c^{B\times D}.
$$

计算过程：

双曲半径近似：

$$
\rho_b
=
\operatorname{arctanh}
\left(
\sqrt{c}\|z^{\mathrm{mac}}_b\|_2
\right).
$$

未加权 HBR loss：

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

加权后进入总损失：

$$
\mathcal{L}_{\mathrm{HBR}}
=
\lambda_{\mathrm{HBR}}
\cdot
\mathcal{L}^{0}_{\mathrm{HBR}}.
$$

输出：

$$
\mathcal{L}_{\mathrm{HBR}}\in\mathbb{R}.
$$

来源或原理：HBR 属于 hyperbolic boundary regularization，用于限制点过度靠近边界。

设计原因：MAC 是硬裁剪，HBR 是软惩罚；二者结合能让 TensorBoard 观察到半径趋势，并减少 HPEC energy 被边界数值主导。

## 八、HPEC prototype energy

输入：

$$
z_b=z^{\mathrm{mac}}_b,
\qquad
p_{k,m}\in\mathbb{B}_c^D.
$$

计算过程：

prototype cone aperture：

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

cone angle：

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

prototype energy：

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

类别能量：

$$
\pi_{b,k,m}
=
\frac{
\exp(-E_{b,k,m}/\tau_p)
}{
\sum_{r=1}^{P}\exp(-E_{b,k,r}/\tau_p)
},
\qquad
E_{b,k}
=
\sum_{m=1}^{P}
\pi_{b,k,m}E_{b,k,m}.
$$

分类概率：

$$
p_{b,k}
=
\frac{\exp(-E_{b,k})}
{\sum_{r=1}^{K}\exp(-E_{b,r})}.
$$

输出：

$$
E\in\mathbb{R}^{B\times K},
\qquad
p\in\mathbb{R}^{B\times K}.
$$

来源或原理：HPEC / Hyperbolic entailment cone 用 cone angle 和双曲距离衡量样本是否落入 prototype 类别区域。

设计原因：该路径的目标是让模块 4 以双曲 prototype energy 讲分类故事，而不是退回普通线性分类头。

## 九、采纳与适配

采纳的部分：

- 使用 Lorentz lifting 和有向入边/出边图卷积承接模块 2 的非对称因果图。
- 使用显式 Lorentz-to-Poincare bridge，避免把 Lorentz 表示直接当成 Poincare 表示。
- 使用 MAC 和 HBR 记录/约束 Poincare 半径。
- 最终分类仍以 HPEC energy/prototype 为依据。

项目化适配的部分：

- 没有默认全程 `float64`，而是通过 `module34_geo_dtype` 控制几何计算精度。
- 没有覆盖旧 `hgcn_hpec` 路径；默认仍为旧路径，便于复现实验和消融对照。
- LP 输出复用 `latest_module3_output.z_global` 表示进入 HPEC 的 Poincare embedding，同时缓存 `graph_lorentz`、`poincare_bridge`、`poincare_mac` 等中间量。

## 十、关键参数

| 参数 | 含义 |
| --- | --- |
| `module34_arch` | `hgcn_hpec` 或 `lp_brain_hpec` |
| `lorentz_layers` | Directed Lorentz GCN 层数 |
| `lorentz_curvature` | Lorentz/Poincare 曲率 |
| `lorentz_alpha_out_init` | 出边聚合初始权重 |
| `lorentz_max_tangent_norm` | Lorentz 切空间限幅 |
| `module34_geo_dtype` | `auto`、`float32` 或 `float64` |
| `mac_min_radius`、`mac_max_radius` | MAC 安全半径范围 |
| `hbr_safe_radius`、`hbr_loss_weight` | HBR 边界半径惩罚 |

## 十一、诊断量

训练过程中会记录：

$$
\mathrm{lp\_lorentz\_constraint\_error}
=
\frac{1}{BN}
\sum_{b=1}^{B}\sum_{i=1}^{N}
\left|
\langle h_{b,i},h_{b,i}\rangle_{\mathcal{L}}
+
\frac{1}{c}
\right|.
$$

$$
\mathrm{lp\_in\_aggregation\_norm}
=
\frac{1}{BN}
\sum_{b=1}^{B}\sum_{i=1}^{N}
\|m^{\mathrm{in}}_{b,i}\|_2.
$$

$$
\mathrm{lp\_out\_aggregation\_norm}
=
\frac{1}{BN}
\sum_{b=1}^{B}\sum_{i=1}^{N}
\|m^{\mathrm{out}}_{b,i}\|_2.
$$

$$
\mathrm{lp\_mac\_radius\_mean}
=
\frac{1}{B}
\sum_{b=1}^{B}
\|z^{\mathrm{mac}}_b\|_2.
$$

这些字段只用于控制台打印、TensorBoard 和中间量可视化，不直接参与训练。真正参与训练的是加权后的 HBR loss：

$$
\mathcal{L}_{\mathrm{HBR}}
=
\lambda_{\mathrm{HBR}}
\cdot
\mathcal{L}^{0}_{\mathrm{HBR}}.
$$

## 十二、回滚方式

回到旧 HGCN/HPEC：

```bash
--module34_arch hgcn_hpec
```

关闭模块 3/4，走 GCN fallback 消融路径：

```bash
--use_hyperbolic_modules34 0
```
