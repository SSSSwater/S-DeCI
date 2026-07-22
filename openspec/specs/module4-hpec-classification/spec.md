## Purpose

定义 `S-DeCI` 模块 4 的 HPEC 原型能量分类能力：使用模块 3 输出的双曲中心点 `z_global` 与类别 prototype 计算 HPEC angle、aperture/psi、energy matrix、prediction 和 `Loss_HPEC`。

## Requirements

### Requirement: 模块 4 可选支持 Lorentz-to-Poincare bridge 实验对照

模块 4 MAY 在 `lp_brain_hpec` 路径中将模块 3 的 Lorentz 图级 embedding 通过显式 stereographic bridge 投影到 Poincare Ball，再执行 HPEC prototype energy 分类。该路径只属于 experimental/消融对照；默认主路线 MUST 使用模块 3 `hgcn_hpec` 产生的 Poincare `z_global` 直接进入 HPEC 多原型能量层，不得把 bridge、MAC 或 HBR 写成默认必经步骤。

#### Scenario: 执行显式 bridge
- **GIVEN** `graph_lorentz` 的形状为 `[B, D + 1]`
- **AND** 曲率参数为正数
- **WHEN** 模块 4 执行 Lorentz-to-Poincare bridge
- **THEN** 系统 MUST 使用如下 LaTeX 形式或数值等价形式得到 Poincare 表示：

$$
z_{\mathrm{poincare}}
=
\frac{x_{1:D}^{\mathrm{lorentz}}}
{x_0^{\mathrm{lorentz}}+\sqrt{1/c}+\epsilon}.
$$

- **AND** 输出 `z_poincare` 的形状 MUST 为 `[B, D]`
- **AND** `z_poincare` MUST 被投影或裁剪到 Poincare Ball 有效区域
- **AND** 文档 MUST 说明该 bridge 来源于 Lorentz hyperboloid 到 Poincare Ball 的 stereographic projection，且只用于 `lp_brain_hpec` 实验路径

#### Scenario: 缓存 bridge 诊断
- **WHEN** bridge 完成
- **THEN** 系统 MUST 缓存 bridge 前 Lorentz embedding、bridge 后 Poincare embedding 和 Poincare 半径
- **AND** TensorBoard SHOULD 记录 Poincare 半径均值和最大值
- **AND** Poincare 半径诊断 MUST 在文档中写成：

$$
r_b
=
\|z_{\mathrm{poincare},b}\|_2.
$$

$$
\bar{r}
=
\frac{1}{B}
\sum_{b=1}^{B}
r_b,
\qquad
r_{\max}
=
\max_{b=1,\ldots,B}r_b.
$$

- **AND** 文档 MUST 说明这些量只用于诊断，不参与默认 `hgcn_hpec` 主路线；设计原因是 Lorentz-to-Poincare bridge 可能把样本压到原点附近或推向边界，半径统计可以直接暴露该数值问题

### Requirement: 模块 4 可选支持 MAC 半径裁剪

模块 4 MAY 在 LP-Brain-HPEC 实验路径中支持 MAC(Mobius Annulus Clipping) 或等价半径安全环带裁剪，避免样本表示挤到中心或越过球边界。默认 `hgcn_hpec` 主路线不得依赖 MAC 作为核心分类步骤。

#### Scenario: 低半径样本被推入安全环带
- **GIVEN** Poincare 表示 `z` 的范数低于 `mac_min_radius`
- **WHEN** MAC 执行
- **THEN** 系统 MUST 将 `z` 沿原方向缩放到不低于 `mac_min_radius`，公式为：

$$
z_{\mathrm{mac}}
=
z\cdot
\frac{r_{\min}}
{\|z\|_2+\epsilon},
\qquad
\|z\|_2<r_{\min}.
$$

- **AND** 系统 MUST 避免除以 0 或产生 NaN

#### Scenario: 高半径样本被拉回球内
- **GIVEN** Poincare 表示 `z` 的范数高于 `mac_max_radius`
- **WHEN** MAC 执行
- **THEN** 系统 MUST 将 `z` 沿原方向缩放到不高于 `mac_max_radius`，公式为：

$$
z_{\mathrm{mac}}
=
z\cdot
\frac{r_{\max}}
{\|z\|_2+\epsilon},
\qquad
\|z\|_2>r_{\max}.
$$

- **AND** `mac_max_radius` MUST 小于 Poincare Ball 边界半径

#### Scenario: 记录 MAC 诊断
- **WHEN** MAC 执行
- **THEN** 系统 MUST 记录低半径 clip 比例和高半径 clip 比例
- **AND** 可视化或 TensorBoard SHOULD 显示 MAC 后半径分布
- **AND** 低半径 clip 比例 MUST 写成：

$$
\rho_{\mathrm{low}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}
\left[
\|z_b\|_2<r_{\min}
\right].
$$

- **AND** 高半径 clip 比例 MUST 写成：

$$
\rho_{\mathrm{high}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}
\left[
\|z_b\|_2>r_{\max}
\right].
$$

- **AND** MAC 后半径均值 MUST 写成：

$$
\bar{r}_{\mathrm{mac}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\|z^{\mathrm{mac}}_b\|_2.
$$

- **AND** 文档 MUST 说明 MAC 来源于双曲空间数值稳定中的半径安全区间思想；设计原因是避免点靠近原点时 cone aperture 过宽、靠近边界时距离和梯度不稳定。该诊断不应写入默认主路线 loss

### Requirement: 模块 4 可选支持 HBR 半径惩罚

模块 4 MAY 支持 HBR(Hyperbolic Boundary Regularization) 软惩罚，用于限制样本过度靠近 Poincare Ball 边界。HBR 只属于 LP-Brain-HPEC 或数值稳定消融实验，不应加入默认主路线损失。

#### Scenario: 计算 HBR loss
- **GIVEN** MAC 后的 Poincare 表示 `z_stable`
- **AND** `hbr_loss_weight > 0`
- **WHEN** 训练流程计算模块 4 loss
- **THEN** 系统 MUST 计算如下 LaTeX 形式或等价惩罚：

$$
\mathcal{L}_{\mathrm{HBR}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\operatorname{ReLU}
\left(
\operatorname{arctanh}
\left(
\sqrt{c}\|z_b^{\mathrm{stable}}\|_2
\right)
-r_{\mathrm{safe}}
\right)^2.
$$

- **AND** 加权 HBR loss MUST 能加入总 loss
- **AND** HBR loss MUST 能参与 PyTorch autograd 反向传播

#### Scenario: 关闭 HBR loss
- **GIVEN** `hbr_loss_weight == 0`
- **WHEN** 训练流程计算模块 4 loss
- **THEN** HBR loss MUST 不影响总 loss
- **AND** 诊断字段 MUST 显示为 0 或等价空贡献

### Requirement: LP-Brain-HPEC 实验路径保持 HPEC energy final classifier

当用户显式启用 `module34_arch == "lp_brain_hpec"` 实验路径时，该路径 SHALL 继续使用 HPEC energy/prototype 作为最终分类依据，而不是退回普通线性分类头。该约束只用于保证实验路径语义自洽；默认主路线仍是 `hgcn_hpec` 产生 Poincare `z_global` 后进入 HPEC 多原型能量层。

#### Scenario: 使用最小 energy 预测
- **GIVEN** MAC 后 Poincare 表示已经输入 HPEC energy 层
- **WHEN** 模块 4 输出分类结果
- **THEN** 预测类别 MUST 来自 `argmin(energy_matrix)`
- **AND** 概率 MUST 来自 `softmax(-energy_matrix)` 或等价 energy-based probability
- **AND** 指标计算 MUST 使用 energy-based prediction 和 probability
- **AND** 该预测过程 MUST 写成：

$$
\hat{y}_b
=
\operatorname*{argmin}_{k}
E_{b,k}.
$$

$$
p_{b,k}
=
\frac{
\exp(-E_{b,k})
}{
\sum_{r=1}^{K}
\exp(-E_{b,r})
}.
$$

- **AND** 文档 MUST 说明 energy-only 预测的来源是 HPEC 中“低能量表示更符合类别 prototype cone”的原则；设计原因是保证 `lp_brain_hpec` 实验路径自身语义闭合，但完整测试未证明其优于默认融合路径，因此不得替代默认 `hgcn_hpec` 双视角 evidence fusion

#### Scenario: 分类 loss 不绕开 HPEC
- **GIVEN** `module34_arch == "lp_brain_hpec"`
- **WHEN** 训练流程计算 primary loss
- **THEN** primary loss MUST 以 HPEC final CE 或 HPEC margin/energy loss 为主
- **AND** 系统 MUST NOT 默认使用 GCN fallback logits 作为最终分类输出

### Requirement: 模块 4 提供 HPEC 原型能量分类组件

系统 SHALL 提供模块 4 HPEC 原型能量分类组件，用于根据模块 3 输出的双曲中心点 `z_global` 与类别原型计算能量矩阵。

#### Scenario: 创建可复用 HPEC 层

- **GIVEN** 项目需要在 `S-DeCI` 中接入模块 4
- **WHEN** 开发者查看 `layers/`
- **THEN** MUST 能找到 HPEC 原型、角度、孔径和能量计算相关层文件
- **AND** 该层文件 MUST NOT 依赖 `reference/` 目录作为运行时 import 路径

#### Scenario: HPEC 输入输出形状

- **GIVEN** `z_global` 的形状为 `[B, hgcn_hidden_dim]`
- **AND** 类别数为 `K`
- **WHEN** 模块 4 执行 forward
- **THEN** MUST 输出 `energy_matrix`
- **AND** `energy_matrix` 的形状 MUST 为 `[B, K]`
- **AND** MUST 输出每个样本的预测类别

### Requirement: HPEC 原型初始化

模块 4 SHALL 使用可分离的类别原型，并将原型投影到 Poincare Ball 中。

#### Scenario: 初始化原型

- **GIVEN** 类别数 `K` 和双曲中心维度 `D`
- **WHEN** 初始化模块 4
- **THEN** MUST 构造形状为 `[K, D]` 或 `[K, prototypes_per_class, D]` 的类别原型
- **AND** MUST 通过 hyperspherical separation 或等价方法使原型方向尽量分离
- **AND** MUST 按 `hpec_prototype_radius` 缩放后投影到 Poincare Ball
- **AND** 初始化公式 MUST 可写作：

$$
q_{k,m}
=
\operatorname{Normalize}(\tilde{q}_{k,m}),
\qquad
\|q_{k,m}\|_2=1,
$$

$$
p_{k,m}
=
\exp_0^c(r_p q_{k,m}),
\qquad
p_{k,m}\in\mathbb{B}_c^D.
$$

- **AND** 文档 MUST 说明 hyperspherical separation 的来源是 prototype learning 中“类别中心应在方向上尽量可分”的初始化思想，设计原因是避免所有 prototype 初始挤在同一方向

#### Scenario: 原型可配置

- **WHEN** 用户配置模块 4
- **THEN** 系统 MUST 支持配置原型半径
- **AND** MUST 支持配置原型是否可训练
- **AND** MUST 支持配置原型初始化步数或等价初始化强度

### Requirement: HPEC 角度能量函数

模块 4 SHALL 按 HPEC 参考实现计算角度、孔径和 energy，并提供必要的数值稳定处理。

#### Scenario: 计算角度与孔径

- **GIVEN** 双曲样本点 `z_global`
- **AND** 双曲类别原型 `prototype`
- **WHEN** 模块 4 计算 HPEC energy
- **THEN** MUST 计算样本到每个原型的角度 `Xi`
- **AND** MUST 计算每个原型的孔径 `psi`
- **AND** MUST 对 `acos`、`asin` 和除法相关输入做 clamp 或 eps 稳定化
- **AND** 文档 MUST 在该场景或相邻场景中写出：

$$
\psi(p)
=
\arcsin
\left(
\operatorname{clip}
\left[
\frac{
K_{\mathrm{cone}}(1-c\|p\|_2^2)
}{
\sqrt{c}\|p\|_2+\epsilon
}
\right]
\right).
$$

$$
\Xi(p,z)
=
\arccos
\left(
\operatorname{clip}
\left[
\frac{
\langle p,z\rangle
(1+c\|p\|_2^2)
-
\|p\|_2^2(1+c\|z\|_2^2)
}{
\|p\|_2\|p-z\|_2
\sqrt{
1+c^2\|p\|_2^2\|z\|_2^2
-2c\langle p,z\rangle
}
\epsilon
}
\right]
\right).
$$

- **AND** 文档 MUST 说明 $\psi(p)$ 是 prototype cone aperture，$\Xi(p,z)$ 是样本相对 prototype 的共形角度；来源是 Hyperbolic entailment cone / HPEC，设计原因是同时表达“类别原型指向的区域”和“样本是否落入该区域”

#### Scenario: 计算 energy

- **WHEN** 角度 `Xi` 和孔径 `psi` 可用
- **THEN** MUST 计算非负 energy `max(0, Xi - psi)`
- **AND** MUST 保留类别级 energy matrix 供 loss、预测和可视化使用

### Requirement: HPEC energy loss

模块 4 SHALL 提供 HPEC energy loss 和 HPEC logits。默认 `hgcn_hpec` 主路线中，HPEC 不直接替换全部分类路径，而是先形成双曲原型证据，再与欧氏局部结构 logits 融合；只有 `lp_brain_hpec` 等显式 experimental energy-only 路径 MAY 使用纯 `argmin(energy_matrix)` 作为最终预测。

#### Scenario: 计算 HPEC loss

- **GIVEN** `energy_matrix` 和真实标签 `label`
- **WHEN** 系统计算模块 4 分类 loss
- **THEN** 若使用 energy CE，MUST 以负能量作为 logits：

$$
\ell^{\mathrm{energy}}_{b,k}
=
-E_{b,k},
\qquad
\mathcal{L}_{\mathrm{energy\_ce}}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{
\exp(-E_{b,y_b})
}{
\sum_{k=1}^{K}\exp(-E_{b,k})
}.
$$

- **AND** 若使用 margin energy loss，MUST 写成：

$$
E^+_b=E_{b,y_b},
\qquad
E^-_b
=
\frac{1}{K-1}
\sum_{k\ne y_b}
\operatorname{ReLU}
\left(
m+E^+_b-E_{b,k}
\right),
$$

$$
\mathcal{L}_{\mathrm{HPEC}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left(
E^+_b+E^-_b
\right).
$$

- **AND** 文档 MUST 说明 $E^+_b$ 惩罚真实类别能量过高，$E^-_b$ 惩罚非真实类别能量没有比真实类别高出 margin
- **AND** `Loss_HPEC` MUST 对 batch 求平均并能参与 autograd

#### Scenario: 默认主路线融合 HPEC 证据

- **GIVEN** 默认 `module34_arch == "hgcn_hpec"`
- **AND** 模块 4 已输出 $E_{b,k}$ 与 prototype similarity evidence
- **WHEN** S-DeCI 形成最终分类输出
- **THEN** MUST 先得到 HPEC 双曲 logits：

$$
\ell^{\mathrm{hyper}}_{b,k}
=
-E_{b,k}
+\lambda_{\mathrm{evi}}\bar{s}_{b,k}.
$$

- **AND** 二分类时 MAY 将 HPEC margin 校准为双曲 evidence 增量：

$$
m_b=\ell^{\mathrm{hyper}}_{b,1}-\ell^{\mathrm{hyper}}_{b,0},
\qquad
\tilde{m}_b=
\frac{m_b-\mu_m}{\sigma_m+\epsilon}s_{\mathrm{cal}},
$$

$$
r^{\mathrm{hyper}}_b
=
\left[
-\frac{1}{2}\tilde{m}_b,
\frac{1}{2}\tilde{m}_b
\right].
$$

- **AND** 默认最终 logits MUST 可写作：

$$
\hat{Y}_b
=
\ell^{\mathrm{base}}_b
+\lambda_{\mathrm{hyp}}g_b r^{\mathrm{hyper}}_b,
$$

其中 $\ell^{\mathrm{base}}_b$ 是欧氏局部结构分类证据，$g_b$ 是可选 evidence gate。代码参数名中保留 `residual` 是为了兼容既有实验脚本，论文叙事中应理解为双曲原型证据增量。
- **AND** 默认指标 MUST 使用最终融合 logits $\hat{Y}$，而不是强制使用 `argmin(energy_matrix)`
- **AND** 文档 MUST 说明这样设计的原因：HPEC 负责提供双曲原型能量证据，但小样本中纯 energy-only 容易校准偏移，因此默认与欧氏局部结构证据做双视角 evidence fusion，而不是形成主从式附属分支

### Requirement: 模块 4 诊断缓存

模块 4 SHALL 缓存关键 HPEC 中间量，供训练诊断和可视化使用。

#### Scenario: 缓存 HPEC 中间量

- **WHEN** `S-DeCI` 启用模块 4 并完成一次 forward
- **THEN** MUST 能读取原型、角度矩阵、孔径、energy matrix、预测类别和 `Loss_HPEC`
- **AND** 缓存中间量 MUST 不改变 `S-DeCI.forward()` 的主返回值
- **AND** 角度矩阵、孔径和 energy matrix 的形状 MUST 写清：

$$
\Xi\in\mathbb{R}^{B\times K\times P},
\qquad
\psi\in\mathbb{R}^{K\times P},
\qquad
E\in\mathbb{R}^{B\times K\times P}.
$$

$$
E^{\mathrm{class}}_{b,k}
=
\sum_{m=1}^{P}
q_{b,k,m}E_{b,k,m}
\in
\mathbb{R}^{B\times K}.
$$

- **AND** prototype 余弦诊断 SHOULD 写成：

$$
\operatorname{ProtoCosMean}
=
\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}
\left|
\frac{
\log_0^c(p_a)^\top\log_0^c(p_b)
}{
\|\log_0^c(p_a)\|_2
\|\log_0^c(p_b)\|_2+\epsilon
}
\right|.
$$

- **AND** 文档 MUST 说明这些缓存量只用于可视化、TensorBoard 和诊断；除非明确出现在总 loss 公式中，否则不得把诊断量写成训练目标

### Requirement: HPEC energy 结合 cone violation 与 Poincare distance

模块 4 SHALL 按 HPEC / entailment cone 原理计算 prototype-level energy，并可加入 Poincare distance 项增强小样本 fMRI 中的原型判别。公式必须使用 LaTeX 描述，不能只写成普通文本。

#### Scenario: 计算 prototype-level energy

- **GIVEN** 样本双曲点 $z_b$、类别 $k$ 的第 $m$ 个 prototype $p_{k,m}$、角度 $\Xi(p_{k,m},z_b)$ 与孔径 $\psi(p_{k,m})$
- **WHEN** 模块 4 计算 prototype-level energy
- **THEN** 系统 MUST 先计算 cone violation：

$$
v_{b,k,m}
=
\operatorname{ReLU}
\left(
\Xi(p_{k,m},z_b)
-
\psi(p_{k,m})
\right).
$$

- **AND** 系统 MAY 按权重加入 Poincare distance：

$$
d_c(z_b,p_{k,m})
=
\frac{2}{\sqrt{c}}
\operatorname{arctanh}
\left(
\sqrt{c}
\left\|
(-p_{k,m})\oplus_c z_b
\right\|_2
\right).
$$

$$
E_{b,k,m}
=
v_{b,k,m}
+
\lambda_d\,d_c(z_b,p_{k,m}).
$$

- **AND** 当 `hpec_distance_weight == 0` 时，上式 MUST 退化为纯 HPEC cone violation energy
- **AND** 文档 MUST 说明 cone violation 来源于 HPEC entailment cone，distance 项来源于 prototype learning 中“离本类原型更近”的判别直觉，二者合用是为了同时约束“角度落入 cone”和“距离靠近原型”

#### Scenario: 计算 aperture 与 angle

- **GIVEN** prototype $p$ 位于 Poincare Ball 内
- **WHEN** 模块 4 计算 HPEC aperture
- **THEN** 系统 MUST 使用如下形式或数值等价形式：

$$
\psi(p)
=
\arcsin
\left(
\operatorname{clip}
\left[
\frac{
K_{\mathrm{cone}}(1-c\|p\|^2)
}{
\sqrt{c}\|p\|+\epsilon
}
\right]
\right).
$$

- **AND** 系统 MUST 对 `asin`、`acos` 和除法输入做 clamp/eps 稳定化
- **AND** 文档 MUST 说明 prototype 半径越大，cone 越窄，表示类别概念越具体；prototype 越接近原点，cone 越宽，表示类别概念越泛化

#### Scenario: 计算 Poincare cone angle

- **GIVEN** 样本双曲点 $z_b$ 与 prototype $p_{k,m}$ 均位于 Poincare Ball 内
- **WHEN** 模块 4 计算 HPEC cone violation
- **THEN** 文档 MUST 写出共形角度的 LaTeX 形式，例如：

$$
\Xi(p,z)
=
\arccos
\left(
\operatorname{clip}
\left[
\frac{
\langle p,z\rangle
\left(1+c\|p\|^2\right)
-
\|p\|^2
\left(1+c\|z\|^2\right)
}{
\|p\|\,
\|p-z\|\,
\sqrt{1+c^2\|p\|^2\|z\|^2-2c\langle p,z\rangle}
+\epsilon
}
\right]
\right).
$$

- **AND** MUST 说明该公式来源于 Hyperbolic entailment cone / HPEC 的 Poincare Ball 共形角度
- **AND** MUST 说明角度项用于判断样本是否位于类别 prototype 的 cone 区域内，distance 项用于补充原型学习中“同类更近”的判别直觉

### Requirement: HPEC prototype 初始化与 warm-start

模块 4 SHOULD 支持 hyperspherical separation 初始化，并可选使用训练 batch 的 `z_global` 做 prototype warm-start。warm-start 的目的是让 prototype 起点接近真实嵌入分布，但不得把测试标签或真实因果图泄漏进训练。

#### Scenario: 使用 batch warm-start 初始化 prototype

- **GIVEN** `hpec_data_init == 1`
- **AND** 当前处于训练阶段
- **AND** 已经通过模块 3 得到 $z_{\mathrm{global}}$
- **WHEN** 模块 4 首次计算 label-aware primary loss
- **THEN** 模块 4 SHOULD 在切空间中计算每类样本中心：

$$
\mu_k
=
\frac{1}{|\mathcal{B}_k|}
\sum_{b:y_b=k}
\log_0^c(z_b).
$$

- **AND** prototype 初始化 SHOULD 结合类别中心与 hyperspherical separation：

$$
\tilde{p}_{k,m}
=
\operatorname{Normalize}
\left(
\mu_k+\delta_{k,m}
\right),
\qquad
p_{k,m}
=
\exp_0^c
\left(
r_p\tilde{p}_{k,m}
\right).
$$

- **AND** $\delta_{k,m}$ MUST 是用于分散同类多个 prototype 的可复现扰动或正交方向
- **AND** 该初始化 MUST 只使用训练 batch 标签，不得使用测试标签、测试 embedding 或真实因果矩阵

#### Scenario: 不使用 warm-start 时的初始化

- **GIVEN** `hpec_data_init == 0`
- **WHEN** 初始化模块 4
- **THEN** prototype MUST 通过 hyperspherical separation 或等价方法初始化
- **AND** 初始 prototype 在切空间中的余弦相似度 SHOULD 被约束在较低范围，以避免所有类别原型挤在一起

#### Scenario: 多 prototype 聚合为类别 energy

- **GIVEN** prototype-level energy $E_{b,k,m}$ 已计算完成
- **WHEN** 模块 4 将形状 `[B, classes, P]` 聚合为 `[B, classes]`
- **THEN** 类别级 energy SHOULD 使用非负 softmin 加权平均：

$$
q_{b,k,m}
=
\frac{
\exp(-E_{b,k,m}/\tau_p)
}{
\sum_{r=1}^{P}\exp(-E_{b,k,r}/\tau_p)
},
\qquad
E_{b,k}
=
\sum_{m=1}^{P}q_{b,k,m}E_{b,k,m}.
$$

- **AND** 系统 MUST NOT 使用会让类别 energy 变成负值的无约束 `-temperature * logsumexp(-energy / temperature)` 作为默认聚合
- **AND** 文档 MUST 说明该设计让最匹配的 prototype 权重更大，同时保持“低 energy 表示更接近类别”的语义

### Requirement: HPEC prototype similarity evidence

模块 4 SHALL 在 energy 之外提供 prototype similarity evidence，用于形成双曲 evidence 增量。该 evidence 必须来自 Poincare 点的切空间方向相似度，而不是额外自由分类头。

#### Scenario: 计算 prototype similarity evidence

- **GIVEN** 样本点 $z_b\in\mathbb{B}_c^D$ 和 prototype $p_{k,m}\in\mathbb{B}_c^D$
- **WHEN** 模块 4 计算 prototype similarity evidence
- **THEN** MUST 先映射到原点切空间：

$$
z^{\mathrm{tan}}_b
=
\log_0^c(z_b),
\qquad
p^{\mathrm{tan}}_{k,m}
=
\log_0^c(p_{k,m}).
$$

- **AND** MUST 计算余弦相似度：

$$
s_{b,k,m}
=
\frac{
(z^{\mathrm{tan}}_b)^\top p^{\mathrm{tan}}_{k,m}
}{
\|z^{\mathrm{tan}}_b\|_2
\|p^{\mathrm{tan}}_{k,m}\|_2
+\epsilon
}.
$$

- **AND** MUST 将同类多个 prototype 聚合为类别 evidence：

$$
s_{b,k}
=
\log
\sum_{m=1}^{P}
\exp
\left(
\frac{s_{b,k,m}}{\tau_s}
\right).
$$

- **AND** MUST 在类别维中心化，避免 evidence 的整体尺度漂移影响分类：

$$
\bar{s}_{b,k}
=
s_{b,k}
-
\frac{1}{K}
\sum_{r=1}^{K}s_{b,r}.
$$

- **AND** HPEC 双曲 logits MUST 可写作：

$$
\ell^{\mathrm{hyper}}_{b,k}
=
-E_{b,k}
+\lambda_{\mathrm{evi}}\bar{s}_{b,k}.
$$

- **AND** 文档 MUST 说明该设计来源于 prototype learning 的方向相似性证据，设计原因是 cone violation 的 margin 在小样本中可能过窄，切空间方向相似度可作为更平滑的补充信号

### Requirement: HPEC prototype Sinkhorn EMA 慢更新

模块 4 SHOULD 支持 Sinkhorn EMA 更新 prototype。默认主路线中，prototype 不应无约束快速追随每个 batch 的训练噪声。

#### Scenario: 计算 Sinkhorn EMA prototype 更新

- **GIVEN** 训练 batch 中类别 $k$ 的样本集合 $\mathcal{B}_k=\{b\mid y_b=k\}$
- **AND** 已经得到 $z^{\mathrm{tan}}_b$ 与 $p^{\mathrm{tan}}_{k,m}$
- **WHEN** 模块 4 更新 prototype
- **THEN** MUST 先计算同类样本到同类 prototype 的相似度矩阵：

$$
S^{(k)}_{b,m}
=
\frac{
(z^{\mathrm{tan}}_b)^\top p^{\mathrm{tan}}_{k,m}
}{
\|z^{\mathrm{tan}}_b\|_2
\|p^{\mathrm{tan}}_{k,m}\|_2
+\epsilon
},
\qquad
b\in\mathcal{B}_k.
$$

- **AND** SHOULD 用 Sinkhorn-Knopp 得到近似均衡分配：

$$
Q^{(k)}
=
\operatorname{Sinkhorn}
\left(
\exp
\left(
\frac{S^{(k)}}{\epsilon_s}
\right)
\right),
\qquad
Q^{(k)}\in\mathbb{R}^{|\mathcal{B}_k|\times P}.
$$

- **AND** MUST 计算每个 prototype 的 batch 目标中心：

$$
\bar{z}^{\mathrm{tan}}_{k,m}
=
\frac{
\sum_{b\in\mathcal{B}_k}
Q^{(k)}_{b,m}
z^{\mathrm{tan}}_b
}{
\sum_{b\in\mathcal{B}_k}
Q^{(k)}_{b,m}
+\epsilon
}.
$$

- **AND** MAY 与固定初始化锚点 $a^{\mathrm{tan}}_{k,m}$ 混合以减少 batch 噪声：

$$
\tilde{z}^{\mathrm{tan}}_{k,m}
=
(1-\lambda_a)\bar{z}^{\mathrm{tan}}_{k,m}
+\lambda_a a^{\mathrm{tan}}_{k,m}.
$$

- **AND** MUST 执行 EMA：

$$
p^{\mathrm{tan},(e+1)}_{k,m}
=
\alpha_{\mathrm{ema}}p^{\mathrm{tan},(e)}_{k,m}
+(1-\alpha_{\mathrm{ema}})
\tilde{z}^{\mathrm{tan}}_{k,m}.
$$

- **AND** MUST 将更新后的切空间 prototype 限制到有效半径壳层并映射回 Poincare Ball：

$$
p^{(e+1)}_{k,m}
=
\operatorname{proj}_c
\left(
\exp_0^c
\left(
\operatorname{ShellClip}
\left(
p^{\mathrm{tan},(e+1)}_{k,m}
\right)
\right)
\right).
$$

- **AND** 文档 MUST 说明 Sinkhorn 分配的原因是避免多个 prototype 被同一批少数样本占用，EMA 的原因是让 prototype 更像稳定类别中心而不是 batch 噪声

### Requirement: 模块 4 支持标准与互补视图诊断

模块 4 SHALL 在训练期互补视图启用时输出可比较的 HPEC energy 与 prototype 匹配量，而最终预测保持使用标准视图 logits。

#### Scenario: 互补视图不改变评估预测
- **GIVEN** 互补视图训练机制已启用
- **WHEN** 模块 4 接收标准和互补 `z_global`
- **THEN** 系统 MUST 缓存两视图 HPEC 诊断
- **AND** 验证、测试和推理指标 MUST 使用标准视图最终 logits

