## ADDED Requirements

### Requirement: 模块 4 支持 Lorentz-to-Poincare bridge

模块 4 SHALL 在 `lp_brain_hpec` 路径中将模块 3 的 Lorentz 图级 embedding 通过显式 stereographic bridge 投影到 Poincare Ball，再执行 HPEC prototype energy 分类。

#### Scenario: 执行显式 bridge

- **GIVEN** `graph_lorentz` 的形状为：

$$
g^{\mathcal{L}}
\in
\mathbb{R}^{B\times(D+1)}
$$

- **AND** 曲率参数满足：

$$
c>0
$$

- **WHEN** 模块 4 执行 Lorentz-to-Poincare bridge
- **THEN** 系统 MUST 将 Lorentz embedding 拆成 time-like 和 spatial 分量：

$$
g^{\mathcal{L}}_b
=
\left[
g^{\mathcal{L}}_{b,0},
g^{\mathcal{L}}_{b,1:D}
\right]
$$

- **AND** MUST 使用 stereographic projection 得到 Poincare 表示：

$$
z^{\mathrm{bridge}}_b
=
\frac{
g^{\mathcal{L}}_{b,1:D}
}{
g^{\mathcal{L}}_{b,0}
+
\frac{1}{\sqrt{c}}
+
\epsilon
}
$$

- **AND** 输出形状 MUST 为：

$$
z^{\mathrm{bridge}}
\in
\mathbb{R}^{B\times D}
$$

- **AND** 输出 MUST 被投影或裁剪到 Poincare Ball 有效区域：

$$
\|z^{\mathrm{bridge}}_b\|_2
<
\frac{1}{\sqrt{c}}
$$

- **AND** 文档 MUST 说明该公式来源于 Lorentz hyperboloid 到 Poincare Ball 的标准 stereographic bridge
- **AND** 文档 MUST 说明这样设计的原因是模块 3 的 Lorentz 表示不能直接送入定义在 Poincare Ball 中的 HPEC energy

#### Scenario: 缓存 bridge 诊断

- **WHEN** bridge 完成
- **THEN** 系统 MUST 缓存 bridge 前 Lorentz embedding、bridge 后 Poincare embedding 和 Poincare 半径
- **AND** Poincare 半径均值 SHOULD 计算为：

$$
\mathrm{bridge\_radius\_mean}
=
\frac{1}{B}
\sum_{b=1}^{B}
\|z^{\mathrm{bridge}}_b\|_2
$$

- **AND** Poincare 半径最大值 SHOULD 计算为：

$$
\mathrm{bridge\_radius\_max}
=
\max_{b=1,\ldots,B}
\|z^{\mathrm{bridge}}_b\|_2
$$

- **AND** 这些诊断量 MUST 标注为“不参与训练”

### Requirement: 模块 4 支持 MAC 半径裁剪

模块 4 SHALL 在 LP-Brain-HPEC 路径中支持 MAC(Mobius Annulus Clipping) 或等价半径安全环带裁剪，避免样本表示挤到中心或越过球边界。

#### Scenario: 低半径样本被推入安全环带

- **GIVEN** Poincare 表示：

$$
z^{\mathrm{bridge}}_b\in\mathbb{R}^{D}
$$

- **AND** 半径：

$$
r_b
=
\|z^{\mathrm{bridge}}_b\|_2
$$

- **AND** $r_b<r_{\min}$
- **WHEN** MAC 执行
- **THEN** 系统 MUST 将半径裁剪到：

$$
\tilde{r}_b
=
r_{\min}
$$

- **AND** 输出 MUST 为：

$$
z^{\mathrm{mac}}_b
=
z^{\mathrm{bridge}}_b
\cdot
\frac{\tilde{r}_b}{r_b+\epsilon}
$$

- **AND** 系统 MUST 避免除以 $0$ 或产生 NaN
- **AND** 文档 MUST 说明低半径裁剪的原因是防止样本全部靠近原点，导致 HPEC cone aperture 过宽、类别不可分

#### Scenario: 高半径样本被拉回球内

- **GIVEN** Poincare 表示半径满足：

$$
r_b>r_{\max}
$$

- **WHEN** MAC 执行
- **THEN** 系统 MUST 将半径裁剪到：

$$
\tilde{r}_b
=
r_{\max}
$$

- **AND** `mac_max_radius` MUST 小于 Poincare Ball 边界半径：

$$
r_{\max}
<
\frac{1}{\sqrt{c}}
$$

- **AND** 输出 MUST 仍为：

$$
z^{\mathrm{mac}}_b
=
z^{\mathrm{bridge}}_b
\cdot
\frac{\tilde{r}_b}{r_b+\epsilon}
$$

- **AND** 文档 MUST 说明高半径裁剪的原因是避免双曲距离、`arctanh` 和 HPEC energy 在边界附近数值爆炸

#### Scenario: 记录 MAC 诊断

- **WHEN** MAC 执行
- **THEN** 系统 MUST 记录低半径 clip 比例：

$$
\mathrm{low\_clip\_ratio}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}[r_b<r_{\min}]
$$

- **AND** 系统 MUST 记录高半径 clip 比例：

$$
\mathrm{high\_clip\_ratio}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}[r_b>r_{\max}]
$$

- **AND** 系统 SHOULD 记录 MAC 后半径均值：

$$
\mathrm{mac\_radius\_mean}
=
\frac{1}{B}
\sum_{b=1}^{B}
\|z^{\mathrm{mac}}_b\|_2
$$

- **AND** 可视化或 TensorBoard SHOULD 显示 MAC 后半径分布
- **AND** 上述诊断量 MUST 标注为“不参与训练”

### Requirement: 模块 4 支持 HBR 半径惩罚

模块 4 SHALL 支持 HBR(Hyperbolic Boundary Regularization) 软惩罚，用于限制样本过度靠近 Poincare Ball 边界。

#### Scenario: 计算 HBR loss

- **GIVEN** MAC 后的 Poincare 表示：

$$
z^{\mathrm{mac}}
\in
\mathbb{B}_c^{B\times D}
$$

- **AND** `hbr_loss_weight > 0`
- **WHEN** 训练流程计算模块 4 loss
- **THEN** 系统 MUST 先计算双曲半径近似：

$$
\rho_b
=
\operatorname{arctanh}
\left(
\sqrt{c}
\|z^{\mathrm{mac}}_b\|_2
\right)
$$

- **AND** MUST 计算未加权 HBR loss：

$$
\mathcal{L}^{0}_{\mathrm{HBR}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\operatorname{ReLU}
\left(
\rho_b-R_{\mathrm{safe}}
\right)^2
$$

- **AND** 加权 HBR loss MUST 为：

$$
\mathcal{L}_{\mathrm{HBR}}
=
\lambda_{\mathrm{HBR}}
\cdot
\mathcal{L}^{0}_{\mathrm{HBR}}
$$

- **AND** 加权 HBR loss MUST 能加入总 loss：

$$
\mathcal{L}_{\mathrm{total}}
\leftarrow
\mathcal{L}_{\mathrm{total}}
+
\mathcal{L}_{\mathrm{HBR}}
$$

- **AND** HBR loss MUST 能参与 PyTorch autograd 反向传播
- **AND** 文档 MUST 区分未加权项 $\mathcal{L}^{0}_{\mathrm{HBR}}$ 与加权项 $\mathcal{L}_{\mathrm{HBR}}$

#### Scenario: 关闭 HBR loss

- **GIVEN** `hbr_loss_weight == 0`
- **WHEN** 训练流程计算模块 4 loss
- **THEN** 加权 HBR loss MUST 满足：

$$
\mathcal{L}_{\mathrm{HBR}}
=
0
$$

- **AND** HBR loss MUST 不影响总 loss
- **AND** 诊断字段 MUST 显示为 $0$ 或等价空贡献

### Requirement: LP-Brain-HPEC 保持 HPEC energy final classifier

LP-Brain-HPEC 路径 SHALL 继续使用 HPEC energy/prototype 作为最终分类依据，而不是退回普通线性分类头。

#### Scenario: 使用 energy prediction

- **GIVEN** MAC 后 Poincare 表示已经输入 HPEC energy 层：

$$
z_b=z^{\mathrm{mac}}_b
$$

- **AND** 每类 prototype 为：

$$
p_{k,m}\in\mathbb{B}_c^D
$$

- **WHEN** 模块 4 输出分类结果
- **THEN** prototype cone aperture MUST 为：

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
\right)
$$

- **AND** prototype-level energy MUST 为：

$$
E_{b,k,m}
=
\operatorname{ReLU}
\left(
\Xi(p_{k,m},z_b)-\psi(p_{k,m})
\right)
+
\lambda_d d_c(z_b,p_{k,m})
$$

- **AND** 类别能量 MUST 使用 softmin 或等价多 prototype 聚合：

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
\pi_{b,k,m}E_{b,k,m}
$$

- **AND** 预测类别 MUST 来自最小 energy：

$$
\hat{y}_b
=
\operatorname*{argmin}_{k}
E_{b,k}
$$

- **AND** 概率 MUST 来自 energy-based softmax：

$$
p_{b,k}
=
\frac{
\exp(-E_{b,k})
}{
\sum_{r=1}^{K}
\exp(-E_{b,r})
}
$$

- **AND** 指标计算 MUST 使用 energy-based prediction 和 probability，除非文档明确将该路径改为 residual fusion 对照
- **AND** 文档 MUST 说明该步骤来源于 HPEC / hyperbolic entailment cone 和 prototype learning
- **AND** 文档 MUST 说明这样设计的原因是保持模块 4 的双曲 prototype energy 叙事，而不是让普通线性分类头接管最终分类

#### Scenario: 分类 loss 不绕开 HPEC

- **GIVEN** `module34_arch == "lp_brain_hpec"`
- **WHEN** 训练流程计算 primary loss
- **THEN** primary loss MUST 以 HPEC final CE 或 HPEC margin/energy loss 为主
- **AND** 若使用 CE，公式 MUST 为：

$$
\mathcal{L}_{\mathrm{HPEC\_CE}}
=
-
\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{
\exp(-E_{b,y_b})
}{
\sum_{k=1}^{K}
\exp(-E_{b,k})
}
$$

- **AND** 系统 MUST NOT 默认使用 GCN fallback logits 作为最终分类输出
- **AND** 若后续为了稳定性加入 residual fusion，文档 MUST 重新写出最终 logits、loss 和指标口径，且明确这是否仍属于 LP-Brain-HPEC 主实验路径
