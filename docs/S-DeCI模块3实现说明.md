# S-DeCI 模块 3 实现说明

> 说明：本文档最初记录“只实现模块 3、暂不接模块 4”的阶段性实现。当前默认主路线已经升级为模块 1/2/3/4 联合路径，完整公式以 [新模块设计.md](E:/WorkingSpace/my_experience/docs/新模块设计.md) 和 OpenSpec 主规范为准。本文件保留模块 3 的工程接入说明，但不再把早期静态重构或“未实现模块 4”写作当前事实。

## 当前实现范围

模块 3 将模块 1 输出的低频生理节点特征 `C` 与模块 2 输出的分类图 $A_{\mathrm{cls}}$ 结合，经过 Backclip、Poincare Ball 投影、HGCN 图传播和 `mean_std` readout 得到全脑双曲中心点 `z_global`。当模块 4 启用时，`z_global` 继续进入 HPEC 多原型能量层；当模块 4 关闭时，模型可退回 `logmap0(z_global)` 的线性分类头。

## 数据流

1. 模块 1 从 BOLD 时间序列中提取低频生理节点特征：

$$
C\in\mathbb{R}^{B\times N\times D}.
$$

2. 模块 2 使用历史时间窗预测未来时间点，学习跨时间因果图：

$$
A_{\mathrm{lag}}\in\mathbb{R}^{L\times N\times N},
\qquad
A_0\in\mathbb{R}^{N\times N}.
$$

3. 分类图由因果图与样本 FC 门控融合得到：

$$
A_{\mathrm{cls},b}
=
F_b\odot
\left(
1+\eta\bar{A}_{\mathrm{lag}}
\right),
\qquad
\bar{A}_{\mathrm{lag}}
=
\frac{1}{L}\sum_{\ell=1}^{L}A_{\mathrm{lag}}^{(\ell)}.
$$

其中 $F_b\in\mathbb{R}^{N\times N}$ 是第 $b$ 个样本的相关系数矩阵，$\eta$ 是因果门控强度，$\bar{A}_{\mathrm{lag}}$ 是跨 lag 平均后的有向时序因果图。这样设计的原因是：FC 提供被试级稳定连接基底，$A_{\mathrm{lag}}$ 提供方向性调制，避免模块 2 学到的全局因果图完全替代样本自身连接模式。

4. 模块 3 接收 `C` 和未 detach 的 $A_{\mathrm{cls}}$，输出：
   - `C_clipped: [B, N, d_model]`
   - `H0: [B, N, d_model]`
   - `H_gcn: [B, N, hgcn_hidden_dim]`
   - `z_global: [B, hgcn_hidden_dim]`
   - `logmap0(z_global): [B, hgcn_hidden_dim]`
5. 模块 4 启用时，`z_global` 与 HPEC prototypes 计算 energy/prototype evidence，并形成双曲 evidence 增量；模块 4 关闭时，`hgcn_classifier(logmap0(z_global))` 生成分类输出。

设计来源与原因：模块 3 采用 HGCN / Poincare Ball 的“切空间计算、双曲空间表示”原则。脑网络可被理解为 ROI、功能子网络和全脑状态的层级结构，双曲空间比欧氏空间更适合表达这种潜在层级；同时默认 `mean_std` 读出只保留节点分布的一阶中心和二阶离散程度，自由度低、几何含义更清晰，更适合小样本 fMRI。

## Loss 构成

启用模块 3 后，当前 temporal 主路线的训练总损失为：

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{cls}}
+
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0).
$$

其中：

$$
\mathcal{L}_{\mathrm{cls}}
=
\mathrm{CE}
\left(
\hat{Y},
y
\right).
$$

$\hat{Y}$ 是最终 logits：模块 4 关闭时可来自 `hgcn_classifier(logmap0(z_global))`；模块 4 开启时来自欧氏局部结构 logits 与 HPEC 双曲原型 evidence 的双视角融合。$\mathcal{L}_{\mathrm{pred}}$ 来源于模块 2 的历史时间窗预测未来时间点，$h(A_0)$ 只约束同时间残余图。训练中不使用 `A_true`、`A_structure_true` 或任何真实因果矩阵监督。分类 loss 会通过 HGCN 中使用的 $A_{\mathrm{lag}}$ / $A_{\mathrm{cls}}$ 回传到模块 2 因果图参数。

## 当前默认 readout 实现方式

当前默认使用 `mean_std` readout，而不是早期的 `TangentFrechetReadout` 或 `node_stats` 作为主路线。读取过程为：

$$
U_{b,n}
=
\log_0^c(H_{\mathrm{gcn},b,n}),
\qquad
U\in\mathbb{R}^{B\times N\times H}.
$$

节点统计为：

$$
s_b
=
\left[
\operatorname{mean}_{n}(U_{b,n}),
\operatorname{std}_{n}(U_{b,n})
\right].
$$

图级切空间表示和双曲中心为：

$$
z^{\mathrm{tan}}_b
=
f_{\mathrm{readout}}(s_b),
\qquad
z_{\mathrm{global},b}
=
\exp_0^c(z^{\mathrm{tan}}_b).
$$

mean 表示全脑平均状态，std 表示 ROI 异质性。这样设计的原因是：它比 attention readout 或逐样本黎曼迭代 Fréchet mean 更快、更稳定，也不容易在 116 节点小样本交叉验证中增加过拟合。早期 `node_stats` 会额外拼接坐标级 max，但切空间坐标轴本身没有稳定的脑区物理含义，max 还容易放大单个 ROI 噪声，因此默认主线去掉该项，只把 `node_stats` 保留为消融入口。

## 可视化入口

显式设置 `visualize_causal=1` 时，每个 fold 训练结束后会调用 `S-DeCI.visualize_causal_intermediates()` 保存 heatmap。默认不保存图片。

可视化内容包括模块 2 的 `C`、temporal prediction、temporal prediction error、`A_lag_mean`、`A0`、`A_cls`、`A_learned`、方向差分矩阵，以及模块 3 的 `C_clipped`、`H0`、`H_gcn`、`z_global`、`logmap0(z_global)` 和归一化邻接矩阵。

3D 张量由 `utils.tensor_visualization.visualize_tensors` 只显示 Batch0 或指定 batch，并在副标题中显示维度提示。
