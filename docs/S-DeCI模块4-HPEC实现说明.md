# S-DeCI 模块4 HPEC 实现说明

本文档说明 `S-DeCI` 中模块 4 HPEC energy/prototype evidence 的实现和调用方式。原始 [新模块设计.md](E:/WorkingSpace/my_experience/docs/新模块设计.md) 是主设计文档，本文件用于记录落地实现细节。

## 功能概览

模块 4 接在模块 3 之后：

1. 模块 1 输出 `Cycle/seasonal feature C`。
2. 模块 2 使用时间序列预测式 Temporal NTS-NOTEARS 学习因果图 `A_learned`。
3. 模块 3 使用 `C` 和 `A_learned` 经过 HGCN 得到双曲中心点 `z_global`。
4. 模块 4 使用 `z_global` 与类别 `prototype` 计算 HPEC angle、`psi/aperture` 和 `energy_matrix`。
5. 默认 `hgcn_hpec` 主路线中，`energy_matrix` 不直接作为唯一分类器，而是转换为双曲原型 evidence 增量后，与欧氏局部结构 evidence 在 logit 空间融合。

启用模块 4 时，最终分类由欧氏局部结构 evidence 和 HPEC 双曲原型 evidence 共同形成。先用 energy 和 prototype similarity 得到双曲 logits：

$$
\ell^{\mathrm{hyper}}_{b,k}
=
-E_{b,k}
+
\lambda_{\mathrm{evi}}\bar{s}_{b,k}.
$$

其中 $\bar{s}_{b,k}$ 来自样本和 prototype 在 Poincare 原点切空间中的方向相似度中心化结果：

$$
z^{\mathrm{tan}}_b=\log_0^c(z_b),
\qquad
p^{\mathrm{tan}}_{k,m}=\log_0^c(p_{k,m}),
$$

$$
s_{b,k,m}
=
\frac{
(z^{\mathrm{tan}}_b)^\top p^{\mathrm{tan}}_{k,m}
}{
\|z^{\mathrm{tan}}_b\|_2
\|p^{\mathrm{tan}}_{k,m}\|_2+\epsilon
},
\qquad
\bar{s}_{b,k}
=
s_{b,k}
-
\frac{1}{K}\sum_{r=1}^{K}s_{b,r}.
$$

二分类时，将双曲 margin 标准化为 evidence 增量：

$$
m_b
=
\ell^{\mathrm{hyper}}_{b,1}
-
\ell^{\mathrm{hyper}}_{b,0},
\qquad
\tilde{m}_b
=
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

最终 logits 为双视角 evidence fusion：

$$
\hat{Y}_b
=
\ell^{\mathrm{base}}_b
+
\lambda_{\mathrm{hyp}}g_b r^{\mathrm{hyper}}_b.
$$

这里的 $r^{\mathrm{hyper}}$ 沿用代码中的 residual 变量名，但论文叙事中应理解为“双曲原型证据增量”，不是把 HPEC 降级成欧氏分支的附属项。$\ell^{\mathrm{base}}$ 表示欧氏局部结构视角，$r^{\mathrm{hyper}}$ 表示双曲层级原型视角；二者在 logit 空间融合，避免把欧氏 FC embedding 直接注入双曲切空间造成跨流形语义污染。

默认训练、验证和测试指标均使用 $\hat{Y}$：

$$
\hat{y}_b
=
\operatorname*{argmax}_{k}\hat{Y}_{b,k},
\qquad
p_{b,k}
=
\operatorname{softmax}(\hat{Y}_b)_k.
$$

训练总损失为：

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{cls}}(\hat{Y},y)
+
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0)
+
\mathcal{L}_{\mathrm{hpec\_stab}}
+
\mathcal{L}_{\mathrm{distill}}.
$$

其中真实因果矩阵不会作为监督信号参与训练。模块 4 的设计来源是 HPEC / entailment cone：若样本点落在某个类别 prototype 的 cone 内并且距离较近，则该类别 energy 更低。

这样设计的原因是：完整实验中纯 `argmin(energy_matrix)` 容易在小样本上发生校准偏移；但完全不用 HPEC 又失去双曲原型解释。因此默认路线把 HPEC 作为可解释的双曲 evidence，使它参与最终边界，同时避免把欧氏 FC embedding 直接注入 Poincare 切空间。

## 关键文件

- `layers/hpec_energy_layer.py`：HPEC prototype 初始化、angle、psi、energy、loss 和 prediction。
- `models/S_DeCI.py`：模块 4 初始化、forward 接入、中间量缓存、HPEC primary loss。
- `exp/exp_classification_CV.py`：优先使用模型内部 primary/classification loss；默认指标使用最终融合 logits $\hat{Y}$，显式 energy-only 实验路径才使用 `argmin(energy)` 和 `softmax(-energy)`。
- `run_cv.py`：模块 4 参数入口。
- `test_training_smoke.py`、`test_matai_small_sample.py`：根目录测试脚本参数入口。

## 常用参数

- `--use_hpec_module4`：是否启用模块 4，启用时需要 `--use_hgcn_module3 1`。
- `--hpec_prototype_radius`：类别 prototype 半径，默认 `0.3`。
- `--hpec_cone_k`：HPEC cone aperture 的 `K` 参数，默认 `0.1`。
- `--hpec_margin`：非真实类别 energy 的 margin，默认 `1.0`。
- `--hpec_trainable_prototypes`：prototype 是否参与训练，默认 `0`。
- `--hpec_init_steps`：hyperspherical separation 初始化步数。
- `--hpec_eps`：数值稳定用 eps。

根目录测试脚本使用横线风格参数，例如：

```bash
python test_training_smoke.py --use-hpec-module4 1 --hpec-init-steps 50
```

`run_cv.py` 使用下划线风格参数，例如：

```bash
python run_cv.py --model S-DeCI --use_hgcn_module3 1 --use_hpec_module4 1
```

## 可视化

显式设置 `visualize_causal=1` 后，每个 fold 训练完成会保存 train/test 中间量 heatmap，并在最终 epoch 后保存 train/test 联合 t-SNE。

模块 4 新增可视化内容包括：

- HPEC prototypes
- HPEC angle matrix
- HPEC psi/aperture
- HPEC energy matrix
- `softmax(-energy)` probability，仅用于 energy-only 诊断或实验路径
- predicted labels
- ground truth labels

测试集真实 label 只用于 forward 之后绘图标注，不会作为模型输入。

## 回退方式

设置 `--use_hpec_module4 0` 即可回退到模块 3 的 `logmap0(z_global)` 线性分类头；若同时设置 `--use_hgcn_module3 0`，则回退到原 Cycle/seasonal logits 分类路径。
