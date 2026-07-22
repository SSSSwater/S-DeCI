# S-DeCI 多原型 HPEC 实现说明

本文档记录 `S-DeCI` 模块 4 从每类单个 HPEC prototype 扩展到每类多个 prototype 的实现方案。原始 `docs/新模块设计.md` 作为初始参考，不在本变更中直接修改。

## 目标

- 每个类别维护 `hpec_prototypes_per_class` 个 prototype，用于表达类内多样性。
- 保留 HPEC energy 分类路径，并将多个 prototype 的 energy 聚合为类别级 energy。
- 保留当前实验中仍有效的 HPEC energy / teacher distill / 半径约束 / prototype separation 诊断与损失。
- 早期设计中的 `L_mle`、`L_pcl`、`L_pal` 已在后续实验中移除，不再作为当前主路径。
- 在 heatmap 和最终 epoch t-SNE 中显示多 prototype。

## 关键参数

- `hpec_prototypes_per_class`：每个类别的 prototype 数量。设置为 `1` 时退化为接近旧版单 prototype 行为。
- `hpec_proto_temperature`：prototype 相似度和 soft-min 聚合使用的温度。
- `hpec_energy_loss_weight`：HPEC energy auxiliary loss 权重。
- `hpec_teacher_distill_weight`：使用稳定 teacher logits 校准 HPEC 能量方向的 distill 权重。
- `hpec_z_radius_loss_weight` / `hpec_z_radius_target`：约束双曲表示半径，避免 `z_global` 过度塌缩或过度靠近边界。
- `hpec_prototype_separation_loss_weight` / `hpec_prototype_separation_max_cos`：在切空间约束 prototype 方向，减少 prototype 挤在一起。
- `hpec_trainable_prototypes`：是否让 prototype 通过梯度更新。
- `hpec_use_sinkhorn_ema`、`hpec_ema_alpha`：可选的 EMA / Sinkhorn prototype 更新控制参数。

## Loss 构成

启用模块 4 后，总损失保持联合训练结构，并可按权重加入当前保留的 HPEC auxiliary loss：

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{HPEC\_final\_CE}}
+
\mathcal{L}_{\mathrm{module2}}
+
\lambda_{\mathrm{energy}}\mathcal{L}_{\mathrm{HPEC\_energy}}
+
\lambda_{\mathrm{distill}}\mathcal{L}_{\mathrm{teacher\_distill}}
+
\lambda_z\mathcal{L}_{z\_\mathrm{radius}}
+
\lambda_{\mathrm{sep}}\mathcal{L}_{\mathrm{prototype\_separation}}.
$$

其中模块 2 项为：

$$
\mathcal{L}_{\mathrm{module2}}
=
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0).
$$

其中：

- $\mathcal{L}_{\mathrm{HPEC\_final\_CE}}$：最终 HPEC logits 的分类交叉熵。
- $\mathcal{L}_{\mathrm{HPEC\_energy}}$：类别级 energy loss，类别级 energy 由多个 prototype-level energy 聚合得到。
- $\mathcal{L}_{\mathrm{teacher\_distill}}$：让 HPEC energy logits 与稳定 teacher logits 的方向保持一致，降低训练集追噪声。
- $\mathcal{L}_{z\_\mathrm{radius}}$：约束 `z_global` 半径。
- $\mathcal{L}_{\mathrm{prototype\_separation}}$：约束 prototype 方向相似度不超过设定阈值。

## 可视化

启用 `visualize_causal` 后：

- heatmap 中显示多 prototype、prototype-level energy、类别级 energy。
- 最终 epoch t-SNE 中绘制 train/test 样本和所有 prototype 点。
- prototype 点使用星形 marker，颜色与类别对应。

## 回滚

若多 prototype 训练不稳定，可使用：

```bash
--hpec_prototypes_per_class 1 --hpec_energy_loss_weight 0 --hpec_teacher_distill_weight 0 --hpec_z_radius_loss_weight 0 --hpec_prototype_separation_loss_weight 0
```

这会回退到接近当前单 prototype HPEC 的行为。
