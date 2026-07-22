## Purpose

定义模块 4 HPEC 的每类多 prototype 分类能力：用多个 prototype 表达同一类别内部的连接模式差异，并通过 prototype-level energy、相似度、可选 distillation 校准、半径约束和 prototype separation 支持诊断与训练。

## Requirements

### Requirement: 每类多 prototype 表示

系统 SHALL 将 HPEC 模块 4 的类别原型从每类单个 prototype 扩展为每类多个 prototype，用于表达同一诊断类别内部的多种连接模式。

#### Scenario: 初始化多 prototype

- **GIVEN** `use_hpec_module4 == 1`
- **AND** `hpec_prototypes_per_class > 1`
- **WHEN** 模型初始化 HPEC 模块 4
- **THEN** prototype 张量 MUST 使用形状 `[classes, hpec_prototypes_per_class, hgcn_hidden_dim]`
- **AND** 每个 prototype MUST 被投影或约束在 HPEC 使用的 Poincare Ball 有效区域内

#### Scenario: 单 prototype 回退

- **GIVEN** `use_hpec_module4 == 1`
- **AND** `hpec_prototypes_per_class == 1`
- **WHEN** 模型执行 forward 和 loss 计算
- **THEN** 行为 MUST 退化为接近当前每类单 prototype 的 HPEC 分类路径
- **AND** 该回退路径 MUST 不要求用户修改模型源码

### Requirement: 多 prototype 类别能量聚合

系统 SHALL 对每个样本与每个类别下的多个 prototype 计算 prototype-level energy，并将其聚合为类别级 energy。默认 `hgcn_hpec` 主路线中，类别级 energy 先形成 HPEC evidence 增量，再与欧氏局部结构 logits 融合用于预测和指标；显式 energy-only 实验路径 MAY 直接用类别级 energy 预测。

#### Scenario: 计算 prototype-level energy

- **GIVEN** `z_global` 的形状为 `[B, hgcn_hidden_dim]`
- **AND** prototype 张量形状为 `[classes, K, hgcn_hidden_dim]`
- **WHEN** HPEC 模块 4 执行 forward
- **THEN** 系统 MUST 计算形状为 `[B, classes, K]` 的 prototype-level energy 或等价中间量
- **AND** 系统 MUST 保留该中间量供诊断和可视化使用

#### Scenario: 聚合类别级 energy

- **GIVEN** prototype-level energy 已计算完成
- **WHEN** 模型需要输出分类结果
- **THEN** 系统 MUST 将 `[B, classes, K]` 聚合为 `[B, classes]` 的类别级 energy，公式可写作：

$$
q_{b,k,m}
=
\frac{\exp(-E_{b,k,m}/\tau_p)}
{\sum_{r=1}^{P}\exp(-E_{b,k,r}/\tau_p)},
\qquad
E_{b,k}
=
\sum_{m=1}^{P}q_{b,k,m}E_{b,k,m}.
$$

- **AND** 默认主路线 MUST 通过如下融合 logits 得到预测类别：

$$
\ell^{\mathrm{hyper}}_{b,k}
=
-E_{b,k}
+
\lambda_{\mathrm{evi}}\bar{s}_{b,k},
\qquad
\hat{Y}_b
=
\ell^{\mathrm{base}}_b
+
\lambda_{\mathrm{hyp}}g_b r^{\mathrm{hyper}}_b,
$$

$$
\hat{y}_b
=
\operatorname*{argmax}_{k}\hat{Y}_{b,k}.
$$

- **AND** 显式 energy-only 实验路径 MAY 通过类别级 energy 的 `argmin` 或等价 energy-based 规则得到预测类别：

$$
\hat{y}_b
=
\operatorname*{argmin}_{k}E_{b,k}.
$$

- **AND** `S-DeCI.forward()` MUST 保持返回与当前训练指标兼容的 score/logit 张量
- **AND** 设计原因 MUST 写明：softmin 聚合保留“低 energy 更匹配类别”的语义；默认融合 logits 保证训练目标、指标和 HPEC evidence 增量 的使用方式一致，避免 energy-only 在小样本中校准不稳。

### Requirement: HPEC energy 与 prototype 正则

系统 SHALL 使用最终融合 logits 分类作为模块 4 默认主监督，并让 HPEC energy/prototype evidence 以双曲证据增量和可选辅助约束的方式参与训练。系统 MAY 支持当前实验证明仍保留的 distillation 校准、表示半径约束和 prototype separation 约束；已废弃的 `L_mle`、`L_pcl`、`L_pal` 不应作为当前主路径要求。

#### Scenario: 计算 HPEC final CE 与 energy loss

- **GIVEN** HPEC 模块 4 已完成 forward
- **AND** 存在训练标签
- **WHEN** 训练流程计算模块 4 primary loss
- **THEN** 系统 MUST 使用 `hpec_final_ce_loss` 监督最终融合 logits：

$$
\mathcal{L}_{\mathrm{final\_ce}}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{
\exp(\hat{Y}_{b,y_b})
}{
\sum_{k=1}^{K}\exp(\hat{Y}_{b,k})
}.
$$

- **AND** 当 `hpec_energy_loss_weight > 0` 时 MUST 加入 HPEC energy loss 的加权贡献
- **AND** HPEC energy loss MUST 能参与 PyTorch autograd 反向传播
- **AND** 设计原因 MUST 写明：最终 CE 对齐模型真实输出，energy loss 只约束双曲原型语义，不单独决定默认预测。

#### Scenario: 可选 distillation 校准 HPEC evidence

- **GIVEN** `hpec_teacher_distill_weight > 0`
- **AND** 存在欧氏局部结构 logits 或等价参考 logits
- **WHEN** 训练流程计算模块 4 loss
- **THEN** 系统 MUST 计算 HPEC evidence logits 与参考 logits 的 distillation loss，例如：

$$
\mathcal{L}_{\mathrm{distill}}
=
T^2\,
\operatorname{KL}
\left(
\operatorname{softmax}\left(\frac{\ell^{\mathrm{ref}}}{T}\right)
\middle\|
\operatorname{softmax}\left(\frac{\ell^{\mathrm{hpec}}}{T}\right)
\right).
$$

- **AND** 该 loss MUST 只作为稳定 HPEC 能量方向的辅助项，不替代 HPEC energy 分类路径
- **AND** 设计原因 MUST 写明：distillation 校准只用于消融测试小样本 logits 稳定性，默认权重为 0；它不是默认主监督，也不是让模块 4 退化为复制普通分类器。

#### Scenario: 约束双曲表示半径

- **GIVEN** `hpec_z_radius_loss_weight > 0`
- **WHEN** 模块 3 输出 `z_global`
- **THEN** 系统 MUST 计算 `z_global` 半径与 `hpec_z_radius_target` 的偏差，例如：

$$
\mathcal{L}_{z\_\mathrm{radius}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left(
\|z_b\|_2-r_z
\right)^2.
$$

- **AND** 该加权项 MUST 能加入总 loss
- **AND** 设计原因 MUST 写明：Poincare 点若全部靠近原点，cone aperture 过宽；若贴近边界，数值不稳定，因此需要温和半径约束。

#### Scenario: 避免 prototype 方向坍缩

- **GIVEN** `hpec_prototype_separation_loss_weight > 0`
- **WHEN** HPEC prototype 位于 Poincare Ball 中
- **THEN** 系统 MUST 在切空间中计算 prototype 方向相似度：

$$
p^{\mathrm{tan}}_{k,m}
=
\log_0^c(p_{k,m}),
\qquad
\cos_{(k,m),(r,n)}
=
\frac{
(p^{\mathrm{tan}}_{k,m})^\top p^{\mathrm{tan}}_{r,n}
}{
\|p^{\mathrm{tan}}_{k,m}\|_2
\|p^{\mathrm{tan}}_{r,n}\|_2+\epsilon
}.
$$

- **AND** 当相似度超过 `hpec_prototype_separation_max_cos` 时 MUST 产生 separation penalty：

$$
\mathcal{L}_{\mathrm{proto\_sep}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}
\operatorname{ReLU}
\left(
|\cos_{a,b}|-\gamma_{\max}
\right)^2.
$$

- **AND** 该约束 MUST 保留类间可分性，同时不强制同类多个 prototype 完全重合
- **AND** 设计原因 MUST 写明：prototype 方向坍缩会使不同类别 energy 接近，切空间余弦分离可以直接约束原型方向而不破坏 Poincare 半径语义。

### Requirement: 多 prototype 中间量可诊断

系统 SHALL 缓存多 prototype HPEC 的关键中间量，便于分析原型分布、样本匹配关系和分类效果。

#### Scenario: 缓存多 prototype 输出

- **GIVEN** HPEC 模块 4 完成 forward
- **WHEN** 开发者读取模型缓存或可视化函数
- **THEN** 系统 MUST 能读取多 prototype 张量
- **AND** 系统 MUST 能读取 prototype-level energy 或相似度
- **AND** 系统 MUST 能读取类别级 energy、预测类别和 probability 或等价 score

#### Scenario: 缓存 HPEC loss 与 prototype 诊断

- **GIVEN** 训练流程计算了 HPEC final CE、energy loss、distillation 校准、半径约束或 prototype separation
- **WHEN** 开发者读取模型 loss 诊断量
- **THEN** 系统 MUST 能分别读取这些 loss 的未加权值
- **AND** 系统 MUST 能读取它们加入总 loss 后的加权贡献
- **AND** 系统 MUST 能读取 prototype 方向相似度、最大相似度或等价坍缩诊断

### Requirement: 可靠 TP EMA 作为多 prototype 更新候选

系统 SHALL 支持以可靠 true-positive 样本的独立 EMA 更新每类多个 prototype，并将 Sinkhorn 保留为 legacy 消融。

#### Scenario: 可靠样本分配与慢更新
- **GIVEN** `hpec_prototype_update_mode == "reliable_tp_ema"`
- **WHEN** 训练 batch 的 optimizer 更新完成
- **THEN** prototype MUST 只由预测正确、高置信度且可选视图一致的样本移动
- **AND** 同类样本 MUST 按最低 energy 分配到多个 prototype
- **AND** 该更新 MUST 在无梯度上下文中执行
