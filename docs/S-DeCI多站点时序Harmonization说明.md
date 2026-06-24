# S-DeCI 多站点输入时序 Harmonization 说明

## 背景

本项目的论文中心是消除多站点 fMRI 数据偏移。多站点数据常见偏移来自扫描仪、采集协议、站点预处理差异和被试组成差异。如果直接把所有站点时序输入模型，模型容易学习站点特征，而不是 MDD/HC 相关神经特征。

## 文献动机

多站点 fMRI harmonization 常见路线包括：

- `ComBat` / empirical Bayes harmonization：常用于去除 site/batch effect。Fortin 等人在多站点皮层厚度 harmonization 中说明它可以减少站点相关技术差异并尽量保留生物学差异；Yu 等人在多站点 fMRI functional connectivity 上进一步验证了 ComBat 对站点效应校正的作用。
- site regression / residualization：用训练数据估计站点偏移，再对特征做残差化。
- domain-adversarial / site-invariant representation：Ganin 等人的 DANN 思路通过 gradient reversal layer 让表征难以预测 domain/site，从而鼓励学习 domain-invariant feature。
- domain generalization / shared-space alignment：学习跨站点共享表示，减少模型依赖站点特异模式。

当前先实现输入时序阶段的轻量版本：`site_zscore`。它不需要额外站点标签文件，直接从 MDD 文件名中的 `sXX` 解析站点。

## 当前实现

新增参数：

```bash
--time-series-harmonization none
--time-series-harmonization site_zscore
--site-harmonization-min-samples 2
```

实现位置：

- `data_provider/data_loader_CV.py`
  - 从文件名解析 `site_id`，例如 `sub-control_s17_1_0028_AAL116_features_timeseries.mat` 解析为 `s17`。
  - Dataset 保存 `sample_paths` 和 `site_ids`。
- `data_provider/data_factory_CV.py`
  - 每个 fold 内只使用训练集估计每个站点、每个 ROI 的均值和标准差。
  - 训练集和验证集都使用训练集估计出的统计量做 transform。
  - 若某站点训练样本数不足，则回退到训练集全局统计。

## 避免数据泄漏

`site_zscore` 必须在交叉验证 fold 内 fit：

- train fold：fit site mean/std，并 transform train。
- test fold：只 transform，不参与 fit。

这样不会使用测试集站点分布信息来帮助训练。

## Site-adversarial head

当前已新增 `site-adversarial head`：

```bash
--use-site-adversarial 0
--lambda-site-adversarial 0.02
--site-grl-lambda 1.0
--site-adversarial-dropout 0.1
```

注意：该分支目前作为消融能力保留，但默认不启用。MDD 1 fold / 20 epoch 的当前实测中，`site_zscore` 可以保持 macro F1 并小幅提升 AUC；而直接启用 site-adversarial head 会降低 macro F1/AUC。一个合理解释是某些 fold 中 site 与 diagnosis label 存在混杂，强行去站点可能同时抹掉诊断相关信号。

其逻辑是：

1. 数据集从文件名解析 `site_id`，并在开启 `use_site_adversarial` 时把 `site_label` 放入 batch。
2. `S-DeCI` 在模块 3 的 `z_tangent` 后接一个 site classifier。
3. site classifier 前使用 `gradient reversal layer`。
4. site classifier 自身被训练去预测站点，但主干网络收到反向梯度，因而被鼓励学习更难预测站点的 `z_global/z_tangent`。

这属于 domain-adversarial / site-invariant representation 思路，和输入阶段 `site_zscore` 互补：

- `site_zscore` 减少输入时序的站点均值/尺度偏移。
- `site-adversarial head` 减少高层表示中的站点可预测性。

## 当前默认建议

用于 MDD 测试时，默认推荐：

```bash
--time-series-harmonization site_zscore
--use-site-adversarial 0
```

如果需要尝试站点对抗，请先打开：

```bash
--print-data-info 1
```

训练入口会打印每个 fold 的 `site × label` 计数表。如果某些站点几乎只包含某一类标签，说明站点和诊断标签高度相关，此时启用站点对抗可能会削弱分类信号。

## 后续方向

1. 在 FC 或模块 1 输出特征上实现 `ComBat` 风格 harmonization。
2. 在 HPEC prototype 中加入 site-balanced prototype 初始化，避免某站点主导类别 prototype。
3. 尝试更温和的特征对齐策略，例如 site-balanced sampler、class-aware site reweighting、CORAL/MMD alignment，避免在 site-label 混杂较强时过度擦除诊断信号。

## 参考文献

- Fortin et al., Harmonization of cortical thickness measurements across scanners and sites, NeuroImage, 2018.
- Yu et al., Statistical harmonization corrects site effects in functional connectivity measurements from multi-site fMRI data, Human Brain Mapping, 2018.
- Ganin et al., Domain-Adversarial Training of Neural Networks, JMLR, 2016.
