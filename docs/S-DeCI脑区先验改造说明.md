# S-DeCI 脑区先验改造说明

## 背景

当前 S-DeCI 在 MDD 数据上容易出现训练集严重过拟合、测试集 `z_global` 难以分开的情况。仅增加 HGCN/HPEC 自由度会让模型更容易记住训练样本，因此本次改造引入 MDD 静息态 fMRI 文献中反复出现的脑网络先验，让模块 3 的 readout 更关注具有诊断意义的网络级信息。

## 文献线索

MDD 静息态 fMRI 和 functional connectivity 研究中较常见的异常网络包括：

- `DMN`：medial prefrontal cortex、PCC/precuneus、angular gyrus、medial temporal 等。
- `fronto-limbic / affective network`：OFC、ACC/MCC、insula、hippocampus、amygdala 等。
- `salience network`：anterior insula、ACC/MCC、SMA、thalamus 等。
- `cognitive control / frontoparietal network`：DLPFC、inferior frontal、parietal 区域等。
- `subcortical-thalamic / striatal loop`：caudate、putamen、pallidum、thalamus。
- 一些 whole-brain FC 分类研究也报告 visual、sensorimotor、cerebellum 与 MDD 分类有关。

这些线索来自 MDD 大样本 resting-state FC、dorsal nexus、whole-brain FC 分类和网络异常综述类研究。实现中没有把这些区域作为硬标签监督，而是作为轻量 readout prior。

## 当前实现

文件：`layers/hyperbolic_gcn_layer.py`

模块 3 的 readout 现在支持 `use_brain_network_prior`：

- `0`：使用当前较稳的低自由度图统计 readout，仅融合 `mean/std/max`。
- `1`：在 AAL116 下启用固定 ROI gating，将 MDD 文献相关网络的 ROI 表示轻微放大。

已定义的 AAL116 分组包括：

- `dmn`
- `fronto_limbic`
- `control`
- `salience`
- `subcortical`
- `sensorimotor`
- `visual`
- `cerebellum`

注意：第一次尝试的“可学习 network attention + network summary fusion”过拟合更严重，因此已改为固定轻量 gating，不增加可学习网络权重。

## 实测结果

测试设置均为 MDD、AAL116、`1 iteration / 1 fold / 20 epoch`，模块 1/2/3/4 均启用。

| 配置 | Accuracy | Macro F1 | ROC AUC | 结论 |
| --- | ---: | ---: | ---: | --- |
| 无脑区先验，低自由度统计 readout | 0.7250 | 0.6800 | 0.6506 | 当前综合最好 |
| 可学习脑网络先验 | 0.5000 | 0.4921 | 0.5618 | 明显过拟合，不建议 |
| 固定 ROI gating 脑区先验 | 0.7125 | 0.6690 | 0.6590 | AUC 略好，但 Macro F1 低于无先验 |

因此 `test_mdd_best_config.py` 默认保持 `use_brain_network_prior=0`，保留 `--use-brain-network-prior 1` 作为后续消融和继续改造入口。

## 后续建议

1. 不建议继续增加可学习 attention 自由度，当前数据规模下容易训练集 100% 而测试集混杂。
2. 更推荐将脑区先验用于：
   - 固定 ROI gating；
   - 网络级辅助可视化；
   - 网络内/网络间 FC 统计；
   - prototype 初始化时按网络 summary 而不是全局 `z_global` 初始化。
3. 若继续提升分类特征，下一步可以单独构造 network-level feature branch：每个网络输出一个 summary，再计算网络间关系矩阵，由 HPEC 或简单线性头融合，而不是只把所有 ROI 压成一个 `z_global`。
