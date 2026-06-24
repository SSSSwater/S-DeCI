# S-DeCI 多原型 HPEC 实现说明

本文档记录 `S-DeCI` 模块 4 从每类单个 HPEC prototype 扩展到每类多个 prototype 的实现方案。原始 `docs/新模块设计.md` 作为初始参考，不在本变更中直接修改。

## 目标

- 每个类别维护 `hpec_prototypes_per_class` 个 prototype，用于表达类内多样性。
- 保留 HPEC energy 分类路径，并将多个 prototype 的 energy 聚合为类别级 energy。
- 引入 prototype-related loss：`L_mle`、`L_pcl`、`L_pal`。
- 在 heatmap 和最终 epoch t-SNE 中显示多 prototype。

## 关键参数

- `hpec_prototypes_per_class`：每个类别的 prototype 数量。设置为 `1` 时退化为接近旧版单 prototype 行为。
- `hpec_proto_temperature`：prototype 相似度和 soft-min 聚合使用的温度。
- `lambda_hpec_mle`：`L_mle` 权重。
- `lambda_hpec_pcl`：`L_pcl` 权重。
- `lambda_hpec_pal`：`L_pal` 权重。

## Loss 构成

启用模块 4 后，总损失保持原联合训练结构，并可额外加入多 prototype loss：

```text
Loss_total =
  Loss_HPEC
  + alpha * Loss_Recon
  + lambda * Loss_DAG
  + gamma * L1
  + lambda_hpec_mle * L_mle
  + lambda_hpec_pcl * L_pcl
  + lambda_hpec_pal * L_pal
```

其中：

- `Loss_HPEC`：类别级 energy loss，类别级 energy 由多个 prototype-level energy 聚合得到。
- `L_mle`：让样本更匹配真实类别的 prototype 分布，同时远离异类 prototype。
- `L_pcl`：约束 prototype 之间的类间结构，并避免不同类别 prototype 混在一起。
- `L_pal`：让样本靠近真实类别下最相似的 prototype。

## 可视化

启用 `visualize_causal` 后：

- heatmap 中显示多 prototype、prototype-level energy、类别级 energy。
- 最终 epoch t-SNE 中绘制 train/test 样本和所有 prototype 点。
- prototype 点使用星形 marker，颜色与类别对应。

## 回滚

若多 prototype 训练不稳定，可使用：

```bash
--hpec_prototypes_per_class 1 --lambda_hpec_mle 0 --lambda_hpec_pcl 0 --lambda_hpec_pal 0
```

这会回退到接近当前单 prototype HPEC 的行为。
