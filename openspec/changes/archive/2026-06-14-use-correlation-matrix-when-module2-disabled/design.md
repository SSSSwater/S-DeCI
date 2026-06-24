## Context

`S-DeCI` 当前已有模块 1、模块 2、模块 3 和模块 4。模块 3 的 HGCN readout 需要 adjacency 才能执行图传播：模块 2 开启时 adjacency 来自 `A_learned`；但模块 2 关闭时，现有模型没有图输入来源。

数据集中已经存在样本级相关系数矩阵 `.mat` 文件，例如 Abide 下的 `sub-control50030_AAL116_correlation_matrix.mat`。用户也明确指出后续数据中可能存在 `sub-xxx_xxx_features_sub_correlation_matrix.mat` 这类命名。因此本次设计应按“同一 subject 对应同一 correlation matrix”来解析文件，而不是硬编码某一个文件名。

## Goals / Non-Goals

**Goals:**

- 在 `use_causal_module2=0` 且 `use_hgcn_module3=1` 时，让模块 3 使用样本相关系数矩阵作为 adjacency。
- 数据加载器在需要时返回 `(x_enc, label, correlation_matrix)`，训练/验证/可视化流程能够兼容两字段和三字段 batch。
- 模块 3 支持 batch adjacency `[B, N, N]`，并保持原有全局 adjacency `[N, N]` 兼容。
- 相关矩阵文件缺失、shape 不匹配或 `.mat` key 不可用时，给出清晰错误。
- 模块 2 开启时不改变现有因果学习路径。

**Non-Goals:**

- 不重新设计模块 2 因果学习方法。
- 不在本次变更中修改原始 `docs/新模块设计.md`。
- 不改变 `models/DeCI.py`。
- 不把相关系数矩阵当作真实因果矩阵监督模块 2。
- 不在模块 2 开启时混合使用 `A_learned` 和 sample correlation adjacency，除非后续单独变更要求。

## Decisions

### 1. 数据加载阶段返回样本相关矩阵

新增配置建议为 `use_sample_correlation_when_module2_disabled`，默认由 `use_causal_module2=0 && use_hgcn_module3=1` 推导为开启，也允许用户显式关闭。

当该配置开启时，Dataset 在加载时间序列样本时同时寻找对应相关矩阵文件，并缓存为 `self.correlation_matrices`。`__getitem__` 返回三元组：

```python
(time_series, label, correlation_matrix)
```

如果配置未开启，保持原有二元组：

```python
(time_series, label)
```

选择原因：在数据层完成 subject 对齐最可靠，训练循环只负责搬运 batch，不需要猜测文件路径。

### 2. 文件名解析按 subject 和 protocol 兼容多种模式

相关矩阵解析规则：

- 从时间序列文件所在目录优先查找同一 subject、同一 protocol 的 correlation matrix。
- 支持当前数据集的 `sub-xxx_<protocol>_correlation_matrix.mat`。
- 支持用户描述的 `sub-xxx_xxx_features_sub_correlation_matrix.mat`。
- 若存在多个候选，优先选择 protocol 明确匹配的文件。

`.mat` 内容优先读取 `data` key；如不存在，可尝试常见候选 key，并在失败时输出文件路径和可用 keys。

### 3. 训练循环兼容二元组与三元组 batch

`exp/exp_classification_CV.py` 增加 batch unpack helper：

```python
x_enc, label, corr = unpack_batch(batch)
y_hat = model(x_enc, correlation_matrix=corr)
```

当 batch 没有 `corr` 时，继续调用 `model(x_enc)`。

选择原因：改动集中，避免复制训练、验证、可视化和 t-SNE 多处 batch 解包逻辑。

### 4. 模块 3 支持 batch adjacency

`Module3HGCNReadout.forward(cycle_features, adjacency)` 需要支持：

- `[N, N]`：所有样本共享同一图，保持现有模块 2 路径。
- `[B, N, N]`：每个样本使用自己的相关矩阵图。

HGCN 聚合可以通过 `torch.einsum` 分支处理：

```python
# 全局图
agg = torch.einsum("ij,bjd->bid", adjacency.T, features)

# batch 图
agg = torch.einsum("bij,bjd->bid", adjacency.transpose(1, 2), features)
```

相关矩阵输入需要做非负化、可选绝对值、去 NaN、self-loop 和归一化。建议新增参数 `sample_correlation_mode`，默认 `abs`，表示把正负相关都视作连接强度。

### 5. S-DeCI forward 增加可选 correlation_matrix

`S-DeCI.forward()` 增加可选参数：

```python
forward(self, x_enc, correlation_matrix=None)
```

当模块 2 开启时忽略该参数并使用 `A_learned`。当模块 2 关闭且模块 3 开启时，必须提供 `correlation_matrix`，否则抛出清晰错误。

模块 2 关闭路径不产生 causal auxiliary loss，`get_aux_loss()` 返回 `None`，训练循环只使用分类 loss。

## Risks / Trade-offs

- [Risk] 相关矩阵文件命名在不同数据集间不完全一致 → 通过多候选解析、protocol 优先匹配和清晰错误提示缓解。
- [Risk] batch adjacency 增加 HGCN 计算和显存开销 → 仅在模块 2 关闭回退路径启用，且矩阵规模为 116x116 可控。
- [Risk] 相关矩阵含负值、NaN 或非对称值 → 默认 `abs + nan_to_num + clamp_min`，并在文档中说明这是图连接强度而非因果方向。
- [Risk] 数据加载返回三元组可能影响旧训练循环 → 通过统一 batch unpack helper 兼容二元组和三元组。

## Migration Plan

1. 扩展数据加载器，按需加载和返回样本相关矩阵。
2. 扩展 collate 和训练/验证/可视化/t-SNE 流程，兼容二元组与三元组 batch。
3. 扩展 `S-DeCI.forward()` 和模块 3 HGCN，支持 sample-level adjacency。
4. 添加训练入口参数和根目录测试脚本参数。
5. 新增说明文档，说明模块 2 关闭路径的调用方式和文件命名要求。
6. 运行低预算训练，覆盖 `use_causal_module2=0,use_hgcn_module3=1` 路径。

回滚方式：保持 `use_causal_module2=1`，或关闭 `use_sample_correlation_when_module2_disabled`；必要时删除数据加载三元组和 batch adjacency 分支。

## Open Questions

- 用户后续数据中 `sub-xxx_xxx_features_sub_correlation_matrix.mat` 的 `.mat` key 是否仍为 `data`？首版按 `data` 优先并提供 key fallback。
- 相关矩阵负值默认取 `abs` 是否符合所有实验？首版暴露 `sample_correlation_mode`，支持 `abs`、`positive`、`raw`。
