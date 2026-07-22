## Context

当前 S-DeCI 的模块 2 已经采用 temporal SEM/NTS-NOTEARS 思路：使用时间序列历史窗口预测未来时间点，并输出 `A_lag`、`A0`、`A_effective` 等图结构。这个方向可以利用时间先后关系减少静态图学习中的 Markov equivalence class 问题，但现有边学习器偏局部线性，对复杂脑区交互、不同滞后阶数和多种功能网络模式的表达能力有限。

本变更在现有模块 2 框架上引入多头注意力，但不把 raw attention 直接当作因果矩阵。新的模块 2 使用多头 lag-window cross-attention 作为候选影响建模器，再通过结构门控、时序预测损失和稀疏/平滑/无环约束得到稳定的显式因果图。

`A0` 在该设计中有明确边界：它表示同时间片残余依赖，用于吸收无法由过去时间窗解释的同步相关或残差耦合，并作为 DAGMA/NOTEARS 无环约束的承载对象。`A0` 不是跨时间主因果图，也不应默认作为模块 3 的分类 adjacency。模块 3 的主图仍应来自 `A_lag.mean(dim=0)` 或其与样本相关矩阵融合后的 `A_cls`。

## Goals / Non-Goals

**Goals:**

- 在模块 2 中新增 `Attention-guided Temporal NTS-NOTEARS` 方法。
- 保留当前 temporal NTS-NOTEARS 的输入输出语义，降低对模块 3/4 和训练流程的影响。
- 使用多头 attention 表达不同 lag、不同脑区交互模式，但通过结构门控得到稳定可解释的 `A_lag`。
- 保留 `A0` 作为同时间片残余依赖与 DAG 约束载体，避免把同步残差误塞进跨时间因果图。
- 继续使用简洁损失：预测、稀疏、lag 平滑、`A0` 无环。
- 提供可视化和日志诊断，帮助判断 attention-guided 因果图是否有效。

**Non-Goals:**

- 不引入真实因果矩阵监督；合成数据中的真实图只用于训练后评估。
- 不把 raw attention weight 直接解释为因果矩阵。
- 不改动模块 3/4 的主要接口。
- 不新增大规模 Transformer over `T*N` token，避免训练成本和过拟合进一步上升。
- 不修改原始 `doc/` 参考文档；项目更新说明应另建文档。

## Decisions

### Decision 1: 使用 lag-window cross-attention，而不是全局 Transformer

模块 2 的目标是学习 `x_j(t-l) -> x_i(t)`，因此 attention 只在历史 lag window 到当前目标节点之间发生。对每个目标节点 `i`，query 来自目标节点自身历史摘要，key/value 来自所有源节点在不同 lag 上的历史值。

备选方案是把所有 `T*N` token 输入完整 Transformer。该方案表达能力更强，但训练成本高、注意力图更难聚合成稳定的脑区 adjacency，也更容易在 MDD/ABIDE 这类样本量下过拟合。因此第一版采用受约束的 lag-window attention。

### Decision 2: 使用结构门控 `G_lag` 聚合 attention

多头 attention 输出的是 batch/time/head 相关的动态权重，不能直接作为稳定因果图。新增可学习结构门控：

```text
G_lag[h, l, parent, child] = sigmoid(theta[h, l, parent, child])
```

有效边权由 `attention * G_lag` 得到，再对 batch、time、head 聚合为：

```text
A_lag[l, parent, child] = mean_{batch,time,head}(attention * G_lag)
```

这样 `attention` 负责动态预测分配，`G_lag` 负责稳定结构选择。

具体地，attention 公式约定为：

```text
score_{h,l,parent->child}(t)
  = < q_{h,child}(t), k_{h,l,parent}(t) > / sqrt(d_h)

attn_{h,l,parent->child}(t)
  = softmax_parent(score_{h,l,parent->child}(t))

gate_{h,l,parent->child}
  = sigmoid(theta_{h,l,parent->child})

edge_{h,l,parent->child}(t)
  = attn_{h,l,parent->child}(t) * gate_{h,l,parent->child}

A_lag[l,parent,child]
  = mean_{batch,time,head}(edge_{h,l,parent->child}(t))
```

其中：

- `q_{h,child}(t)` 来自目标 child 节点自身历史窗口的摘要，而不是来自未来真值。
- `k_{h,l,parent}(t)` 和 `v_{h,l,parent}(t)` 来自 parent 节点在第 `l+1` 个 lag 的历史值。
- `softmax_parent` 表示对所有源节点 parent 归一化，从而表达“预测当前 child 时更依赖哪些历史 source”。
- `A_lag[parent, child]` 的方向语义与现有项目保持一致。

该公式的设计目的不是让 raw attention 直接充当因果图，而是让 attention 先描述动态依赖，再由结构门控把它沉淀为稳定图。

### Decision 3: `A_lag` 不施加 DAG，`A0` 施加 DAG

`A_lag` 表示过去到未来的跨时间边，方向由时间箭头固定，因此不对 `A_lag` 强制 DAG。对它施加 L1 稀疏和 lag smoothness 即可。

`A0` 表示同时间片残余依赖，没有时间箭头提供方向，因此继续使用 DAGMA/NOTEARS 式无环约束。它的作用是吸收同步残差，减少 `A_lag` 被迫解释所有即时相关的压力。

### Decision 4: 预测损失仍是模块 2 主监督

模块 2 的核心监督仍是未来时间点预测：

```text
L_module2 =
  lambda_pred * L_pred
+ lambda_sparse * L1(A_lag, A0, G_lag)
+ lambda_smooth * smooth(A_lag)
+ lambda_dag * h(A0)
```

不新增复杂 counterfactual、prototype 或对比损失，避免再次出现损失过多、训练可拟合但泛化弱的问题。

### Decision 5: 下游图继续兼容现有模块 3

模块 2 输出仍保持 `a0`、`a_lag`、`a_shared`、`a_effective`、`dag_metadata`。模块 3 默认使用：

```text
learned_graph = A_lag.mean(dim=0)
A_cls = (1 - blend) * learned_graph + blend * sample_corr
```

当 `classification_graph_source == "learned"` 时直接使用 learned graph；当 `classification_graph_source == "sample_correlation"` 时继续使用样本相关矩阵。

### Decision 6: 归因校正作为诊断优先，不作为第一版训练损失

可以通过 `|d x_hat_i(t) / d x_j(t-l)|` 得到 gradient attribution 图，用于训练后诊断或可选导出 `A_attr`。第一版不把 attribution 加入 loss，避免新增训练不稳定因素。

## Risks / Trade-offs

- [Risk] attention 表达能力增强后可能更容易过拟合  
  → Mitigation: 限制 head 数和 head_dim，保留 dropout、稀疏门控和 lag smoothness，并用完整 5-fold 结果判断是否默认启用。

- [Risk] raw attention 被误解为因果图  
  → Mitigation: 文档、变量命名和可视化中区分 `attention_map`、`G_lag`、`A_lag`；下游只使用聚合后的 `A_lag` 或 `A_cls`。

- [Risk] `A0` 被误用于模块 3 分类图  
  → Mitigation: 规范中明确 `A0` 只作为同时间残余依赖和 DAG 约束对象，模块 3 默认使用 `A_lag.mean(dim=0)`。

- [Risk] 参数入口变多，影响日常训练可读性  
  → Mitigation: 只暴露少量参数：方法名、head 数、head_dim、attention dropout；损失权重复用现有 temporal 模块 2 权重。

- [Risk] 新方法效果不稳定  
  → Mitigation: 保留当前 temporal NTS-NOTEARS 路径，训练脚本允许切换；若完整验证不如当前默认路径，回滚默认方法即可。

## Migration Plan

1. 新增 attention-guided temporal causal learner，保持和 `TemporalSEMCausalLearner` 兼容的输出结构。
2. 在 `S_DeCI.py` 中增加方法选择，将 `causal_graph_method == "attn_nts_notears"` 或等价参数映射到新 learner。
3. 更新训练入口，暴露少量 attention 参数，并保持当前默认配置可继续运行。
4. 更新可视化和日志字段，输出 `A_lag`、`A_lag_mean`、`A0`、`A_cls` 与 attention/gate 诊断。
5. 运行低预算 smoke、MDD 1 fold 快速训练、MDD 5 fold 对比实验。
6. 回滚时将默认方法切回现有 `nts_notears`，并移除新 learner 的调用入口；由于接口兼容，模块 3/4 不需要回滚。

## Open Questions

- 第一版是否默认启用 `attn_nts_notears`，还是先作为可选实验方法保留？建议先可选，完整 5-fold 有正效果后再设为默认。
- 是否在训练后导出 gradient attribution 图作为正式诊断产物？建议第一版实现可选导出，不参与 loss。
- `A0` 是否参与最终可视化中的图对比排序？建议可视化，但不参与模块 3 默认分类图。
