## Context

当前项目的 S-DeCI 已经形成四个模块的串联结构：模块 1 提取节点特征，模块 2 学习静态因果图，模块 3/4 使用图结构完成 HGCN/HPEC 分类。现有模块 2 主要基于 `[B, N, d_model]` 节点特征做重构式图学习；这对分类可用，但对 fMRI 的时间动态因果解释不够直接。

本次变更将模块 1 的可用输出扩展为节点时间序列，使模块 2 可以通过预测式 SEM 学习时间因果关系。`docs/新模块设计.md` 继续作为初始参考，不直接修改；本次实现需要新增一份项目当前设计说明。

## Goals / Non-Goals

**Goals:**

- 让 S-DeCI 模块 1 能向模块 2 提供节点级时间序列输出，而不只提供压缩后的节点特征。
- 新增 `TemporalSEMCausalLearner`，用历史窗口预测当前/下一步时间序列。
- 同时学习 `A0` 与 `A_lag`：`A0` 表示同一时间片的 contemporaneous DAG，`A_lag` 表示时间滞后有向影响。
- 将模块 2 损失改为预测式 SEM：预测误差、DAGMA 调度、稀疏、时间平滑、图稳定性和样本残差图约束。
- 收紧样本图自由度：样本级差异只能作为低秩、小幅度 residual graph。
- 在训练和验证中输出图学习诊断，不只输出 acc/AUC。
- 保留当前静态模块 2 作为 fallback，便于对照和回滚。

**Non-Goals:**

- 不在本变更中重写模块 3 HGCN 或模块 4 HPEC 的核心算法。
- 不修改 `docs/` 中已有初始参考文档。
- 不引入真实因果矩阵监督；真实图只用于合成实验后的评价和可视化。
- 不暴露过多低层超参数到 `run_cv.py`，只保留必要的高层开关。

## Decisions

### Decision 1: 模块 1 输出时间序列与节点特征并存

模块 1 不直接删除现有 `[B, N, d_model]` 节点特征输出，而是新增或缓存 `[B, T', N]` 或 `[B, T', N, d]` 的节点时间序列输出。模块 2 temporal SEM 优先使用时间序列，模块 3/4 仍可继续使用节点特征。

原因：模块 3/4 当前已稳定依赖节点特征，直接替换会扩大风险；时间因果学习需要保留时序口径。

备选方案：只让模块 1 输出时间序列，再由模块 2/3 自行聚合特征。该方案改动范围更大，容易破坏现有训练入口，暂不采用。

### Decision 2: 新增 temporal SEM learner，而不是强行改造当前 `CausalGraphLearner`

新增 `TemporalSEMCausalLearner`，负责：

- 输入节点时间序列。
- 生成 `A0: [N, N]` 和 `A_lag: [L, N, N]`。
- 用过去 `L` 个时间步预测目标时间步。
- 输出 `X_hat`、`A_shared`、`A_effective`、`A_delta` 和诊断信息。

原因：当前 `CausalGraphLearner` 是静态特征重构器，直接塞入 time-lag 逻辑会让接口混乱，也不利于回滚。

### Decision 3: `A0` 使用 DAGMA 调度，`A_lag` 不强制 DAG

`A0` 表示同一时间片内的结构关系，必须使用 DAGMA log-det 约束并执行调度。`A_lag` 表示从过去到未来的方向，时间箭头天然约束方向，不强制无环，但需要稀疏和平滑。

DAGMA 调度采用阶段式：

- warmup 阶段：以预测损失为主，弱 DAG。
- barrier 阶段：逐步增大 log-det DAG 权重。
- refine 阶段：增强稀疏与图稳定性，输出更清晰的 `A0`。

### Decision 4: 样本图采用低秩 residual graph

样本级图不再自由学习完整 `[B, N, N]` 矩阵，而是：

`A_delta_i = scale * low_rank(U_i @ V_i^T)`，并进行 off-diagonal mask、幅度裁剪、L1 和 deviation 正则。

原因：fMRI 样本数通常有限，自由样本图很容易记住训练标签。低秩、小幅度 residual 能表达个体差异，同时降低过拟合风险。

### Decision 5: 诊断指标成为训练输出的一部分

除了分类指标，训练/验证必须记录：

- temporal prediction loss。
- `A0` DAG penalty、log-det barrier 状态、无环程度。
- `A0/A_lag` 稀疏度、平均边权、方向性。
- 样本 residual graph 的均值、最大值、低秩约束 loss。
- fold 间 `A_shared` 相似度。
- top-k edge frequency / stability。
- `A0`、`A_lag`、`A_effective` 和 edge frequency heatmap。

这些指标用于判断“图是否稳定且可解释”，而不仅是分类是否过拟合。

## Risks / Trade-offs

- [Risk] 时间序列预测式 SEM 训练更慢。  
  → Mitigation: 先支持较小 `lag_order` 和低预算 smoke test，再扩展到全量训练。

- [Risk] 模块 1 输出时间序列后张量形状复杂，容易与现有 `[B, N, d]` 路径混淆。  
  → Mitigation: 明确缓存字段名，如 `latest_temporal_series`、`latest_node_features`，并在可视化标题中标注来源。

- [Risk] DAGMA 调度过强会牺牲预测能力，过弱会图不清晰。  
  → Mitigation: 使用阶段式调度并输出每阶段权重与 penalty，默认参数尽量保守。

- [Risk] 样本 residual graph 仍可能过拟合。  
  → Mitigation: 默认关闭或低幅度启用；启用时强制低秩、小幅度、强正则，并输出 residual 诊断。

- [Risk] 真实 fMRI 因果方向受 TR、混杂和预处理影响。  
  → Mitigation: 把 learned graph 作为模型学习到的有效连接/动态依赖图，避免在文档中承诺严格生物因果。

## Migration Plan

1. 新增 temporal SEM learner，不删除现有静态 `CausalGraphLearner`。
2. 修改 S-DeCI 模块 1 缓存节点时间序列，同时保留节点特征路径。
3. 增加 `causal_learning_target` 或等价高层开关：`static_feature` / `temporal_sem`。
4. 训练循环读取 temporal SEM 的 auxiliary loss 与诊断指标。
5. 增加低预算 smoke test、MDD 1 fold test、合成时间序列 SEM test。
6. 新增项目说明文档，记录当前实现与原始 `docs/` 的差异。

回滚方案：将 `causal_learning_target` 设为 `static_feature` 或关闭模块 2，即可回到当前静态模块 2 或相关矩阵 fallback 路径。

## Open Questions

- 模块 1 时间序列输出使用 seasonal 分支、residual 分支，还是二者组合，需要实现时基于 DeCI block 实际可获得的中间量确认。
- `A0` 与 `A_lag` 传递给模块 3 时，是求和、加权融合，还是只使用 `A0 + mean(A_lag)`，需要通过实验比较。
- 图稳定性指标的保存粒度是每 epoch、每 fold 结束，还是仅最终 checkpoint，需要结合训练耗时决定。
