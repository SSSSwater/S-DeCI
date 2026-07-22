## Context

当前 S-DeCI 模块二已经提供 `nts_notears` 与 `dag_sampling` 两条方法路径，并通过 reconstruction、DAG penalty 和 L1 sparsity 组成辅助 loss。实际训练中模块二输入通常为模块一输出的节点特征 `C: [B, N, d_model]`，缺少原始时间序列 lag 维度，因此它并不是严格的 NTS-NOTEARS，而是 feature-level 的可微 DAG learner。

项目近期在 MDD/ABIDE 训练中观察到因果矩阵变化较小、分类结果容易过拟合、模块二启用后未稳定优于相关矩阵 fallback。这说明当前模块二需要更稳定的图参数化、尺度控制、样本级差异表达和训练调度。

约束：
- 保留现有模块，不删除模块一、二、三、四。
- 保持 `S-DeCI` 的模块开关和现有 `sample correlation fallback` 兼容。
- 不直接修改 `docs/` 下初始参考文档，新增更新说明文档记录项目当前设计。
- 新增行为应可通过参数切回旧路径。

## Goals / Non-Goals

**Goals:**

- 明确模块二当前默认是 feature-level DAG learner，而不是严格 NTS-NOTEARS。
- 新增更适合 116 ROI 大图的 `dagma_logdet` DAG 约束或同等 log-det DAG penalty。
- 给模块二输入增加独立标准化，降低节点特征尺度对图学习的影响。
- 支持“共享图 + 样本残差图”的可选图结构，使不同样本可产生轻量差异化 adjacency。
- 引入可配置 loss 调度，让 reconstruction、DAG、L1、样本残差约束在训练中逐步生效。
- 扩展诊断与可视化，便于观察图是否真的在更新、是否有方向性、是否存在样本差异。

**Non-Goals:**

- 本次不完整实现严格的 temporal NTS-NOTEARS / DYNOTEARS lag 结构。
- 本次不改变数据集读取格式，不要求新增 HRF 或 fMRI 专用预处理。
- 本次不删除现有 `nts_notears`、`dag_sampling`、HGCN、HPEC 或 GCN fallback。
- 本次不强制启用样本残差图；它应是可配置能力。

## Decisions

### 1. 默认推荐 `dagma_logdet`，保留现有方法对照

模块二新增 `causal_graph_method=dagma_logdet`。该方法使用 log-det / M-matrix 风格无环约束，相比 `trace(matrix_exp(A*A))-N` 在较大图上更稳定，也比 hard DAG sampling 更连续。

备选方案：
- 继续默认 `nts_notears`：保留兼容，但大图下固定权重 DAG penalty 容易不充分。
- 默认 `dag_sampling`：天然拓扑顺序约束，但 hard permutation 会让优化更跳，适合作为对照。

### 2. 输入标准化放在 causal learner 内部

模块二在 `CausalGraphLearner.forward()` 内对输入 `C` 执行可配置归一化，例如 `layer_norm` 或 `node_feature_zscore`。这样模块二可以独立稳定，不依赖模块一输出尺度刚好合适。

备选方案：
- 在 `S_DeCI.forward()` 中标准化：实现简单，但会把模块二前处理散落到模型层。
- 不标准化：保留旧行为，但不利于因果图学习稳定。

### 3. 样本残差图只作为小幅修正

模块二保留全局共享图 `A_shared`，并可选从样本特征生成 `A_delta[B, N, N]`。最终图为：

`A_effective = clamp_or_sigmoid(A_shared + delta_scale * A_delta)`

其中 `A_delta` 必须对角线为 0，并受 L1、幅度或 deviation loss 约束。

理由：
- 共享图保留全体样本共同连接模式。
- 样本残差图提供标签相关差异表达空间。
- 强约束避免每个样本都学出任意图导致过拟合。

备选方案：
- 只学全局图：稳定但可能掩盖组间差异。
- 完全样本级图：表达力强但小样本 fMRI 极易过拟合。

### 4. 使用 loss 调度而不是固定权重

新增权重调度参数，例如：
- `causal_loss_schedule=constant|warmup|linear`
- `causal_dag_warmup_epochs`
- `causal_l1_warmup_epochs`
- `sample_graph_reg_warmup_epochs`

训练早期以 reconstruction / classification 稳定表征为主，中后期逐步增强 DAG 与 sparsity。该策略比固定很小的 `lambda_causal_dag` 更容易让图最终接近 DAG，同时减少早期图僵死。

备选方案：
- NOTEARS augmented Lagrangian 外循环：理论更完整，但需要较大训练流程改造。
- 固定权重：实现最简单，但已观察到效果不稳定。

### 5. 诊断输出成为模块二的一等能力

模块二应暴露：
- `A_shared`
- `A_delta` 或样本图统计
- `A_effective`
- `dag_penalty`
- `l1_loss`
- `sample_graph_l1_loss`
- `sample_graph_deviation_loss`
- adjacency 更新幅度和方向性指标

可视化应区分共享图和样本图，避免只看一张平均图误判“图没有变化”。

## Risks / Trade-offs

- [Risk] 样本残差图提升表达力但增加过拟合风险 → Mitigation: 默认关闭或小 `delta_scale`，增加 L1/deviation 约束，并通过 MDD 1fold 和全 kfold 对比。
- [Risk] `dagma_logdet` 约束实现不当会出现数值不稳定 → Mitigation: 使用 `sI - A*A`、安全 margin、eps、异常诊断，并保留旧约束回退。
- [Risk] 过多新参数让训练入口复杂 → Mitigation: 在 `run_cv.py` 中按模块二分组，提供合理默认值和独立测试脚本。
- [Risk] feature-level DAG 仍不是真正时间因果 → Mitigation: 文档中明确语义；后续如需严格 NTS/DYNOTEARS，再另起变更接入 lag temporal input。

## Migration Plan

1. 在 `layers/causal_graph_learner.py` 中加入 DAGMA/log-det 约束、输入标准化、样本残差图和扩展诊断。
2. 在 `models/S_DeCI.py` 中读取新的模块二输出，并将 `A_effective` 传给模块三或 GCN fallback。
3. 在训练流程中加入 loss 调度，使模块二辅助 loss 可按 epoch 动态权重计算。
4. 在 `run_cv.py` 和独立测试脚本中暴露新参数，默认值保持现有行为可回退。
5. 新增项目更新说明文档，记录修改后模块二设计。
6. 运行低预算 smoke、MDD 1fold、模块二合成数据训练检查。

回滚策略：
- 将 `causal_graph_method` 切回 `nts_notears` 或 `dag_sampling`。
- 将 `use_sample_graph_residual=0`。
- 将 `causal_input_norm=none`。
- 将 loss schedule 切回 `constant`，并使用当前旧权重。
