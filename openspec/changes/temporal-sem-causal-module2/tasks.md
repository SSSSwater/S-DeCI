## 1. 模块 1 时间序列输出

- [x] 1.1 检查 `layers/DeCI_Layer.py` 与 `models/S_DeCI.py`，确定 DeCI block 中可稳定取得的 seasonal/residual 时间序列中间量。
- [x] 1.2 修改 `S-DeCI` 缓存逻辑，新增 `latest_temporal_series`，并保持 `latest_node_features` 不变。
- [x] 1.3 在 `use_deci_module1=0` 时，将原始或归一化后的 `[B, T, N]` 输入作为 temporal SEM 回退输入。
- [x] 1.4 为模块 1 时间序列输出添加中文注释和可视化标题，明确它不同于 `[B, N, d_model]` 节点特征。

## 2. Temporal SEM 因果学习器

- [x] 2.1 新增正式可复用的 `TemporalSEMCausalLearner` 文件，不放入 `module_2_test/` 测试目录。
- [x] 2.2 实现 `lag_order` 历史窗口构造，支持输入 `[B, T, N]` 与 `[B, T, N, D]`。
- [x] 2.3 实现 `A0: [N, N]` contemporaneous DAG 参数化，并保证对角线为 0。
- [x] 2.4 实现 `A_lag: [L, N, N]` time-lagged directed graph 参数化，并保证每个 lag 的对角线为 0。
- [x] 2.5 实现基于 `A0` 与 `A_lag` 的时间序列预测式 SEM forward，输出 `X_hat` 和预测目标。
- [x] 2.6 输出统一结构，包含 `A0`、`A_lag`、`A_shared`、`A_effective`、可选 `A_delta`、`X_hat`、预测目标和诊断字典。

## 3. Loss 与 DAGMA 调度

- [x] 3.1 实现 temporal prediction loss，替代 temporal SEM 路径中的静态特征重构主目标。
- [x] 3.2 为 `A0` 实现 DAGMA log-det loss，并记录 log-det 状态、谱半径和数值保护信息。
- [x] 3.3 实现 DAGMA 完整调度：warmup、barrier、refine 三阶段，并暴露当前阶段和有效权重。
- [x] 3.4 实现 `A0` 与 `A_lag` 的 sparsity loss。
- [x] 3.5 实现 lag graph smoothness 或相邻 lag 稳定性 loss。
- [x] 3.6 确保 temporal SEM loss 不使用真实因果矩阵监督。

## 4. 受限样本残差图

- [x] 4.1 将样本级图限制为低秩 residual graph，避免自由学习完整 `[B, N, N]`。
- [x] 4.2 对 `A_delta` 应用 off-diagonal mask、幅度缩放或裁剪。
- [x] 4.3 增加 `A_delta` 的 L1、deviation 和低秩约束诊断。
- [x] 4.4 默认关闭或使用保守幅度启用样本 residual graph，保持旧配置可回退。

## 5. S-DeCI 接入

- [x] 5.1 增加高层配置 `causal_learning_target`，支持 `static_feature` 与 `temporal_sem`。
- [x] 5.2 在 `S-DeCI` 中根据 `causal_learning_target` 选择当前静态 `CausalGraphLearner` 或新增 `TemporalSEMCausalLearner`。
- [x] 5.3 将 temporal SEM 输出的 `A0/A_lag` 融合为下游 HGCN/GCN fallback 可用的 `A_effective`。
- [x] 5.4 保证 `A_effective` 为 `[N, N]` 或 `[B, N, N]` 时，下游图传播层均能正确处理。
- [x] 5.5 更新 `get_aux_loss()`、`get_aux_losses()` 和缓存字段，使 temporal SEM loss 与诊断可被训练流程读取。

## 6. CLI、测试脚本与文档

- [x] 6.1 在 `run_cv.py` 中新增少量高层参数：`causal_learning_target`、`temporal_lag_order` 和必要诊断开关，避免暴露过多低层超参数。
- [x] 6.2 更新 `test_mdd_best_config.py` 或新增独立测试脚本，使 IDE 可直接运行 temporal SEM smoke test。
- [x] 6.3 新增项目当前设计说明文档，描述模块 1 时间序列输出与 temporal SEM 模块 2，不修改原始 `docs/` 参考文档。
- [x] 6.4 同步更新相关 OpenSpec spec 文档，保持中文描述，必要英文关键词保留。

## 7. 诊断、可视化与指标

- [x] 7.1 训练日志除 acc/AUC 外，输出 temporal prediction loss、DAGMA stage、DAG penalty、sparsity、方向性和 residual graph 幅度。
- [x] 7.2 fold 结束后保存 `A0`、`A_lag`、`A_effective` 和可选 `A_delta` heatmap。
- [x] 7.3 实现 fold graph stability 指标，比较不同 fold 的 learned graph 相似度。
- [x] 7.4 实现 top-k edge frequency 统计，区分 `A0` 与 `A_lag`。
- [x] 7.5 将图稳定性、top-k 边频率和主要边列表保存到输出目录，便于后续实验对比。

## 8. 验证

- [x] 8.1 运行 `py_compile` 覆盖新增 temporal SEM 文件、`models/S_DeCI.py`、训练入口和测试脚本。
- [x] 8.2 运行 synthetic temporal SEM smoke test，确认预测式 SEM loss 下降且输出图和可视化。
- [x] 8.3 运行 MDD 1 iteration / 1 fold 低预算训练，确认 temporal SEM 路径可训练并输出图诊断。
- [x] 8.4 与当前 `static_feature` 模块 2 和相关矩阵 fallback 做同数据低预算对比。
- [x] 8.5 汇总推荐默认配置、已知风险和后续仍需研究的问题。
