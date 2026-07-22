## 1. 模块 2 learner 实现

- [x] 1.1 新增 attention-guided temporal causal learner 文件，并保持 `TemporalSEMOutput` 兼容字段：`x_hat`、`target`、`a0`、`a_lag`、`a_shared`、`a_effective`、`dag_penalty`、`dag_metadata`。
- [x] 1.2 实现 lag-window cross-attention：使用历史 `x_{t-1}, ..., x_{t-L}` 预测 `x_t`，并支持 `[B,T,N]` 与 `[B,T,N,D]` 输入。
- [x] 1.3 按公式实现 `score=<q_child,k_parent>/sqrt(d_h)`、`attn=softmax_parent(score)`、`edge=attn*sigmoid(theta)`，并保持 `parent -> child` 方向语义。
- [x] 1.4 实现结构门控 `G_lag`，用 `attention * gate` 聚合得到稳定 `A_lag: [lag_order,N,N]`，并保证对角线为 0。
- [x] 1.5 实现 `A0` 同时间片残余依赖图，并仅对 `A0` 计算 DAGMA/NOTEARS 风格无环约束。
- [x] 1.6 实现 auxiliary loss：`temporal_pred_loss`、`temporal_sparse_loss`、`temporal_smooth_loss`、`causal_dag_loss`，不加入真实因果监督和额外无效辅助损失。
- [x] 1.7 在 `dag_metadata` 或等价诊断字典中输出 attention entropy、gate mass、`A_lag` mass/directionality、`A0` mass/directionality。

## 2. S-DeCI 接入

- [x] 2.1 在 `models/S_DeCI.py` 中接入 `causal_graph_method == "attn_nts_notears"` 或等价配置，初始化新的 attention-guided learner。
- [x] 2.2 保持当前 `nts_notears` temporal 路径可回退，不破坏现有默认训练。
- [x] 2.3 确保模块 3 默认使用 `A_lag.mean(dim=0)` 或 `a_effective` 作为 learned causal graph，不默认使用 `A0`。
- [x] 2.4 保持 `classification_graph_source == "blend"` 时与样本相关矩阵按 `module2_sample_correlation_blend` 融合，并保持 learned graph 可微。
- [x] 2.5 缓存 attention-guided 中间量，供可视化、图诊断和训练日志读取。
- [x] 2.6 添加简洁中文注释，说明 `A_lag` 是跨时间主因果图，`A0` 是同时间片残余依赖图。

## 3. 训练入口与默认参数

- [x] 3.1 更新 `run_cv.py` 和根目录测试脚本的 `causal_graph_method` choices，支持 `attn_nts_notears`。
- [x] 3.2 新增少量 attention 参数：head 数、head dim、attention dropout、分类图尺度；损失权重复用现有 temporal 模块 2 参数。
- [x] 3.3 确保参数 help 使用中文说明，必要英文关键词保留。
- [x] 3.4 不新增 raw attention 对比损失、真实因果监督损失、prototype 类辅助损失等无效窗口。
- [x] 3.5 更新 result 记录逻辑，只记录模块设计相关参数和最终测试指标。

## 4. 可视化与诊断

- [x] 4.1 更新因果图可视化，保存 `A_lag_mean`、每个 lag 的 `A_lag[k]`、`A0` 和最终 `A_cls`。
- [x] 4.2 可视化标题或副标题明确标注 `A0` 为同时间片残余依赖，`A_lag_mean` 为跨时间主因果图。
- [x] 4.3 更新训练日志，在 `[Graph Diagnostics]` 中打印 attention entropy、gate mass、`A_lag` 和 `A0` 相关诊断量。
- [x] 4.4 确保测试集可视化不把真实 label 传入模型 forward，label 仅用于训练后可视化对照。

## 5. 文档与规范同步

- [x] 5.1 新建项目修改说明文档，描述 attention-guided temporal NTS-NOTEARS 的数据流、`A0` 作用、损失函数和下游图选择，不修改原始 `doc/` 参考文档。
- [x] 5.2 更新或补充 OpenSpec 主规范，使模块 2、S-DeCI 和训练脚本要求与本次实现一致。
- [x] 5.3 检查旧规范中已删除的无效 prototype auxiliary loss、counterfactual loss 等描述，避免新文档继续引用。

## 6. 验证

- [x] 6.1 运行 `py_compile` 检查新增 learner、`models/S_DeCI.py`、训练入口和实验流程文件。
- [x] 6.2 使用随机 `[B,T,N]` 输入做 forward/backward 单元级检查，确认 `A_lag`、`A0`、`A_cls` 形状正确且无 NaN。
- [x] 6.3 运行 MDD 1 fold 快速训练，确认 attention-guided 模块 2 能跑通并保存指标。
- [x] 6.4 与当前 temporal NTS-NOTEARS 路径做 MDD 5-fold 对比并记录到 `result.xlsx`：当前最佳 attention 配置为 heads=1、head_dim=4、dropout=0.1、graph_scale=1，结果 acc=66.42%、precision=60.35%、recall=57.48%、macro F1=56.81%、AUC=60.63%。
- [x] 6.5 检查训练日志中 `A_lag` 随 epoch 变化，并确认分类 loss 能回传到 attention-guided 图参数。
