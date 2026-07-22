## Why

当前 S-DeCI 模块 2 主要在模块 1 压缩后的节点特征 `C: [B, N, d]` 上做静态重构式因果图学习，方向性证据容易被特征压缩削弱，也容易学到“有利于重构/分类”的图而不是能解释时间动态的图。为了更贴近 fMRI 时间序列因果建模，需要让模块 1 输出可用于时间预测的节点时间序列，并将模块 2 改为预测式 SEM（Structural Equation Model）因果学习。

## What Changes

- 修改模块 1 的输出口径：除现有节点特征外，S-DeCI 应支持输出模块 1 分解后的节点时间序列，用作模块 2 的时间因果输入。
- 新增时间序列预测式因果图学习路径：模块 2 学习 contemporaneous DAG `A0` 与 time-lagged directed graph `A_lag`，通过历史时间窗口预测当前/下一步节点序列。
- 将模块 2 的核心损失从静态 reconstruction 为主，改为预测式 SEM 损失：`L_pred + DAGMA schedule + sparse + smooth/stability + sample residual regularization`。
- DAGMA 不再只是固定权重 penalty，而是实现完整调度流程：训练早期弱 DAG/重预测，中后期逐步增强 log-det DAG 约束与稀疏约束。
- 收紧样本级图：样本图只允许低秩、小幅度、强正则的 residual graph，避免样本图成为过拟合分类通道。
- 除 `acc / AUC` 外，训练与验证需要输出图学习诊断：预测误差、DAG penalty、稀疏度、方向性、样本图幅度、fold graph stability、top-k 边稳定性、`A0/A_lag` 可视化。
- 保留旧静态模块 2 作为回退路径，必要时可通过配置退回当前 `nts_notears / dagma_logdet / dag_sampling` 静态图学习。

## Capabilities

### New Capabilities

- `temporal-sem-causal-learning`: 描述模块 1 时间序列输出、模块 2 时间序列预测式 SEM 因果学习、DAGMA 调度、样本残差图约束和图稳定性诊断。

### Modified Capabilities

- `s-deci-model`: S-DeCI 的模块 1 输出口径与模块 2 输入路径改变，需要支持节点时间序列进入模块 2，并将 learned temporal graph 传递给后续模块。
- `module2-causal-learning`: 模块 2 从静态特征重构式图学习扩展为时间序列预测式 SEM，可保留旧静态路径作为 fallback。

## Impact

- 影响 `models/S_DeCI.py`：模块 1 需要缓存/输出节点时间序列；模块 2 调用路径需要支持 temporal input。
- 影响 `layers/causal_graph_learner.py` 或新增独立 temporal causal learner 文件：新增 `TemporalSEMCausalLearner`、DAGMA 调度、lag graph、低秩样本残差图。
- 影响 `run_cv.py` 与测试脚本：新增少量高层开关，避免暴露过多低层超参数。
- 影响训练循环 `exp/exp_classification_CV.py`：需要记录 temporal SEM loss、图稳定性、top-k 边稳定性等诊断。
- 影响可视化：新增 `A0`、`A_lag`、effective graph、sample residual graph、fold stability heatmap 和 top-k edge frequency 图。
- 新增项目更新说明文档，不能修改 `docs/` 中作为初始参考的原始设计文档。
- 回滚方案：保留旧模块 1 特征输出与旧静态模块 2，配置关闭 temporal SEM 后仍按当前 S-DeCI 路径训练。
