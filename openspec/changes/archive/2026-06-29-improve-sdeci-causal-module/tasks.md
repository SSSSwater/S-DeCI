## 1. 核心因果学习器

- [x] 1.1 在 `layers/causal_graph_learner.py` 中新增 `dagma_logdet` 或等价 log-det DAG penalty，并保留 `notears`、`analytic`、`dag_sampling` 现有路径。
- [x] 1.2 为 `CausalGraphLearner` 增加 `causal_input_norm` 配置，支持 `none` 与至少一种稳定标准化方式。
- [x] 1.3 将模块二输出结构扩展为包含 `A_shared`、`A_effective`、可选 `A_delta` 和扩展诊断字段。
- [x] 1.4 实现可选样本残差图生成逻辑，并保证对角线为 0、形状为 `[B, N, N]`。
- [x] 1.5 增加样本残差图的 L1、deviation 和幅度控制参数。

## 2. Loss 与训练调度

- [x] 2.1 实现模块二 auxiliary loss 权重调度函数，支持 `constant` 和 `warmup`。
- [x] 2.2 在 `S-DeCI` 或训练流程中将当前 epoch 传给模块二 loss 计算路径。
- [x] 2.3 将 reconstruction、DAG、L1、sample graph 正则的原始值与加权值分别暴露到诊断字典。
- [x] 2.4 保持旧配置可回退：关闭样本残差图、关闭输入标准化、使用旧方法时应保持现有训练行为。

## 3. S-DeCI 接入

- [x] 3.1 修改 `models/S_DeCI.py`，使下游 HGCN 或 GCN fallback 优先使用模块二输出的 `A_effective`。
- [x] 3.2 确保 `[N, N]` 共享图和 `[B, N, N]` 样本级图均能被图传播层正确处理。
- [x] 3.3 更新 `get_aux_loss()`、`get_aux_losses()` 和相关缓存，使新增 loss 与诊断可被训练流程读取。
- [x] 3.4 为新增 S-DeCI 逻辑添加简洁中文注释，必要英文关键词保留。

## 4. CLI 与文档

- [x] 4.1 在 `run_cv.py` 中新增模块二方法、输入标准化、样本残差图和 loss 调度参数，并保持中文 help。
- [x] 4.2 更新独立测试脚本或新增测试参数预设，便于对比 `dagma_logdet`、`nts_notears`、`dag_sampling`。
- [x] 4.3 新建一份项目当前模块二更新说明文档，不覆盖 `docs/` 初始参考文档。
- [x] 4.4 确认新增参数不会破坏现有 `test_mdd_best_config.py` 默认基线。

## 5. 可视化与诊断

- [x] 5.1 扩展 `S-DeCI` 中间量可视化，加入共享图、样本残差图、有效图和相关标题。
- [x] 5.2 在训练打印或诊断中增加 DAG penalty、图变化幅度、方向性指标和样本残差图正则值。
- [x] 5.3 确保验证/测试可读取诊断信息，但不更新调度状态或模型参数。

## 6. 验证

- [x] 6.1 运行 Python 编译检查：`py_compile` 覆盖修改过的核心文件。
- [x] 6.2 运行低预算 S-DeCI smoke test，确认新默认配置可训练。
- [x] 6.3 运行 MDD 1 fold / 1 iteration 对比测试，至少比较当前最佳 fallback 与一个启用模块二的新配置。
- [x] 6.4 运行模块二合成数据训练检查，确认 `dagma_logdet` 能输出有效图、指标与可视化。
- [x] 6.5 汇总观察结果，记录推荐默认参数与仍需后续研究的问题。
