## 1. 参数与配置

- [x] 1.1 在 `run_cv.py` 中新增或统一 `use_deci_module1`、`use_causal_module2`、`use_hyperbolic_modules34` 参数，help 使用中文描述。
- [x] 1.2 在根目录训练测试脚本中暴露同样的模块开关，并写入构造出的 experiment args。
- [x] 1.3 增加参数归一化逻辑，保证旧参数 `use_hgcn_module3`、`use_hpec_module4` 与 `use_hyperbolic_modules34` 不产生不一致状态。
- [x] 1.4 在训练开始每个 fold 时打印当前模块开关状态和分类路径名称。

## 2. S-DeCI 模型分支

- [x] 2.1 在 `models/S_DeCI.py` 中实现模块 1 禁用时的 raw projection 分支，输入 `[B, T, N]` 输出 `[B, N, d_model]`。
- [x] 2.2 保持模块 1 启用时现有 DeCI/Cycle feature 路径不变。
- [x] 2.3 将模块 1 输出统一缓存为后续模块使用的节点特征，并在缓存中标注 `cycle_feature` 或 `raw_projected_feature` 来源。
- [x] 2.4 校验模块 2 禁用时必须使用 batch 中的 `correlation_matrix`，缺失时给出清晰错误。

## 3. GCN Fallback

- [x] 3.1 在 `layers/` 中复用或新增普通 Euclidean GCN 层，支持 `[N, N]` 和 `[B, N, N]` adjacency。
- [x] 3.2 在 `S-DeCI` 中实现 `use_hyperbolic_modules34 == 0` 时的 GCN fallback 初始化和 forward。
- [x] 3.3 GCN fallback 使用模块 2 的 `A_learned` 或模块 2 禁用后的样本相关矩阵作为 adjacency。
- [x] 3.4 GCN fallback 输出与现有二分类/多分类训练流程兼容的 logits 或分数。
- [x] 3.5 在 GCN fallback 路径缓存 adjacency、GCN hidden、readout feature 和分类输出。

## 4. Loss 与训练流程

- [x] 4.1 调整训练 loss 选择：HGCN/HPEC 路径继续使用 HPEC primary loss、多 prototype loss 和模块 2 auxiliary loss。
- [x] 4.2 调整 GCN fallback loss：模块 2 启用时使用分类 loss 加模块 2 auxiliary loss，模块 2 禁用时只使用分类 loss。
- [x] 4.3 确保模块 1 禁用只改变节点特征来源，不改变当前路径应使用的 loss 类型。
- [x] 4.4 确保所有路径每次迭代只执行一次联合 `backward()`。

## 5. 可视化与日志

- [x] 5.1 更新中间量可视化，使模块 1 禁用时保存 raw projected feature，并明确标注不是 Cycle/seasonal feature。
- [x] 5.2 更新可视化，使 GCN fallback 路径保存实际 adjacency、GCN hidden/readout 和 logits。
- [x] 5.3 可视化文件名或标题区分 `hgcn_hpec` 与 `gcn_fallback`。
- [x] 5.4 保持最后 epoch 的 train/test t-SNE 输出可用，并按训练集/测试集样式与真实 label 颜色区分。

## 6. 验证

- [x] 6.1 运行 OpenSpec 校验，确认本变更 artifact 有效。
- [x] 6.2 使用低预算训练验证全模块路径：模块 1、模块 2、模块 3/4 均启用。
- [x] 6.3 使用低预算训练验证模块 1 禁用路径。
- [x] 6.4 使用低预算训练验证模块 2 禁用且使用 sample correlation matrix 的路径。
- [x] 6.5 使用低预算训练验证模块 3/4 禁用后的 GCN fallback 路径。
- [x] 6.6 检查训练日志包含 loss、accuracy、precision、recall、macro F1、ROC AUC 和当前模块路径。
