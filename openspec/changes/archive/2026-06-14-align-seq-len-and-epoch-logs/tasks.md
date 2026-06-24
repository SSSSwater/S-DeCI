## 1. 数据长度对齐

- [x] 1.1 检查 ABIDE AAL116 样本数只有 138 的原因，确认是 `Abide_Dataset` 按 `seq_len=300` 过滤长度导致。
- [x] 1.2 统计 ABIDE AAL116 TS 实际文件数和长度分布，确认数据目录中存在 1025 个可用样本。
- [x] 1.3 修改 `data_provider/data_factory_CV.py`，在 `collate_fn` 中按 `seq_len` 对长序列截断、短序列补 0。
- [x] 1.4 修改 `data_provider/data_loader_CV.py`，移除 `Abide_Dataset` 的长度过滤。
- [x] 1.5 修改 `data_provider/data_loader_CV.py`，移除 `MDD_Dataset` 的长度过滤，保持同一数据加载口径。
- [x] 1.6 确认相关矩阵 fallback 仍只做 stack，不参与时间维截断或补零。

## 2. Epoch 日志格式

- [x] 2.1 修改 `exp/exp_classification_CV.py` 的 `_format_metric_line`，按类别分行打印 epoch 结果。
- [x] 2.2 保留分隔线，让不同 epoch、loss、训练指标和验证指标边界清晰。
- [x] 2.3 将 loss 字段压缩到 `[Loss]` 一行。
- [x] 2.4 将训练指标压缩到 `[Train Metrics]` 一行。
- [x] 2.5 将验证指标压缩到 `[Validation Metrics]` 一行。

## 3. 验证

- [x] 3.1 运行 `python -m py_compile data_provider/data_factory_CV.py data_provider/data_loader_CV.py`。
- [x] 3.2 验证 ABIDE AAL116 TS 加载样本数为 1025，类别分布为 control 537、patient 488。
- [x] 3.3 验证 ABIDE 5 fold 第一折样本数为 train 820、test 205。
- [x] 3.4 验证 DataLoader 输出 batch 形状为 `[16, 300, 116]`。
- [x] 3.5 运行 ABIDE 低预算训练冒烟测试，确认全量加载和补零/截断策略可训练。
- [x] 3.6 运行 `python -m py_compile exp/exp_classification_CV.py`。
- [x] 3.7 预览 `_format_metric_line` 输出，确认日志按类别分行且高度可控。
