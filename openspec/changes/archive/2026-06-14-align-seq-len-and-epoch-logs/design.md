## Context

当前训练入口通过 `args.seq_len` 决定模型输入时间长度。原先部分数据集在 Dataset 加载阶段过滤时间长度，例如 ABIDE 只保留 `signal.shape[0] == seq_len` 的样本。这种实现让 `seq_len` 同时承担“模型输入长度”和“样本筛选条件”两个职责，导致 ABIDE 大量样本被静默丢弃。

训练日志方面，S-DeCI 已经打印较多 loss 和 metric 字段。单行输出太长，全部字段换行又占用过多终端高度，因此需要中间方案。

## Goals / Non-Goals

**Goals:**

- `seq_len` 只定义模型输入时间长度，不再隐式过滤样本。
- 对任意时间长度样本，训练 batch 中的时间维都对齐为 `seq_len`。
- epoch 日志按类别分行，兼顾可读性和高度。
- 保持训练、验证、可视化、t-SNE 流程的 batch 格式兼容。

**Non-Goals:**

- 本变更不引入 attention mask 或 padding mask。
- 本变更不改变相关矩阵文件的读取与形状。
- 本变更不调整 K-Fold 随机种子、early stopping 指标或 loss 口径。

## Decisions

### 1. 在 `collate_fn` 中统一处理时间长度

`collate_fn` 接收 batch 后，对每个时间序列执行：

```text
if T >= seq_len:
    x = x[:seq_len]
else:
    x = concat(x, zeros(seq_len - T, channel))
```

选择原因：Dataset 层负责读取原始样本，DataLoader/collate 层负责 batch 对齐。这使不同长度样本都能进入训练，并让对齐策略集中在一个地方。

### 2. Dataset 不再按时间长度过滤 ABIDE/MDD

`Abide_Dataset` 与 `MDD_Dataset` 只根据 `protocol` 和 `data_type` 匹配文件名，不再检查 `signal.shape[0] == seq_len`。

选择原因：长度过滤会改变实验样本口径，且对 ABIDE 影响很大。将长度对齐交给 collate 后，样本数应反映数据目录中实际可用文件数量。

### 3. 相关矩阵不参与时间维补零

样本级相关矩阵是 `[N, N]` 图结构，不含时间维。collate 仅 stack 相关矩阵，不截断或补零。

选择原因：相关矩阵和时间序列的语义不同，修改相关矩阵形状会破坏模块 3 adjacency 输入。

### 4. epoch 日志按类别分行

日志格式固定为：

```text
========================================================================
Epoch x/y
------------------------------------------------------------------------
[Loss] ...
------------------------------------------------------------------------
[Train Metrics] ...
------------------------------------------------------------------------
[Validation Metrics] ...
========================================================================
```

选择原因：每类信息保持一行，能快速横向比较同组字段；分隔线让连续 epoch 之间边界清晰。

## Risks / Trade-offs

- [Risk] 短序列补零后，模型没有 padding mask，尾部 0 可能成为可学习模式。  
  Mitigation: 当前先使用简单固定长度策略；若后续结果受影响，再新增 mask-aware 模型输入。

- [Risk] 全量 ABIDE 样本数增加后，训练耗时上升。  
  Mitigation: 这是正确样本口径带来的必要成本；低预算测试脚本仍可通过 fold/epoch/batch 参数控制耗时。

- [Risk] 日志一行字段仍可能较长。  
  Mitigation: 当前按类别分为三行，已经明显降低高度；若字段继续增加，再考虑把 loss 拆成 primary/aux 两行。

## Verification

- `python -m py_compile data_provider/data_factory_CV.py data_provider/data_loader_CV.py`
- ABIDE AAL116 TS 样本数验证：
  - `n = 1025`
  - label 分布：control `537`、patient `488`
  - 原始长度范围：`120-300`
- DataLoader batch 验证：
  - `batch data shape = (16, 300, 116)`
  - 5 fold 第一折 train `820`、test `205`
- ABIDE 低预算训练冒烟测试通过：
  - `test_training_smoke.py --data Abide --train-epochs 1 --kfold 2 ...`
- `python -m py_compile exp/exp_classification_CV.py`
- `_format_metric_line` 格式预览通过，日志按 `[Loss]`、`[Train Metrics]`、`[Validation Metrics]` 三行显示。
