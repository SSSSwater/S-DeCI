## Why

前几轮检查训练/测试流程时发现两个已经实际修改的问题需要补入 OpenSpec：

1. `Abide_Dataset` 原先只加载长度等于 `seq_len=300` 的样本，导致 ABIDE AAL116 时间序列实际只进入训练 `138` 个样本，而数据目录中存在 `1025` 个可用样本。
2. epoch 指标打印原先是单行长文本，字段多时不便于查看；后来改成全部换行又过高，因此最终调整为按类别分行显示。

这些行为会影响训练数据口径和实验可读性，需要纳入主流程规格，避免后续回滚或误改。

## What Changes

- 数据加载阶段不再因为时间长度不等于 `seq_len` 而丢弃 ABIDE/MDD 样本。
- `collate_fn` 在组 batch 时统一根据 `args.seq_len` 对齐时间维：
  - 长序列截断到 `seq_len`
  - 短序列在末尾补 0 到 `seq_len`
- 相关矩阵 fallback 仍保持 `[N, N]` 原形，不参与时间维截断/补零。
- epoch 日志从单行长串改为按类别分行：
  - `[Loss]` 一行
  - `[Train Metrics]` 一行
  - `[Validation Metrics]` 一行
  - 类别之间使用分隔线，提高可读性且控制高度。

## Capabilities

### Modified Capabilities

- `project-structure`: 数据加载职责需要明确 `seq_len` 对齐策略。
- `training-test-scripts`: 训练日志打印格式需要明确按类别分行显示。

## Impact

- 影响文件：
  - `data_provider/data_factory_CV.py`
  - `data_provider/data_loader_CV.py`
  - `exp/exp_classification_CV.py`
- 行为影响：
  - ABIDE AAL116 TS 默认 `seq_len=300` 时加载样本数从 `138` 变为 `1025`。
  - K-Fold 训练样本数会随全量样本增加而变化，例如 5 fold 下 ABIDE 单折约为 train `820`、test `205`。
  - 使用补零后的短序列会在尾部引入 0 padding；模型当前没有显式 mask，因此这是固定长度建模策略的一部分。
- 回滚方式：
  - 若需要恢复旧行为，可在对应 Dataset 中重新添加长度过滤，但不建议这样做，因为会隐式丢弃大量有效样本。
