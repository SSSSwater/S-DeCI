## MODIFIED Requirements

### Requirement: 数据加载职责

项目 SHALL 通过 `data_provider/` 中的数据集类与 DataLoader 组织训练数据，并清晰区分“原始样本读取”和“batch 时间长度对齐”的职责。

#### Scenario: Dataset 读取原始时间序列

- **WHEN** 实例化 `Abide_Dataset`、`MDD_Dataset` 或其他时间序列数据集类
- **THEN** Dataset MUST 根据 `args.data_path`、`args.data_type` 和 `args.protocol` 匹配可用样本文件
- **AND** Dataset MUST NOT 因样本时间长度不等于 `args.seq_len` 而静默丢弃样本
- **AND** Dataset SHOULD 保留原始时间序列张量，供 collate 阶段统一对齐

#### Scenario: DataLoader 按 seq_len 对齐时间维

- **GIVEN** batch 中存在形状为 `[T, N]` 的时间序列样本
- **WHEN** `collate_fn` 使用 `args.seq_len` 组装 batch
- **THEN** 若 `T > args.seq_len`，系统 MUST 沿时间维截断到 `args.seq_len`
- **AND** 若 `T < args.seq_len`，系统 MUST 在时间维末尾补 0 到 `args.seq_len`
- **AND** 输出 batch 的时间序列形状 MUST 为 `[B, args.seq_len, N]`

#### Scenario: 相关矩阵保持图结构形状

- **GIVEN** batch 中包含样本级相关矩阵
- **WHEN** `collate_fn` 组装 batch
- **THEN** 系统 MUST 保持每个相关矩阵的 `[N, N]` 形状
- **AND** 系统 MUST NOT 对相关矩阵执行时间维截断或补零
