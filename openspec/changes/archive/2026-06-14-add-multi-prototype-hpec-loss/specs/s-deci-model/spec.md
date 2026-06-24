## ADDED Requirements

### Requirement: S-DeCI 支持模块 4 多 prototype 参数

`S-DeCI` SHALL 在启用模块 4 HPEC 时支持每类多 prototype 配置，并将相关超参数传递给 HPEC 层。

#### Scenario: 传入多 prototype 配置

- **GIVEN** 用户通过训练入口设置 `hpec_prototypes_per_class`
- **WHEN** `S-DeCI` 初始化模块 4
- **THEN** 模型 MUST 将 `hpec_prototypes_per_class` 传递给 HPEC 模块 4
- **AND** 模型 MUST 支持配置 `hpec_proto_temperature`
- **AND** 模型 MUST 支持配置 `lambda_hpec_mle`、`lambda_hpec_pcl` 和 `lambda_hpec_pal`

#### Scenario: 模块 4 关闭时不创建多 prototype

- **GIVEN** `use_hpec_module4 == 0`
- **WHEN** `S-DeCI` 初始化
- **THEN** 模型 MUST NOT 初始化多 prototype HPEC 层
- **AND** 新增 prototype loss MUST 不参与训练

### Requirement: S-DeCI 暴露多 prototype loss

`S-DeCI` SHALL 在 forward 后基于 label 计算并暴露多 prototype 相关 loss，使训练流程能够将其加入总 loss。

#### Scenario: 计算 prototype loss

- **GIVEN** `S-DeCI` 已启用模块 3 和模块 4
- **AND** 模块 4 已完成一次 forward
- **WHEN** 训练流程调用模型的 label-aware loss 计算方法
- **THEN** 模型 MUST 能计算 `L_mle`、`L_pcl` 和 `L_pal`
- **AND** 模型 MUST 按配置权重得到 prototype auxiliary loss
- **AND** prototype auxiliary loss MUST 能与 HPEC primary loss 和模块 2 auxiliary loss 一起反向传播

#### Scenario: 总 loss 保持联合训练结构

- **GIVEN** 模块 2、模块 3 和模块 4 均已启用
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 `Loss_HPEC + alpha * Loss_Recon + lambda * Loss_DAG + gamma * L1`
- **AND** 当对应权重大于 0 时 MUST 额外包含 `L_mle`、`L_pcl` 和 `L_pal`
- **AND** 总 loss MUST NOT 使用真实因果矩阵监督

### Requirement: S-DeCI 缓存多 prototype 可视化中间量

`S-DeCI` SHALL 缓存多 prototype HPEC 中间量，供 heatmap 和 t-SNE 可视化使用。

#### Scenario: 缓存多 prototype 矩阵

- **GIVEN** HPEC 模块 4 已完成 forward
- **WHEN** 用户调用 `visualize_causal_intermediates`
- **THEN** 可视化输入 MUST 包含多 prototype 张量或其可读展开形式
- **AND** 可视化输入 SHOULD 包含 prototype-level energy 或 prototype similarity

#### Scenario: t-SNE 使用多 prototype

- **GIVEN** 用户显式开启 `visualize_causal`
- **AND** HPEC 模块 4 使用每类多个 prototype
- **WHEN** 最终 epoch t-SNE 图生成
- **THEN** 图中 MUST 显示每个类别下的多个 prototype 点
- **AND** prototype 点 MUST 与 train/test 样本使用同一 t-SNE 投影坐标系

### Requirement: S-DeCI 多 prototype 关键逻辑中文注释

`S-DeCI` SHALL 在本次新增或改动的多 prototype、prototype loss、可视化逻辑处提供简洁中文注释，必要英文关键词可以保留。

#### Scenario: 注释多 prototype loss

- **WHEN** 开发者查看 `models/S_DeCI.py` 或 HPEC 层文件
- **THEN** 多 prototype 初始化、类别能量聚合、`L_mle`、`L_pcl`、`L_pal` 相关代码 MUST 带有中文注释
- **AND** 注释 MAY 保留 `prototype`、`energy`、`HPEC`、`Poincare Ball`、`logmap0` 等英文关键词
