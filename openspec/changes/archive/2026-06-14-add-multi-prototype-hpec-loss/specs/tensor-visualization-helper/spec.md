## ADDED Requirements

### Requirement: 多 prototype heatmap 可视化

S-DeCI 中间量可视化 SHALL 支持展示多 prototype HPEC 的 prototype 张量和 prototype-level energy。

#### Scenario: 展示多 prototype 张量

- **GIVEN** HPEC prototype 张量形状为 `[classes, K, hidden_dim]`
- **WHEN** 用户显式保存 S-DeCI 中间量 heatmap
- **THEN** 系统 MUST 将 prototype 张量转换为可读的二维或三维 heatmap 输入
- **AND** 标题 MUST 标明类别数、每类 prototype 数和 hidden 维度

#### Scenario: 展示 prototype-level energy

- **GIVEN** HPEC forward 产生 prototype-level energy
- **WHEN** 用户显式保存 S-DeCI 中间量 heatmap
- **THEN** 系统 SHOULD 保存 prototype-level energy 或 similarity heatmap
- **AND** 标题 MUST 能区分类别级 energy 和 prototype-level energy

### Requirement: t-SNE 显示每类多个 prototype

最终 epoch 的 train/test t-SNE 可视化 SHALL 显示每个类别下的多个 prototype，并与样本点使用同一 t-SNE 投影。

#### Scenario: 绘制多 prototype 点

- **GIVEN** 用户显式开启 `visualize_causal`
- **AND** `hpec_prototypes_per_class > 1`
- **WHEN** 系统生成最终 epoch t-SNE
- **THEN** 图中 MUST 绘制所有 prototype 点
- **AND** prototype 点 MUST 使用与 train/test 不同的 marker
- **AND** prototype 点颜色 MUST 与其类别 label 对应

#### Scenario: prototype 与样本同坐标系

- **GIVEN** 训练集样本、测试集样本和 prototype 都需要显示
- **WHEN** 系统执行 t-SNE 降维
- **THEN** 系统 MUST 将样本 embedding 和 prototype embedding 拼接后一起执行 t-SNE
- **AND** 系统 MUST NOT 将 prototype 用单独 t-SNE 或不同坐标系投影后叠加到图上

#### Scenario: 单 prototype t-SNE 兼容

- **GIVEN** `hpec_prototypes_per_class == 1`
- **WHEN** 系统生成最终 epoch t-SNE
- **THEN** 图中 MUST 仍能显示每个类别的单 prototype 点
- **AND** 图例或标题 MUST 不因 prototype 数量为 1 而报错
