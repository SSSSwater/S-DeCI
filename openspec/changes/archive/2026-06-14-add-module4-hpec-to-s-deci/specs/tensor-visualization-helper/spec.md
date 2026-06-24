## ADDED Requirements

### Requirement: 可视化 helper 支持 HPEC 中间量

系统 SHALL 使用既有 tensor 可视化 helper 显示模块 4 HPEC 的关键中间量。

#### Scenario: 保存 HPEC heatmap

- **GIVEN** 用户显式开启 `S-DeCI` 可视化
- **WHEN** 启用模块 4 并完成一个 fold 的训练
- **THEN** 系统 SHOULD 保存 HPEC prototype、angle matrix、aperture 或 `psi`、energy matrix、prediction 和 label 对照 heatmap
- **AND** 文件名 SHOULD 能区分 `train`、`test`、fold 和模块 4 内容

#### Scenario: 显示 label 与预测

- **WHEN** 可视化 HPEC prediction 和 label
- **THEN** 可视化结果 MUST 能显示真实 label
- **AND** 测试集真实 label MUST 仅用于 forward 之后的绘图标注，不得作为模型输入

### Requirement: HPEC 可视化保持现有维度提示规则

HPEC 可视化 SHALL 继续遵守 tensor helper 的维度展示规则。

#### Scenario: 显示矩阵维度

- **WHEN** 可视化 HPEC prototype、angle matrix 或 energy matrix
- **THEN** 每个 subplot MUST 显示原始维度
- **AND** 若输入为三维张量，subplot MUST 明确提示当前展示的是 Batch0 或调用方指定的 batch index

### Requirement: 最终 epoch t-SNE 支持 HPEC 表示

系统 SHALL 在模块 4 启用时继续保存最终 epoch 的 train/test 联合 t-SNE，并优先使用 HPEC 对应的模型表示。

#### Scenario: 生成 HPEC t-SNE

- **GIVEN** 用户显式开启 `visualize_causal`
- **AND** `use_hpec_module4 == 1`
- **WHEN** 每个 fold 最后一个 epoch 结束
- **THEN** 系统 SHOULD 使用 `logmap0(z_global)` 或等价 HPEC 输入表示生成 t-SNE
- **AND** train/test MUST 使用不同 marker 或样式区分
- **AND** 真实 label MUST 使用颜色区分
- **AND** 测试集 label MUST NOT 输入模型 forward

### Requirement: HPEC 可视化默认不影响训练

系统 SHALL 保持 HPEC 可视化为显式调试能力，默认不改变训练、验证或推理行为。

#### Scenario: 默认关闭 HPEC 可视化

- **WHEN** 用户未显式开启可视化配置
- **THEN** 系统 MUST NOT 自动保存 HPEC 图片
- **AND** 可视化缓存和保存逻辑 MUST NOT 改变模型输出、loss 计算或反向传播结果
