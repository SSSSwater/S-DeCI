## ADDED Requirements

### Requirement: S-DeCI 使用模块 2 的有效因果图
`S-DeCI` SHALL 使用模块 2 输出的 `A_effective` 作为下游图路径的 adjacency；当样本残差图关闭时，`A_effective` MUST 与共享图行为兼容。

#### Scenario: 模块 2 输出共享图
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_sample_graph_residual == 0`
- **WHEN** `S-DeCI.forward()` 调用模块 2
- **THEN** 下游 HGCN 或 GCN fallback MUST 使用模块 2 的共享图作为 adjacency
- **AND** adjacency 方向语义 MUST 保持 `A[parent, child]`

#### Scenario: 模块 2 输出样本级有效图
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_sample_graph_residual == 1`
- **WHEN** `S-DeCI.forward()` 调用模块 2
- **THEN** 下游 HGCN 或 GCN fallback MUST 接收 batch 对应的 `A_effective`
- **AND** 图传播层 MUST 支持 `[B, N, N]` 样本级 adjacency 或提供等价 batch 处理
- **AND** 分类 loss MUST 能经 `A_effective` 回传到模块 2 的图学习参数

### Requirement: S-DeCI 按 epoch 应用模块 2 loss 调度
`S-DeCI` 训练流程 SHALL 在计算模块 2 auxiliary loss 时使用当前 epoch 对 loss 权重进行调度。

#### Scenario: 训练流程传入 epoch
- **GIVEN** 模块 2 启用
- **WHEN** 训练流程进入一个新 epoch
- **THEN** 系统 MUST 能将当前 epoch 或等价训练进度传给模块 2 loss 计算
- **AND** 模块 2 MUST 根据配置返回已调度后的 auxiliary loss

#### Scenario: 验证流程不改变训练权重
- **GIVEN** 模型处于验证或测试流程
- **WHEN** 系统执行 forward 和指标计算
- **THEN** 系统 MUST NOT 因验证流程更新 loss 调度状态
- **AND** 验证流程 MUST 能读取模块 2 诊断但不执行参数更新

### Requirement: S-DeCI 训练入口暴露模块 2 新参数
训练入口 SHALL 暴露模块 2 改造所需参数，并以中文 help 或备注分组说明其用途。

#### Scenario: 暴露 DAG 方法参数
- **GIVEN** 用户运行 `run_cv.py --help`
- **WHEN** 查看 S-DeCI 模块 2 参数分组
- **THEN** 用户 MUST 能看到 `dagma_logdet` 相关 `causal_graph_method` 选项
- **AND** MUST 能看到输入标准化、样本残差图和 loss 调度相关参数

#### Scenario: 保持旧参数兼容
- **GIVEN** 用户继续使用旧的 `nts_notears` 或 `dag_sampling` 参数
- **WHEN** 训练入口解析参数
- **THEN** 系统 MUST 正常初始化 S-DeCI
- **AND** 不应要求用户必须启用新方法

### Requirement: S-DeCI 可视化模块 2 新中间量
`S-DeCI` SHALL 在已有中间量可视化中加入模块 2 新增的共享图、样本残差图和有效图诊断。

#### Scenario: 可视化共享图
- **GIVEN** 模块 2 已启用
- **WHEN** 训练结束保存中间量可视化
- **THEN** 可视化 MUST 包含共享因果图或等价统计图
- **AND** 图标题 MUST 标明其为共享图

#### Scenario: 可视化样本残差图
- **GIVEN** `use_sample_graph_residual == 1`
- **WHEN** 训练结束保存中间量可视化
- **THEN** 可视化 MUST 包含 batch 中至少一个样本的残差图或有效图
- **AND** 图标题 MUST 标明其来自样本级图路径
