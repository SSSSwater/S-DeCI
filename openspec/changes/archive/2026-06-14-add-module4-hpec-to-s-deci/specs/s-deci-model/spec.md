## ADDED Requirements

### Requirement: S-DeCI 可配置启用模块 4 HPEC 分类

`S-DeCI` SHALL 支持通过配置启用模块 4 HPEC energy 分类路径，并在未启用时保留模块 3 线性分类回退路径。

#### Scenario: 启用模块 4

- **GIVEN** `use_hpec_module4 == 1`
- **WHEN** 初始化 `S-DeCI`
- **THEN** 系统 MUST 同时要求 `use_hgcn_module3 == 1`
- **AND** 模型 MUST 初始化 HPEC 原型能量分类组件
- **AND** 模型 MUST 使用模块 3 输出的 `z_global` 作为默认 HPEC 输入

#### Scenario: 关闭模块 4

- **GIVEN** `use_hpec_module4 == 0`
- **WHEN** `S-DeCI` 执行 forward
- **THEN** 模型 MUST 保留当前模块 3 `logmap0(z_global)` 线性分类头
- **AND** 模型 MUST 保留模块 3 关闭时的 Cycle/seasonal logits 分类路径

### Requirement: S-DeCI 使用 HPEC energy 替换分类 loss

`S-DeCI` SHALL 在启用模块 4 时使用 `Loss_HPEC` 替换普通 MSE/CE 线性分类 loss，并保留模块 2 辅助损失组成联合训练目标。

#### Scenario: 计算模块 4 联合 loss

- **GIVEN** `S-DeCI` 已启用模块 2、模块 3 和模块 4
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 等价于 `Loss_HPEC(z_global, label) + alpha * Loss_Recon(C, C_hat) + lambda * Loss_DAG(A_learned) + gamma * L1(A_learned)`
- **AND** 总 loss MUST NOT 使用真实因果矩阵监督
- **AND** 分类 loss MUST 能通过模块 4、模块 3 和 `A_learned` 回传到模块 2 因果图学习参数

#### Scenario: 模型提供 primary loss

- **GIVEN** 模块 4 已完成一次 forward
- **WHEN** 训练流程需要分类 loss
- **THEN** `S-DeCI` MUST 能提供 HPEC primary/classification loss
- **AND** 训练流程 MUST 在该 loss 可用时优先使用它，而不是外部普通 criterion

### Requirement: S-DeCI 使用 energy-based 预测与指标

`S-DeCI` SHALL 在启用模块 4 时使用 HPEC energy-based prediction 计算预测类别和指标。

#### Scenario: HPEC 预测

- **GIVEN** `energy_matrix` 的形状为 `[B, classes]`
- **WHEN** 模块 4 产生预测
- **THEN** 预测类别 MUST 使用 `argmin(energy_matrix, dim=1)`
- **AND** 概率类指标 MUST 使用 `softmax(-energy_matrix)` 或等价 energy-based probability

#### Scenario: forward 返回兼容训练流程

- **WHEN** `S-DeCI.forward()` 在模块 4 启用时返回主输出
- **THEN** 返回值 MUST 能被现有训练、验证和测试流程收集
- **AND** 指标计算 MUST 优先使用模型缓存的 HPEC prediction/probability，避免对 energy 输出使用错误的 sigmoid 阈值

### Requirement: S-DeCI 缓存模块 4 诊断量并添加中文注释

`S-DeCI` SHALL 缓存模块 4 关键中间量，并在新增逻辑处提供简洁中文注释。

#### Scenario: 缓存模块 4 中间量

- **WHEN** 模块 4 完成一次 forward
- **THEN** 模型 MUST 能读取 `prototypes`、`angle_matrix`、`aperture` 或 `psi`、`energy_matrix`、HPEC prediction、HPEC probability 和 `Loss_HPEC`
- **AND** 缓存中间量 MUST 不改变 `S-DeCI.forward()` 的主返回值

#### Scenario: 中文注释

- **WHEN** 开发者查看 `models/S_DeCI.py`
- **THEN** 模块 4 初始化、HPEC forward、loss 缓存、energy prediction 和可视化缓存相关代码 MUST 带有中文注释
- **AND** 注释 MAY 保留 `HPEC`、`energy`、`prototype`、`Poincare Ball`、`z_global` 等必要英文关键词
