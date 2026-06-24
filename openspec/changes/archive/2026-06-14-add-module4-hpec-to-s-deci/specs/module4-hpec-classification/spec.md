## ADDED Requirements

### Requirement: 模块 4 提供 HPEC 原型能量分类组件

系统 SHALL 提供模块 4 HPEC 原型能量分类组件，用于根据模块 3 输出的双曲中心点 `z_global` 与类别原型计算能量矩阵。

#### Scenario: 创建可复用 HPEC 层

- **GIVEN** 项目需要在 `S-DeCI` 中接入模块 4
- **WHEN** 开发者查看 `layers/`
- **THEN** MUST 能找到 HPEC 原型、角度、孔径和能量计算相关层文件
- **AND** 该层文件 MUST NOT 依赖 `reference/` 目录作为运行时 import 路径

#### Scenario: HPEC 输入输出形状

- **GIVEN** `z_global` 的形状为 `[B, hgcn_hidden_dim]`
- **AND** 类别数为 `K`
- **WHEN** 模块 4 执行 forward
- **THEN** MUST 输出 `energy_matrix`
- **AND** `energy_matrix` 的形状 MUST 为 `[B, K]`
- **AND** MUST 输出每个样本的预测类别

### Requirement: HPEC 原型初始化

模块 4 SHALL 使用可分离的类别原型，并将原型投影到 Poincare Ball 中。

#### Scenario: 初始化原型

- **GIVEN** 类别数 `K` 和双曲中心维度 `D`
- **WHEN** 初始化模块 4
- **THEN** MUST 构造形状为 `[K, D]` 的类别原型
- **AND** MUST 通过 hyperspherical separation 或等价方法使原型方向尽量分离
- **AND** MUST 按 `hpec_prototype_radius` 缩放后投影到 Poincare Ball

#### Scenario: 原型可配置

- **WHEN** 用户配置模块 4
- **THEN** 系统 MUST 支持配置原型半径
- **AND** MUST 支持配置原型是否可训练
- **AND** MUST 支持配置原型初始化步数或等价初始化强度

### Requirement: HPEC 角度能量函数

模块 4 SHALL 按 HPEC 参考实现计算角度、孔径和 energy，并提供必要的数值稳定处理。

#### Scenario: 计算角度与孔径

- **GIVEN** 双曲样本点 `z_global`
- **AND** 双曲类别原型 `prototype`
- **WHEN** 模块 4 计算 HPEC energy
- **THEN** MUST 计算样本到每个原型的角度 `Xi`
- **AND** MUST 计算每个原型的孔径 `psi`
- **AND** MUST 对 `acos`、`asin` 和除法相关输入做 clamp 或 eps 稳定化

#### Scenario: 计算 energy

- **WHEN** 角度 `Xi` 和孔径 `psi` 可用
- **THEN** MUST 计算非负 energy `max(0, Xi - psi)`
- **AND** MUST 保留形状为 `[B, K]` 的 energy matrix 供 loss、预测和可视化使用

### Requirement: HPEC energy loss

模块 4 SHALL 用 HPEC energy loss 替换启用模块 4 时的普通分类 loss。

#### Scenario: 计算 HPEC loss

- **GIVEN** `energy_matrix` 和真实标签 `label`
- **WHEN** 系统计算模块 4 分类 loss
- **THEN** MUST 取真实类别 energy 作为 positive penalty
- **AND** MUST 对非真实类别使用 margin 形式的 negative penalty
- **AND** MUST 对 batch 求平均得到 `Loss_HPEC`

#### Scenario: 推理使用最小 energy

- **WHEN** 模块 4 进行预测
- **THEN** MUST 使用 `argmin(energy_matrix, dim=1)` 得到预测类别
- **AND** 二分类和多分类指标 MUST 使用该 energy-based prediction 或由 `softmax(-energy)` 得到的概率

### Requirement: 模块 4 诊断缓存

模块 4 SHALL 缓存关键 HPEC 中间量，供训练诊断和可视化使用。

#### Scenario: 缓存 HPEC 中间量

- **WHEN** `S-DeCI` 启用模块 4 并完成一次 forward
- **THEN** MUST 能读取原型、角度矩阵、孔径、energy matrix、预测类别和 `Loss_HPEC`
- **AND** 缓存中间量 MUST 不改变 `S-DeCI.forward()` 的主返回值
