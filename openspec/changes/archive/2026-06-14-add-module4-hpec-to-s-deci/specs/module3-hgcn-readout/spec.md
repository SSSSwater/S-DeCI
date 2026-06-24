## ADDED Requirements

### Requirement: 模块 3 向模块 4 提供稳定 HPEC 输入

模块 3 SHALL 向模块 4 提供稳定的 `z_global` 和 `logmap0(z_global)` 缓存，使 HPEC energy 分类可以复用模块 3 的双曲 readout。

#### Scenario: 暴露 z_global

- **GIVEN** `S-DeCI` 启用模块 3 和模块 4
- **WHEN** 模块 3 完成 HGCN readout
- **THEN** 模块 3 MUST 输出形状为 `[B, hgcn_hidden_dim]` 的 `z_global`
- **AND** `z_global` MUST 位于 Poincare Ball 或可被投影回 Poincare Ball
- **AND** 模块 4 MUST 能直接读取该表示作为默认输入

#### Scenario: 暴露切空间表示

- **WHEN** 模块 3 缓存诊断量
- **THEN** 模块 3 MUST 缓存 `logmap0(z_global)`
- **AND** 该缓存 MUST 可用于线性分类回退、t-SNE 可视化和 HPEC 调试对照

### Requirement: 模块 3 保持 HPEC 接入后的维度语义

模块 3 SHALL 在接入模块 4 后保持 `hgcn_hidden_dim` 的维度语义稳定，默认输出 128 维双曲中心点。

#### Scenario: 默认 HPEC 输入维度

- **GIVEN** 用户未显式设置 `hgcn_hidden_dim`
- **WHEN** `S-DeCI` 同时启用模块 3 和模块 4
- **THEN** `z_global` 的最后一维 MUST 默认为 `128`
- **AND** HPEC prototype 的最后一维 MUST 与 `z_global` 一致

#### Scenario: 自定义 HPEC 输入维度

- **GIVEN** 用户显式设置 `hgcn_hidden_dim`
- **WHEN** 初始化模块 3 和模块 4
- **THEN** HGCN 输出、`z_global` 和 HPEC prototype MUST 使用同一隐藏维度

### Requirement: 模块 3 继续支持分类梯度传入因果图

模块 3 SHALL 在模块 4 启用后继续允许分类目标通过 `A_learned` 回传到模块 2 因果图学习参数。

#### Scenario: HPEC loss 经 HGCN 回传

- **GIVEN** `Loss_HPEC` 已由模块 4 计算
- **WHEN** 训练流程执行一次联合 `backward()`
- **THEN** HPEC 分类梯度 MUST 能经过 `z_global`、`H_gcn` 和 `A_learned` 回传
- **AND** 系统 MUST NOT 在模块 3 与模块 2 之间新增阻断该梯度的默认逻辑

### Requirement: 模块 3 缓存支持模块 4 可视化

模块 3 SHALL 保持并扩展现有缓存，使模块 4 可视化能同时查看 HGCN 表示与 HPEC energy。

#### Scenario: 提供 HGCN 对照中间量

- **WHEN** 用户显式开启 S-DeCI 中间量可视化
- **THEN** 系统 MUST 能读取 `C_clipped`、`H0` 或 Poincare 投影结果、`H_gcn`、`z_global` 和 `logmap0(z_global)`
- **AND** 这些中间量 MUST 能与模块 4 的 prototype、angle 和 energy 在同一批次诊断中对照保存
