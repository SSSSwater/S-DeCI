## ADDED Requirements

### Requirement: S-DeCI 使用模块 3 双曲中心分类

`S-DeCI` SHALL 在启用模块 3 时，使用模块 3 输出的 `z_global` 作为当前阶段分类依据。

#### Scenario: 二分类使用 z_global

- **GIVEN** `configs.classes == 2`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 完成模块 3 readout
- **THEN** 模型 MUST 使用 `z_global` 或 `logmap0(z_global)` 计算二分类输出
- **AND** 返回形状 MUST 为 `[B, 1]`
- **AND** 返回值 MUST 与现有 MSE 二分类训练路径兼容

#### Scenario: 多分类使用 z_global

- **GIVEN** `configs.classes > 2`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 完成模块 3 readout
- **THEN** 模型 MUST 使用 `z_global` 或 `logmap0(z_global)` 计算多分类 logits
- **AND** 返回形状 MUST 为 `[B, classes]`
- **AND** 返回值 MUST 与现有 CE 多分类训练路径兼容

#### Scenario: 模块 3 关闭时保留原分类路径

- **GIVEN** `use_hgcn_module3 == 0`
- **WHEN** `S-DeCI.forward()` 执行分类
- **THEN** 模型 MUST 使用原 Cycle/seasonal logits 分类路径
- **AND** 模型 MUST 继续兼容现有训练测试脚本

### Requirement: S-DeCI 模块 3 联合损失遵守新模块设计

`S-DeCI` SHALL 在启用模块 3 时使用由 `z_global` 分类损失、模块 2 reconstruction loss、DAG loss 和 L1 sparsity loss 组成的联合训练目标。

#### Scenario: 计算模块 3 联合训练 loss

- **GIVEN** `S-DeCI` 已启用模块 2 和模块 3
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 `Loss_cls(z_global, label)`
- **AND** 总 loss MUST 包含 `alpha * Loss_Recon(C, C_hat)`
- **AND** 总 loss MUST 包含 `lambda * Loss_DAG(A_learned)`
- **AND** 总 loss MUST 包含 `gamma * L1(A_learned)`
- **AND** 总 loss MUST NOT 使用真实因果矩阵监督

#### Scenario: 分类 loss 回传到模块 2 因果图

- **GIVEN** `S-DeCI` 已启用模块 2 和模块 3
- **WHEN** 对 `Loss_cls(z_global, label)` 执行反向传播
- **THEN** 梯度 MUST 能从模块 3 通过 `A_learned` 回传到模块 2 因果图学习参数
- **AND** 系统 MUST NOT 提供阻断该分类梯度到模块 2 因果图的配置开关

### Requirement: S-DeCI 模块 3 参数可配置

`S-DeCI` SHALL 提供模块 3 相关训练参数，使用户能够配置是否启用模块 3、双曲中心维度和 HGCN 基础超参数。

#### Scenario: CLI 配置模块 3

- **GIVEN** 用户运行 `run_cv.py` 或根目录训练测试脚本
- **WHEN** 用户传入模块 3 相关参数
- **THEN** 系统 MUST 能配置 `use_hgcn_module3`
- **AND** 系统 MUST 能配置 `hgcn_hidden_dim`
- **AND** 系统 MUST 能配置 HGCN 层数、曲率、Backclip 半径或等价超参数

### Requirement: S-DeCI 模块 3 新增逻辑使用中文注释

`S-DeCI` SHALL 为模块 3 初始化、数据流、联合损失和可视化缓存相关逻辑提供简洁中文注释。

#### Scenario: 查看模块 3 接入注释

- **GIVEN** 开发者查看 `models/S_DeCI.py`
- **WHEN** 阅读模块 3 相关代码
- **THEN** 模块 3 初始化、`z_global` 分类、联合损失缓存和可视化缓存逻辑 MUST 带有中文注释
- **AND** 注释 MAY 保留 `HGCN`、`Poincare`、`Mobius`、`Frechet mean`、`z_global` 等英文关键词

