## ADDED Requirements

### Requirement: 每类多 prototype 表示

系统 SHALL 将 HPEC 模块 4 的类别原型从每类单个 prototype 扩展为每类多个 prototype，用于表达同一诊断类别内部的多种连接模式。

#### Scenario: 初始化多 prototype

- **GIVEN** `use_hpec_module4 == 1`
- **AND** `hpec_prototypes_per_class > 1`
- **WHEN** 模型初始化 HPEC 模块 4
- **THEN** prototype 张量 MUST 使用形状 `[classes, hpec_prototypes_per_class, hgcn_hidden_dim]`
- **AND** 每个 prototype MUST 被投影或约束在 HPEC 使用的 Poincare Ball 有效区域内

#### Scenario: 单 prototype 回退

- **GIVEN** `use_hpec_module4 == 1`
- **AND** `hpec_prototypes_per_class == 1`
- **WHEN** 模型执行 forward 和 loss 计算
- **THEN** 行为 MUST 退化为接近当前每类单 prototype 的 HPEC 分类路径
- **AND** 该回退路径 MUST 不要求用户修改模型源码

### Requirement: 多 prototype 类别能量聚合

系统 SHALL 对每个样本与每个类别下的多个 prototype 计算 prototype-level energy，并将其聚合为类别级 energy 用于预测和指标计算。

#### Scenario: 计算 prototype-level energy

- **GIVEN** `z_global` 的形状为 `[B, hgcn_hidden_dim]`
- **AND** prototype 张量形状为 `[classes, K, hgcn_hidden_dim]`
- **WHEN** HPEC 模块 4 执行 forward
- **THEN** 系统 MUST 计算形状为 `[B, classes, K]` 的 prototype-level energy 或等价中间量
- **AND** 系统 MUST 保留该中间量供诊断和可视化使用

#### Scenario: 聚合类别级 energy

- **GIVEN** prototype-level energy 已计算完成
- **WHEN** 模型需要输出分类结果
- **THEN** 系统 MUST 将 `[B, classes, K]` 聚合为 `[B, classes]` 的类别级 energy
- **AND** 预测类别 MUST 通过类别级 energy 的 `argmin` 或等价 energy-based 规则得到
- **AND** `S-DeCI.forward()` MUST 保持返回与当前训练指标兼容的 score/logit 张量

### Requirement: 多 prototype 最大似然损失

系统 SHALL 支持基于样本与多 prototype 相似度的最大似然损失 `L_mle`，用于提高样本对同类 prototype 分布的匹配概率。

#### Scenario: 计算 L_mle

- **GIVEN** 训练 batch 中存在 `z_global` 或 `logmap0(z_global)`
- **AND** 存在真实类别 label
- **WHEN** `lambda_hpec_mle > 0`
- **THEN** 系统 MUST 计算 `L_mle`
- **AND** `L_mle` MUST 使用同类 prototype 相似度作为正项
- **AND** `L_mle` MUST 使用异类 prototype 相似度作为竞争项
- **AND** `L_mle` MUST 能参与 PyTorch autograd 反向传播

#### Scenario: 关闭 L_mle

- **GIVEN** `lambda_hpec_mle == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** `L_mle` MUST 不影响总 loss

### Requirement: prototype contrastive loss

系统 SHALL 支持 prototype contrastive loss `L_pcl`，用于约束不同类别 prototype 的可分性，并避免 prototype 空间缺少结构约束。

#### Scenario: 计算 L_pcl

- **GIVEN** HPEC 模块 4 已初始化多 prototype
- **WHEN** `lambda_hpec_pcl > 0`
- **THEN** 系统 MUST 计算 `L_pcl`
- **AND** `L_pcl` MUST 使用 prototype 之间的相似度或距离
- **AND** `L_pcl` MUST 区分同类 prototype 和异类 prototype 的关系
- **AND** `L_pcl` MUST 能参与 PyTorch autograd 反向传播

#### Scenario: 避免 prototype collapse

- **GIVEN** `hpec_prototypes_per_class > 1`
- **WHEN** 系统计算或设计 `L_pcl`
- **THEN** `L_pcl` MUST 不强制同一类别内所有 prototype 完全重合
- **AND** 系统 SHOULD 通过较小权重、margin、温度或其他稳定策略保留类内多 prototype 的多样性

### Requirement: prototype alignment loss

系统 SHALL 支持 prototype alignment loss `L_pal`，用于使每个样本靠近其真实类别下最匹配的 prototype。

#### Scenario: 计算 L_pal

- **GIVEN** 训练 batch 中存在样本表示和真实类别 label
- **AND** 每个类别存在一个或多个 prototype
- **WHEN** `lambda_hpec_pal > 0`
- **THEN** 系统 MUST 为每个样本在真实类别的 prototype 中选择最匹配 prototype
- **AND** 系统 MUST 计算样本表示与该 prototype 的距离损失
- **AND** `L_pal` MUST 能参与 PyTorch autograd 反向传播

#### Scenario: 关闭 L_pal

- **GIVEN** `lambda_hpec_pal == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** `L_pal` MUST 不影响总 loss

### Requirement: 多 prototype 中间量可诊断

系统 SHALL 缓存多 prototype HPEC 的关键中间量，便于分析原型分布、样本匹配关系和分类效果。

#### Scenario: 缓存多 prototype 输出

- **GIVEN** HPEC 模块 4 完成 forward
- **WHEN** 开发者读取模型缓存或可视化函数
- **THEN** 系统 MUST 能读取多 prototype 张量
- **AND** 系统 MUST 能读取 prototype-level energy 或相似度
- **AND** 系统 MUST 能读取类别级 energy、预测类别和 probability 或等价 score

#### Scenario: 缓存 prototype loss

- **GIVEN** 训练流程计算了 `L_mle`、`L_pcl` 或 `L_pal`
- **WHEN** 开发者读取模型 loss 诊断量
- **THEN** 系统 MUST 能分别读取这些 loss 的未加权值
- **AND** 系统 MUST 能读取它们加入总 loss 后的加权贡献
