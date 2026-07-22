## ADDED Requirements

### Requirement: 多 prototype 保持类内多样性

系统 SHALL 让每类多个 prototype 在 ABIDE 默认配置中保留类内多样性，而不是坍缩到相同位置。

#### Scenario: prototype diversity

- **GIVEN** `hpec_prototypes_per_class > 1`
- **WHEN** 模块 4 训练进行中
- **THEN** prototype 之间 MUST 受到多样性约束
- **AND** 同类 prototype MUST NOT 被强制完全重合

### Requirement: 多 prototype 支持 ABIDE 正则权重

系统 SHALL 为 ABIDE 默认训练路径提供更温和的多 prototype 权重设置。

#### Scenario: 温和 prototype loss

- **WHEN** ABIDE 默认脚本启用多 prototype
- **THEN** `lambda_hpec_mle`、`lambda_hpec_pcl`、`lambda_hpec_pal` MUST 可配置
- **AND** 默认权重 SHOULD 更偏向稳定而非强分离

### Requirement: 多 prototype 可诊断

系统 SHALL 记录多 prototype 与样本的匹配关系，便于分析类内异质性。

#### Scenario: 输出 prototype map

- **WHEN** 模块 4 完成一个 epoch
- **THEN** 系统 SHOULD 能导出 prototype assignment 或等价匹配表
- **AND** 该输出 SHOULD 支持 train/test 分开分析
