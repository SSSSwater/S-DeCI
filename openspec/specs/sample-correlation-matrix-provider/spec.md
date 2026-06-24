## Purpose

定义样本级相关系数矩阵加载能力：当 `S-DeCI` 关闭模块 2 因果图学习但启用模块 3 HGCN 时，系统能够为每个时间序列样本加载对应的 `[N, N]` correlation matrix 作为图结构。

## Requirements

### Requirement: 样本相关系数矩阵按需加载

系统 SHALL 在需要模块 2 关闭回退路径时，为每个时间序列样本加载对应的相关系数矩阵。

#### Scenario: 返回带相关矩阵的样本

- **GIVEN** 训练配置启用 `use_hgcn_module3`
- **AND** 训练配置关闭 `use_causal_module2`
- **WHEN** Dataset 加载一个时间序列样本
- **THEN** Dataset MUST 同时加载该样本对应的 correlation matrix
- **AND** `__getitem__` MUST 返回 `(x_enc, label, correlation_matrix)`
- **AND** `correlation_matrix` 的形状 MUST 为 `[N, N]`

#### Scenario: 保持原有二元组返回

- **GIVEN** 训练配置不需要样本相关矩阵
- **WHEN** Dataset 加载样本
- **THEN** Dataset MUST 保持返回 `(x_enc, label)`
- **AND** 旧训练流程 MUST 不因新增能力改变默认 batch 格式

### Requirement: 相关矩阵文件名兼容解析

系统 SHALL 根据同一 subject 和 protocol 解析对应的相关系数矩阵文件，并兼容多种文件名模式。

#### Scenario: 解析当前数据集命名

- **GIVEN** 时间序列文件名类似 `sub-control50030_AAL116_features_timeseries.mat`
- **WHEN** 系统查找相关矩阵
- **THEN** MUST 能解析并加载 `sub-control50030_AAL116_correlation_matrix.mat`

#### Scenario: 解析用户指定命名

- **GIVEN** 数据集中存在 `sub-xxx_xxx_features_sub_correlation_matrix.mat`
- **WHEN** 系统查找相关矩阵
- **THEN** MUST 能将该文件识别为对应 subject 的相关系数矩阵
- **AND** MUST 优先选择与当前 protocol 或 subject 最匹配的候选文件

#### Scenario: 相关矩阵缺失

- **GIVEN** 配置要求加载样本相关矩阵
- **WHEN** 某个样本没有可匹配的 correlation matrix 文件
- **THEN** 系统 MUST 以清晰错误失败
- **AND** 错误信息 MUST 包含时间序列文件路径和尝试匹配的相关矩阵模式

### Requirement: 相关矩阵内容读取与校验

系统 SHALL 从 `.mat` 文件读取相关矩阵，并在进入模型前完成基本校验。

#### Scenario: 读取 mat data key

- **GIVEN** 相关矩阵 `.mat` 文件包含 `data` key
- **WHEN** Dataset 读取该文件
- **THEN** MUST 使用 `data` key 作为相关矩阵内容
- **AND** MUST 转换为 `torch.float32`

#### Scenario: 读取 mat key 失败

- **GIVEN** 相关矩阵 `.mat` 文件不包含可识别的矩阵 key
- **WHEN** Dataset 尝试读取该文件
- **THEN** 系统 MUST 以清晰错误失败
- **AND** 错误信息 MUST 包含该 `.mat` 文件中的可用 keys

#### Scenario: 校验矩阵形状

- **GIVEN** 当前样本 ROI 数为 `N`
- **WHEN** Dataset 得到相关矩阵
- **THEN** 相关矩阵形状 MUST 为 `[N, N]`
- **AND** 形状不匹配时 MUST 以清晰错误失败
