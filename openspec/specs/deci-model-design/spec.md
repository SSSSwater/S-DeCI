# DeCI 模型设计规范

## Purpose

本文档用于规范描述本项目中用于 fMRI 脑疾病分类的 DeCI 模型设计。模型实现分布在 `models/DeCI.py` 和 `layers/DeCI_Layer.py`。除上述 DeCI 实现外，`models/` 中的其他模型文件以及 `layers/` 中的非 DeCI 层文件均应视为性能对照基线，除非它们被 DeCI 直接导入。

## Requirements

### Requirement: 模型概要

DeCI 面向多变量 fMRI 时间序列分类任务。模型先将每个 ROI/channel 的时间维序列独立嵌入到潜在维度，然后堆叠一个或多个 DeCI block。每个 block 从当前残差表征中逐步提取趋势分量和季节性分量；每个分量分别通过独立分类头产生 logits；最终预测由所有趋势分量和季节性分量的 logits 相加得到。 ? Requirement SHALL ??????

#### Scenario: 使用 DeCI 进行 fMRI 分类

- **WHEN** 开发者查看 DeCI 模型设计
- **THEN** MUST 将 DeCI 理解为基于 ROI/channel 独立嵌入、趋势分量和季节性分量分解的 fMRI 分类模型
- **AND** MUST 将所有趋势 logits 与季节性 logits 的相加结果作为 DeCI 原始模型的最终预测

### Requirement: 必要模型配置

DeCI 模型应当由 `run_cv.py` 传入的实验参数对象进行配置。 ? Requirement SHALL ??????

#### Scenario: 构造模型

- **WHEN** 实例化 `models.DeCI.Model(configs)`
- **THEN** `configs` MUST 提供 `seq_len`、`channel`、`d_model`、`layer`、`classes`、`dropout` 和 `use_norm`

### Requirement: 输出维度选择

模型应当在二分类任务中使用一维输出，在多分类任务中使用类别数维输出。 ? Requirement SHALL ??????

#### Scenario: 构造二分类模型

- **WHEN** `configs.classes == 2`
- **THEN** 模型 MUST 将 `out_dim` 设为 `1`
- **AND** forward 过程 MUST 对最终 logit 和应用 sigmoid

#### Scenario: 构造多分类模型

- **WHEN** `configs.classes` 大于 `2`
- **THEN** 模型 MUST 将 `out_dim` 设为 `configs.classes`
- **AND** forward 过程 MUST 返回未归一化的类别 logits

### Requirement: 输入张量契约

DeCI 模型应当接收形状为 `[B, T, N]` 的 fMRI 样本张量。 ? Requirement SHALL ??????

#### Scenario: forward 开始执行

- **WHEN** 调用 `Model.forward(x_enc)`
- **THEN** `x_enc` MUST 被解释为 batch size、时间长度和 ROI/channel 变量数量组成的 `[B, T, N]` 张量

### Requirement: 可选时间维归一化

模型应当在嵌入前可选地沿时间维对每个样本进行归一化。 ? Requirement SHALL ??????

#### Scenario: 启用归一化

- **WHEN** `configs.use_norm` 为真
- **THEN** 模型 MUST 从 `x_enc` 中减去 detach 后的时间维均值
- **AND** MUST 除以带 epsilon 稳定项的 detach 后时间维标准差

#### Scenario: 关闭归一化

- **WHEN** `configs.use_norm` 为假
- **THEN** 模型 MUST 不执行该归一化步骤，直接将输入值传入嵌入步骤

### Requirement: Channel-Independent 变量嵌入

模型应当将每个 ROI/channel 的时间序列独立嵌入到 `d_model` 维。 ? Requirement SHALL ??????

#### Scenario: 执行嵌入

- **WHEN** 输入张量形状为 `[B, T, N]`
- **THEN** 模型 MUST 先将其转置为 `[B, N, T]`
- **AND** MUST 沿时间维应用 `nn.Linear(seq_len, d_model)`
- **AND** MUST 得到形状为 `[B, N, D]` 的嵌入张量，其中 `D = d_model`

### Requirement: 渐进式 DeCI Block 堆叠

模型应当将 `configs.layer` 个 DeCI block 依次作用于残差表征。 ? Requirement SHALL ??????

#### Scenario: 执行 block 堆叠

- **WHEN** 嵌入表征进入 DeCI block 堆叠
- **THEN** 每个 `DeCI_Block` MUST 接收当前残差张量
- **AND** MUST 返回 trend logits、seasonal logits 和更新后的残差张量
- **AND** 顶层模型 MUST 收集所有 trend logits 与 seasonal logits

### Requirement: Logit 级融合

模型应当通过相加所有趋势 logits 与所有季节性 logits 来融合分量预测。 ? Requirement SHALL ??????

#### Scenario: 生成最终预测

- **WHEN** 所有 DeCI block 执行完成后
- **THEN** 模型 MUST 计算 `y_hat = sum(trends) + sum(seasonals)`
- **AND** 在二分类任务中 MUST 返回 `sigmoid(y_hat)`
- **AND** 在多分类任务中 MUST 直接返回 `y_hat`

### Requirement: Block 输入契约

`DeCI_Block` 应当处理形状为 `[B, N, D]` 的嵌入残差张量。 ? Requirement SHALL ??????

#### Scenario: Block 接收输入

- **WHEN** 调用 `DeCI_Block.forward(inp)`
- **THEN** `inp` MUST 被解释为 `[B, N, D]`

### Requirement: 趋势分量提取

每个 DeCI block 应当使用 `Trend_ext` 从当前残差表征中提取趋势分量。 ? Requirement SHALL ??????

#### Scenario: 提取趋势分量

- **WHEN** block 接收到 `inp`
- **THEN** MUST 计算 `trend = Trend_ext(inp)`
- **AND** MUST 计算第一次残差更新 `res = inp - trend`

### Requirement: 季节性分量提取

每个 DeCI block 应当使用 `Seasonal_ext` 从去除趋势后的残差中提取季节性分量。 ? Requirement SHALL ??????

#### Scenario: 提取季节性分量

- **WHEN** 第一次残差 `res` 可用
- **THEN** block MUST 计算 `seasonal = Seasonal_ext(res)`
- **AND** MUST 计算下一步残差 `res = res - seasonal`

### Requirement: 分量分类

每个 DeCI block 应当分别对趋势分量和季节性分量进行分类。 ? Requirement SHALL ??????

#### Scenario: 生成分量 logits

- **WHEN** trend 和 seasonal 分量形状为 `[B, N, D]`
- **THEN** block MUST 分别在 ROI/channel 维度上对两个分量取平均
- **AND** MUST 将 trend 平均向量输入 `trend_classifier`
- **AND** MUST 将 seasonal 平均向量输入 `seasonal_classifier`
- **AND** MUST 返回两个分类器输出以及更新后的残差

### Requirement: Depthwise 趋势卷积

`Trend_ext` 应当使用沿潜在维度的一维逐通道卷积估计平滑趋势分量。 ? Requirement SHALL ??????

#### Scenario: 初始化趋势提取器

- **WHEN** 构造 `Trend_ext(channel, kernel_size, dropout)`
- **THEN** MUST 创建 `nn.Conv1d`，其中 `in_channels = channel`、`out_channels = channel`、`kernel_size = kernel_size`、`groups = channel`、`stride = 1`、`padding = 0`
- **AND** MUST 使用 softmax 归一化后的全 1 kernel 初始化卷积权重
- **AND** MUST 将 bias 初始化为 0

### Requirement: 趋势提取器保持形状

`Trend_ext` 应当保持输入张量的 `[B, N, D]` 形状。 ? Requirement SHALL ??????

#### Scenario: 执行趋势提取器 forward

- **WHEN** `Trend_ext.forward(inp)` 接收形状为 `[B, N, D]` 的张量
- **THEN** MUST 沿潜在维度在前端补充 `kernel_size - 1` 个 0
- **AND** MUST 执行 depthwise convolution
- **AND** MUST 返回经过 dropout 后、形状仍为 `[B, N, D]` 的张量

### Requirement: 门控季节性变换

`Seasonal_ext` 应当使用可学习 gate 与残差 MLP 来估计残差表征中的季节性分量。 ? Requirement SHALL ??????

#### Scenario: 执行季节性提取器 forward

- **WHEN** `Seasonal_ext.forward(inp)` 接收形状为 `[B, N, D]` 的张量
- **THEN** MUST 通过两层 GELU MLP 计算 sigmoid gate weights
- **AND** MUST 将输入与 gate weights 相乘
- **AND** MUST 通过 `Gate_out` 投影门控后的值
- **AND** MUST 与原始输入相加形成残差连接
- **AND** MUST 使用 `LayerNorm` 进行归一化

### Requirement: 季节性残差 MLP

`Seasonal_ext` 应当使用残差前馈块进一步细化门控表征。 ? Requirement SHALL ??????

#### Scenario: 执行季节性 MLP

- **WHEN** 第一次归一化后的 seasonal embedding 可用
- **THEN** 提取器 MUST 应用包含 GELU 和 dropout 的两层 MLP
- **AND** MUST 将 MLP 输出加回归一化后的 embedding
- **AND** MUST 应用第二个 `LayerNorm`
- **AND** MUST 返回形状为 `[B, N, D]` 的张量

### Requirement: 二分类兼容性

DeCI 模型应当兼容 `Exp_Main` 中的二分类训练与验证路径。 ? Requirement SHALL ??????

#### Scenario: 执行二分类验证

- **WHEN** `args.classes == 2`
- **THEN** DeCI MUST 返回形状为 `[B, 1]` 的概率
- **AND** 实验代码 MUST 将 squeeze 后的概率与 `0.5` 比较以得到预测类别

### Requirement: 多分类兼容性

DeCI 模型应当兼容 `Exp_Main` 中的多分类训练与验证路径。 ? Requirement SHALL ??????

#### Scenario: 执行多分类验证

- **WHEN** `args.classes > 2`
- **THEN** DeCI MUST 返回形状为 `[B, classes]` 的 logits
- **AND** 实验代码 MUST 应用 softmax 和 argmax 进行指标计算

### Requirement: DeCI 不依赖基线模型

DeCI 实现应当只依赖 PyTorch 与 DeCI layer 模块。 ? Requirement SHALL ??????

#### Scenario: 检查 DeCI import

- **WHEN** 查看 `models/DeCI.py`
- **THEN** 它 MUST 导入 `torch`、`torch.nn` 和 `DeCI_Block`
- **AND** 不应依赖任何对照模型模块

### Requirement: 基线保持可比较但与 DeCI 分离

对照模型应当仍可通过 `Exp_Basic.model_dict` 使用，但不应被视为 DeCI 子模块。 ? Requirement SHALL ??????

#### Scenario: 按名称选择模型

- **WHEN** `args.model` 设置为 `DeCI`
- **THEN** 实验 MUST 实例化 `models.DeCI.Model`
- **AND** DeCI 的 forward 过程不应需要 `models/` 下其他模型文件

## 实现说明

- 当前代码中变量嵌入层命名为 `Variate_Embedding`。
- `models/DeCI.py` 中 block 区域的注释写作 `LiNo Block`，但实际实例化的模块是 `DeCI_Block`。
- `DeCI_Block(configs)` 会将 `configs.d_model` 作为 `Trend_ext` 的卷积 `kernel_size`。
- 分量 logits 在分类前是在 ROI/channel 维度上取平均，不是在 batch 或类别维度上取平均。
- 多分类训练当前会在 `Exp_Main` 中构造 one-hot label，并使用配置得到的 criterion；本文档描述 DeCI 的输出契约，不改变训练代码。
