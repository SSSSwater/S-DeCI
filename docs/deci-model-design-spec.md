# DeCI 模型设计规范文档

## 目的

本文档用于规范描述本项目中用于 fMRI 脑疾病分类的 DeCI 模型设计。模型实现分布在以下两个文件中：

- `models/DeCI.py`：顶层模型装配与 forward 流程。
- `layers/DeCI_Layer.py`：`DeCI_Block`、趋势提取、季节性提取和 block 级分类头。

除上述 DeCI 实现外，`models/` 中的其他模型文件以及 `layers/` 中的非 DeCI 层文件均应视为性能对照基线，除非它们被 DeCI 直接导入。

## 模型概要

DeCI 面向多变量 fMRI 时间序列分类任务。模型先将每个 ROI/channel 的时间维序列独立嵌入到潜在维度，然后堆叠一个或多个 DeCI block。每个 block 从当前残差表征中逐步提取趋势分量和季节性分量；每个分量分别通过独立分类头产生 logits；最终预测由所有趋势分量和季节性分量的 logits 相加得到。

该实现体现以下核心思想：

- channel-independent temporal embedding，即每个 ROI/channel 独立进行时间维嵌入。
- 可选的样本级时间维归一化。
- progressive residual decomposition，即逐层残差分解。
- trend component 与 seasonal component 使用独立分类头。
- 在 logit 层面对不同分解深度和不同分量类型进行融合。

## 配置接口

### 要求：必要模型配置

DeCI 模型应当由 `run_cv.py` 传入的实验参数对象进行配置。

#### 场景：构造模型

- **当** 实例化 `models.DeCI.Model(configs)` 时
- **则** `configs` 必须提供：
  - `seq_len`：输入时间长度。
  - `channel`：ROI/channel 变量数量。
  - `d_model`：潜在嵌入维度。
  - `layer`：堆叠的 `DeCI_Block` 数量。
  - `classes`：目标类别数量。
  - `dropout`：DeCI block 内部使用的 dropout 概率。
  - `use_norm`：控制是否启用输入归一化的标志。

### 要求：输出维度选择

模型应当在二分类任务中使用一维输出，在多分类任务中使用类别数维输出。

#### 场景：构造二分类模型

- **当** `configs.classes == 2` 时
- **则** 模型必须将 `out_dim` 设为 `1`
- **并且** forward 过程必须对最终 logit 和应用 sigmoid。

#### 场景：构造多分类模型

- **当** `configs.classes` 大于 `2` 时
- **则** 模型必须将 `out_dim` 设为 `configs.classes`
- **并且** forward 过程必须返回未归一化的类别 logits。

## 顶层 DeCI 模型

### 要求：输入张量契约

DeCI 模型应当接收形状为 `[B, T, N]` 的 fMRI 样本张量。

#### 场景：forward 开始执行

- **当** 调用 `Model.forward(x_enc)` 时
- **则** `x_enc` 必须被解释为：
  - `B`：batch size。
  - `T`：时间长度。
  - `N`：ROI/channel 变量数量。

### 要求：可选时间维归一化

模型应当在嵌入前可选地沿时间维对每个样本进行归一化。

#### 场景：启用归一化

- **当** `configs.use_norm` 为真时
- **则** 模型必须从 `x_enc` 中减去 detach 后的时间维均值
- **并且** 必须除以带 epsilon 稳定项的 detach 后时间维标准差。

#### 场景：关闭归一化

- **当** `configs.use_norm` 为假时
- **则** 模型必须不执行该归一化步骤，直接将输入值传入嵌入步骤。

### 要求：Channel-Independent 变量嵌入

模型应当将每个 ROI/channel 的时间序列独立嵌入到 `d_model` 维。

#### 场景：执行嵌入

- **当** 输入张量形状为 `[B, T, N]` 时
- **则** 模型必须先将其转置为 `[B, N, T]`
- **并且** 必须沿时间维应用 `nn.Linear(seq_len, d_model)`
- **并且** 必须得到形状为 `[B, N, D]` 的嵌入张量，其中 `D = d_model`。

### 要求：渐进式 DeCI Block 堆叠

模型应当将 `configs.layer` 个 DeCI block 依次作用于残差表征。

#### 场景：执行 block 堆叠

- **当** 嵌入表征进入 DeCI block 堆叠时
- **则** 每个 `DeCI_Block` 必须接收当前残差张量
- **并且** 必须返回 trend logits、seasonal logits 和更新后的残差张量
- **并且** 顶层模型必须收集所有 trend logits 与 seasonal logits。

### 要求：Logit 级融合

模型应当通过相加所有趋势 logits 与所有季节性 logits 来融合分量预测。

#### 场景：生成最终预测

- **当** 所有 DeCI block 执行完成后
- **则** 模型必须计算 `y_hat = sum(trends) + sum(seasonals)`
- **并且** 在二分类任务中必须返回 `sigmoid(y_hat)`
- **并且** 在多分类任务中必须直接返回 `y_hat`。

## DeCI Block

### 要求：Block 输入契约

`DeCI_Block` 应当处理形状为 `[B, N, D]` 的嵌入残差张量。

#### 场景：Block 接收输入

- **当** 调用 `DeCI_Block.forward(inp)` 时
- **则** `inp` 必须被解释为：
  - `B`：batch size。
  - `N`：ROI/channel 变量数量。
  - `D`：潜在嵌入维度。

### 要求：趋势分量提取

每个 DeCI block 应当使用 `Trend_ext` 从当前残差表征中提取趋势分量。

#### 场景：提取趋势分量

- **当** block 接收到 `inp` 时
- **则** 必须计算 `trend = Trend_ext(inp)`
- **并且** 必须计算第一次残差更新 `res = inp - trend`。

### 要求：季节性分量提取

每个 DeCI block 应当使用 `Seasonal_ext` 从去除趋势后的残差中提取季节性分量。

#### 场景：提取季节性分量

- **当** 第一次残差 `res` 可用时
- **则** block 必须计算 `seasonal = Seasonal_ext(res)`
- **并且** 必须计算下一步残差 `res = res - seasonal`。

### 要求：分量分类

每个 DeCI block 应当分别对趋势分量和季节性分量进行分类。

#### 场景：生成分量 logits

- **当** trend 和 seasonal 分量形状为 `[B, N, D]` 时
- **则** block 必须分别在 ROI/channel 维度上对两个分量取平均
- **并且** 必须将 trend 平均向量输入 `trend_classifier`
- **并且** 必须将 seasonal 平均向量输入 `seasonal_classifier`
- **并且** 必须返回两个分类器输出以及更新后的残差。

## 趋势提取器

### 要求：Depthwise 趋势卷积

`Trend_ext` 应当使用沿潜在维度的一维逐通道卷积估计平滑趋势分量。

#### 场景：初始化趋势提取器

- **当** 构造 `Trend_ext(channel, kernel_size, dropout)` 时
- **则** 必须创建 `nn.Conv1d`，其参数为：
  - `in_channels = channel`
  - `out_channels = channel`
  - `kernel_size = kernel_size`
  - `groups = channel`
  - `stride = 1`
  - `padding = 0`
- **并且** 必须使用 softmax 归一化后的全 1 kernel 初始化卷积权重
- **并且** 必须将 bias 初始化为 0。

### 要求：趋势提取器保持形状

`Trend_ext` 应当保持输入张量的 `[B, N, D]` 形状。

#### 场景：执行趋势提取器 forward

- **当** `Trend_ext.forward(inp)` 接收形状为 `[B, N, D]` 的张量时
- **则** 必须沿潜在维度在前端补充 `kernel_size - 1` 个 0
- **并且** 必须执行 depthwise convolution
- **并且** 必须返回经过 dropout 后、形状仍为 `[B, N, D]` 的张量。

## 季节性提取器

### 要求：门控季节性变换

`Seasonal_ext` 应当使用可学习 gate 与残差 MLP 来估计残差表征中的季节性分量。

#### 场景：执行季节性提取器 forward

- **当** `Seasonal_ext.forward(inp)` 接收形状为 `[B, N, D]` 的张量时
- **则** 必须通过两层 GELU MLP 计算 sigmoid gate weights
- **并且** 必须将输入与 gate weights 相乘
- **并且** 必须通过 `Gate_out` 投影门控后的值
- **并且** 必须与原始输入相加形成残差连接
- **并且** 必须使用 `LayerNorm` 进行归一化。

### 要求：季节性残差 MLP

`Seasonal_ext` 应当使用残差前馈块进一步细化门控表征。

#### 场景：执行季节性 MLP

- **当** 第一次归一化后的 seasonal embedding 可用时
- **则** 提取器必须应用包含 GELU 和 dropout 的两层 MLP
- **并且** 必须将 MLP 输出加回归一化后的 embedding
- **并且** 必须应用第二个 `LayerNorm`
- **并且** 必须返回形状为 `[B, N, D]` 的张量。

## 训练与损失兼容性

### 要求：二分类兼容性

DeCI 模型应当兼容 `Exp_Main` 中的二分类训练与验证路径。

#### 场景：执行二分类验证

- **当** `args.classes == 2` 时
- **则** DeCI 必须返回形状为 `[B, 1]` 的概率
- **并且** 实验代码必须将 squeeze 后的概率与 `0.5` 比较以得到预测类别。

### 要求：多分类兼容性

DeCI 模型应当兼容 `Exp_Main` 中的多分类训练与验证路径。

#### 场景：执行多分类验证

- **当** `args.classes > 2` 时
- **则** DeCI 必须返回形状为 `[B, classes]` 的 logits
- **并且** 实验代码必须应用 softmax 和 argmax 进行指标计算。

## 架构边界

### 要求：DeCI 不依赖基线模型

DeCI 实现应当只依赖 PyTorch 与 DeCI layer 模块。

#### 场景：检查 DeCI import

- **当** 查看 `models/DeCI.py` 时
- **则** 它必须导入 `torch`、`torch.nn` 和 `DeCI_Block`
- **并且** 不应依赖任何对照模型模块。

### 要求：基线保持可比较但与 DeCI 分离

对照模型应当仍可通过 `Exp_Basic.model_dict` 使用，但不应被视为 DeCI 子模块。

#### 场景：按名称选择模型

- **当** `args.model` 设置为 `DeCI` 时
- **则** 实验必须实例化 `models.DeCI.Model`
- **并且** DeCI 的 forward 过程不应需要 `models/` 下其他模型文件。

## 实现说明

- 当前代码中变量嵌入层命名为 `Variate_Embedding`。
- `models/DeCI.py` 中 block 区域的注释写作 `LiNo Block`，但实际实例化的模块是 `DeCI_Block`。
- `DeCI_Block(configs)` 会将 `configs.d_model` 作为 `Trend_ext` 的卷积 `kernel_size`。
- 分量 logits 在分类前是在 ROI/channel 维度上取平均，不是在 batch 或类别维度上取平均。
- 多分类训练当前会在 `Exp_Main` 中构造 one-hot label，并使用配置得到的 criterion；本文档描述 DeCI 的输出契约，不改变训练代码。
