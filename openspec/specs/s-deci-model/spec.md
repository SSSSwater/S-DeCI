## Purpose

定义 `S-DeCI` 模型的正式行为：它是基于现有 DeCI block 的独立模型文件，可在模块 1 输出后接入模块 2 因果图学习，并可进一步接入模块 3 HGCN 双曲 readout。启用模块 3 时，当前阶段直接使用 `logmap0(z_global)` 的分类头作为分类依据；关闭模块 3 时，保留原 Cycle/seasonal logits 分类路径。
## Requirements
### Requirement: 独立模块 1 模型文件

系统 SHALL 在 `models/` 下提供 `S-DeCI` 对应的独立模型文件，用于实现模块 1 的 Cycle 分支训练验证，并且不得直接修改现有 `models/DeCI.py` 的主模型逻辑。

#### Scenario: 新模型文件存在

- **WHEN** 开发者查看 `models/` 目录
- **THEN** MUST 能找到 `models/S_DeCI.py`
- **AND** 该文件 MUST 定义与现有模型一致的 `Model` 类

#### Scenario: DeCI 原始模型保持独立

- **WHEN** 新模型实现完成
- **THEN** `models/DeCI.py` 的主 `Model.forward()` 行为 MUST 不因该变更而改变

### Requirement: 复用 DeCI block

新模型 SHALL 复用现有 `layers.DeCI_Layer.DeCI_Block`，不得新建或复制 DeCI block。

#### Scenario: 构建模块 1 block 堆叠

- **WHEN** 新模型初始化
- **THEN** MUST 使用 `DeCI_Block(configs)` 构建 `configs.layer` 个 block
- **AND** MUST 不新增替代性的 DeCI block 类

### Requirement: 模块 1 前端结构

新模型 SHALL 保留 DeCI 前端中的 Channel-Independence embedding 和 Cycle/Drift decomposition 流程。

#### Scenario: 输入嵌入

- **WHEN** 新模型接收形状为 `[B, T, N]` 的输入
- **THEN** MUST 在可选归一化后转置为 `[B, N, T]`
- **AND** MUST 使用 `nn.Linear(seq_len, d_model)` 得到 `[B, N, d_model]` 嵌入

#### Scenario: 执行分解

- **WHEN** 嵌入进入 DeCI block 堆叠
- **THEN** 每个 block MUST 计算 trend logits、seasonal logits 和 residual
- **AND** residual MUST 传递给下一层 block

### Requirement: S-DeCI 接入模块 2 因果图学习

`S-DeCI` SHALL 在模块 1 产生 Cycle/seasonal feature 后接入模块 2 因果图学习，并且模块 2 输入 MUST 保持 `[B, N, d_model]` 的节点特征语义。

#### Scenario: 使用 Cycle feature 作为模块 2 输入

- **WHEN** `S-DeCI` 执行 forward 并完成 DeCI block 分解
- **THEN** 模型 MUST 从 block 内部或等价路径获得形状为 `[B, N, d_model]` 的 Cycle/seasonal feature
- **AND** 模型 MUST 将该 feature 输入模块 2 因果图学习组件

#### Scenario: 多层 Cycle feature 聚合

- **WHEN** `S-DeCI` 使用多个 DeCI block
- **THEN** 模型 MUST 支持将多个 block 的 Cycle/seasonal feature 聚合为模块 2 输入
- **AND** 默认聚合方式 MUST 与当前 seasonal logits 聚合语义保持一致

### Requirement: S-DeCI 分类路径可切换

`S-DeCI` SHALL 在启用模块 3 时使用模块 3 输出的 `z_global` 或 `logmap0(z_global)` 分类；在关闭模块 3 时保留原 Cycle/seasonal logits 分类路径。

#### Scenario: 模块 3 关闭时聚合 seasonal logits

- **GIVEN** `use_hgcn_module3 == 0`
- **WHEN** 所有 block 执行完成
- **THEN** 新模型 MUST 聚合所有 seasonal logits 作为最终分类依据
- **AND** MUST 不将 trend logits 加入最终输出

#### Scenario: 二分类使用 z_global

- **GIVEN** `use_hgcn_module3 == 1`
- **WHEN** `configs.classes == 2`
- **THEN** 模型 MUST 使用 `z_global` 或 `logmap0(z_global)` 计算二分类输出
- **AND** `S-DeCI.forward()` MUST 返回形状为 `[B, 1]` 的分类概率
- **AND** 返回值 MUST 与现有 MSE 二分类训练路径兼容

#### Scenario: 多分类使用 z_global

- **GIVEN** `use_hgcn_module3 == 1`
- **WHEN** `configs.classes > 2`
- **THEN** 模型 MUST 使用 `z_global` 或 `logmap0(z_global)` 计算多分类 logits
- **AND** `S-DeCI.forward()` MUST 返回形状为 `[B, classes]` 的 logits
- **AND** 返回值 MUST 与现有 CE 多分类训练路径兼容

### Requirement: S-DeCI 暴露模块 2 辅助损失

`S-DeCI` SHALL 在 forward 后暴露模块 2 的辅助损失和诊断量，使训练流程能够将因果学习 loss 纳入总 loss，同时不改变 `forward()` 的主返回值。

#### Scenario: 暴露因果学习 loss

- **WHEN** `S-DeCI` 开启模块 2 并完成一次 forward
- **THEN** 模型 MUST 能提供 reconstruction loss、DAG acyclicity loss 和 L1 sparsity loss
- **AND** 这些 loss MUST 能参与 PyTorch autograd 反向传播

#### Scenario: forward 返回值不变

- **WHEN** 训练流程调用 `y_hat = model(x_enc)`
- **THEN** `S-DeCI.forward()` MUST 只返回分类输出 `y_hat`
- **AND** 模块 2 的辅助 loss MUST 通过模型属性或方法读取

### Requirement: S-DeCI 模块 2/3 联合损失遵守设计文档

`S-DeCI` SHALL 在启用模块 2 和模块 3 时使用由分类 loss、模块 2 reconstruction loss、DAG loss 和 L1 sparsity loss 组成的联合训练目标。

#### Scenario: 计算联合训练 loss

- **GIVEN** `S-DeCI` 已启用模块 2 和模块 3
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 等价于 `Loss_cls(logmap0(z_global), label) + alpha * Loss_Recon(C, C_hat) + lambda * Loss_DAG(A_learned) + gamma * L1(A_learned)`
- **AND** 总 loss MUST NOT 使用真实因果矩阵监督
- **AND** 二分类 MSE label MUST 与模型输出保持 `[B, 1]` 形状兼容，避免广播成错误损失

#### Scenario: 一次 backward 联合回传

- **GIVEN** 训练流程已得到分类 loss 和模块 2 辅助 loss
- **WHEN** 执行反向传播
- **THEN** 系统 MUST 将分类 loss 与模块 2 辅助 loss 合成总 loss 后执行一次 `backward()`
- **AND** 分类 loss MUST 能通过模块 3 使用的 `A_learned` 回传到模块 2 因果图学习参数
- **AND** 系统 MUST NOT 提供阻断该分类梯度到模块 2 因果图的配置开关

### Requirement: S-DeCI 关键逻辑中文注释

`S-DeCI` SHALL 在本次新增或改动的关键逻辑处提供简洁中文注释，必要英文关键词可以保留。

#### Scenario: 注释模块 2 接入逻辑

- **WHEN** 开发者查看 `models/S_DeCI.py`
- **THEN** 模块 2 初始化、Cycle feature 聚合、辅助 loss 缓存和可视化触发相关代码 MUST 带有中文注释
- **AND** 注释 MUST 说明当前阶段分类仍只使用 Cycle/seasonal 分支

#### Scenario: 保留必要英文关键词

- **WHEN** 注释中涉及 `Cycle`、`seasonal`、`causal graph`、`DAG` 或 `adjacency`
- **THEN** 注释 MAY 保留这些英文关键词
- **AND** 注释 MUST 便于中文阅读和后续维护

### Requirement: 训练入口可选择新模型

系统 SHALL 将新模型注册到现有模型选择机制，使其可通过 `run_cv.py --model S-DeCI` 或训练测试脚本 CLI 调用。

#### Scenario: 模型注册

- **WHEN** `Exp_Basic` 构建 `model_dict`
- **THEN** `model_dict` MUST 包含新模型名称到新模型模块的映射

#### Scenario: 训练跑通

- **WHEN** 使用 `.venv` Python 运行低预算训练测试并选择新模型
- **THEN** 训练 MUST 完成至少一次 cross-validation 流程
- **AND** MUST 输出 accuracy、precision、recall、macro F1 和 ROC AUC 指标

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

### Requirement: S-DeCI 在模块 2 关闭时使用样本相关矩阵

`S-DeCI` SHALL 在 `use_causal_module2=0` 且 `use_hgcn_module3=1` 时，使用输入 batch 对应的样本相关系数矩阵作为模块 3 adjacency。

#### Scenario: 模块 2 关闭且模块 3 开启

- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 接收到 `correlation_matrix`
- **THEN** 模型 MUST 将 `correlation_matrix` 传入模块 3
- **AND** 模型 MUST NOT 初始化或调用模块 2 因果学习器
- **AND** 模型 MUST NOT 计算 reconstruction、DAG 或 L1 auxiliary loss

#### Scenario: 模块 2 关闭但缺少相关矩阵

- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 未接收到 `correlation_matrix`
- **THEN** 模型 MUST 以清晰错误失败
- **AND** 错误信息 MUST 说明模块 3 需要 sample correlation adjacency 或启用模块 2

#### Scenario: 模块 2 开启时保持原行为

- **GIVEN** `use_causal_module2 == 1`
- **WHEN** `S-DeCI.forward()` 执行
- **THEN** 模型 MUST 继续使用模块 2 产生的 `A_learned` 作为模块 3 adjacency
- **AND** 模型 MUST 继续暴露模块 2 auxiliary loss
- **AND** 输入的 `correlation_matrix` MUST NOT 替代 `A_learned`

### Requirement: S-DeCI 支持模块 4 多 prototype 参数

`S-DeCI` SHALL 在启用模块 4 HPEC 时支持每类多 prototype 配置，并将相关超参数传递给 HPEC 层。

#### Scenario: 传入多 prototype 配置

- **GIVEN** 用户通过训练入口设置 `hpec_prototypes_per_class`
- **WHEN** `S-DeCI` 初始化模块 4
- **THEN** 模型 MUST 将 `hpec_prototypes_per_class` 传递给 HPEC 模块 4
- **AND** 模型 MUST 支持配置 `hpec_proto_temperature`
- **AND** 模型 MUST 支持配置 `lambda_hpec_mle`、`lambda_hpec_pcl` 和 `lambda_hpec_pal`

#### Scenario: 模块 4 关闭时不创建多 prototype

- **GIVEN** `use_hpec_module4 == 0`
- **WHEN** `S-DeCI` 初始化
- **THEN** 模型 MUST NOT 初始化多 prototype HPEC 层
- **AND** 新增 prototype loss MUST 不参与训练

### Requirement: S-DeCI 暴露多 prototype loss

`S-DeCI` SHALL 在 forward 后基于 label 计算并暴露多 prototype 相关 loss，使训练流程能够将其加入总 loss。

#### Scenario: 计算 prototype loss

- **GIVEN** `S-DeCI` 已启用模块 3 和模块 4
- **AND** 模块 4 已完成一次 forward
- **WHEN** 训练流程调用模型的 label-aware loss 计算方法
- **THEN** 模型 MUST 能计算 `L_mle`、`L_pcl` 和 `L_pal`
- **AND** 模型 MUST 按配置权重得到 prototype auxiliary loss
- **AND** prototype auxiliary loss MUST 能与 HPEC primary loss 和模块 2 auxiliary loss 一起反向传播

#### Scenario: 总 loss 保持联合训练结构

- **GIVEN** 模块 2、模块 3 和模块 4 均已启用
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 `Loss_HPEC + alpha * Loss_Recon + lambda * Loss_DAG + gamma * L1`
- **AND** 当对应权重大于 0 时 MUST 额外包含 `L_mle`、`L_pcl` 和 `L_pal`
- **AND** 总 loss MUST NOT 使用真实因果矩阵监督

### Requirement: S-DeCI 模块 1 可禁用

`S-DeCI` SHALL 支持通过配置禁用模块 1 的 DeCI/Cycle 分解，并在禁用后直接从原始时间序列生成节点特征。

#### Scenario: 模块 1 启用时保持现有 Cycle 路径
- **GIVEN** `use_deci_module1 == 1`
- **WHEN** `S-DeCI.forward()` 接收形状为 `[B, T, N]` 的输入
- **THEN** 模型 MUST 执行现有 DeCI block 流程
- **AND** 模型 MUST 继续以 Cycle/seasonal feature 作为模块 2、模块 3/4 或 fallback 分类路径的节点特征来源

#### Scenario: 模块 1 禁用时使用 raw projection
- **GIVEN** `use_deci_module1 == 0`
- **WHEN** `S-DeCI.forward()` 接收形状为 `[B, T, N]` 的输入
- **THEN** 模型 MUST NOT 调用 DeCI block
- **AND** 模型 MUST NOT 提取高频、trend、seasonal 或 residual
- **AND** 模型 MUST 将原始时间序列转为 `[B, N, T]` 后投影为 `[B, N, d_model]`
- **AND** 投影后的 raw feature MUST 能作为模块 2、模块 3/4 或 GCN fallback 的节点特征输入

#### Scenario: 模块 1 禁用时可视化 raw feature
- **GIVEN** `use_deci_module1 == 0`
- **AND** 显式启用中间量可视化
- **WHEN** 模型完成 forward
- **THEN** 模型 MUST 缓存 raw projected feature
- **AND** 可视化标题或文件名 MUST 表明该特征不是 Cycle/seasonal feature

### Requirement: S-DeCI 模块开关组合约束

`S-DeCI` SHALL 对模块 1、模块 2、模块 3/4 的开关组合进行归一化与校验，使训练路径明确且可复现。

#### Scenario: 模块 3 和模块 4 联合启用
- **GIVEN** `use_hyperbolic_modules34 == 1`
- **WHEN** 模型初始化
- **THEN** 模型 MUST 使用 HGCN readout 与 HPEC energy/prototype 分类路径
- **AND** 若实现仍保留 `use_hgcn_module3` 和 `use_hpec_module4`，二者 MUST 被设置为一致的启用状态

#### Scenario: 模块 3 和模块 4 联合禁用
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** 模型初始化
- **THEN** 模型 MUST NOT 初始化 HGCN readout
- **AND** 模型 MUST NOT 初始化 HPEC energy/prototype 分类器
- **AND** 模型 MUST 初始化 GCN fallback 分类路径

#### Scenario: 拒绝不一致的旧参数组合
- **GIVEN** 用户同时传入 `use_hyperbolic_modules34`、`use_hgcn_module3` 或 `use_hpec_module4`
- **WHEN** 参数组合表达出 HGCN 与 HPEC 不一致的状态
- **THEN** 系统 MUST 归一化为 `use_hyperbolic_modules34` 的值或清晰失败
- **AND** 错误信息 MUST 说明模块 3 与模块 4 在本设计中需要联合启用或联合禁用

### Requirement: S-DeCI 根据模块开关选择 loss

`S-DeCI` 训练流程 SHALL 根据当前模块开关组合选择分类 loss 与 auxiliary loss。

#### Scenario: 全模块启用时使用 HPEC 联合 loss
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 1`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 HPEC primary loss
- **AND** 总 loss MUST 包含模块 2 reconstruction、DAG 和 L1 auxiliary loss
- **AND** 若多 prototype loss 权重大于 0，总 loss MUST 包含对应 prototype auxiliary loss

#### Scenario: GCN fallback 且模块 2 启用时使用分类 loss 加因果辅助项
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 GCN fallback 分类 loss
- **AND** 总 loss MUST 包含模块 2 reconstruction、DAG 和 L1 auxiliary loss
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss

#### Scenario: GCN fallback 且模块 2 禁用时只使用分类 loss
- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 GCN fallback 分类 loss
- **AND** 总 loss MUST NOT 包含模块 2 auxiliary loss
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss

