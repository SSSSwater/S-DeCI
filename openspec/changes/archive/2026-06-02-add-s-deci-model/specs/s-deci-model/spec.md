## ADDED Requirements

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

### Requirement: 仅使用 Cycle/seasonal 特征分类
新模型 SHALL 在当前阶段仅使用 Cycle/seasonal 分支进行分类，不使用 Drift/trend logits 参与最终预测。

#### Scenario: 聚合 seasonal logits
- **WHEN** 所有 block 执行完成
- **THEN** 新模型 MUST 聚合所有 seasonal logits 作为最终分类依据
- **AND** MUST 不将 trend logits 加入最终输出

#### Scenario: 输出兼容二分类
- **WHEN** `configs.classes == 2`
- **THEN** 新模型 MUST 输出形状为 `[B, 1]` 的概率
- **AND** MUST 与现有 MSE 二分类训练路径兼容

#### Scenario: 输出兼容多分类
- **WHEN** `configs.classes > 2`
- **THEN** 新模型 MUST 输出形状为 `[B, classes]` 的 logits
- **AND** MUST 与现有 CE 多分类训练路径兼容

### Requirement: 训练入口可选择新模型
系统 SHALL 将新模型注册到现有模型选择机制，使其可通过 `run_cv.py --model S-DeCI` 或训练测试脚本 CLI 调用。

#### Scenario: 模型注册
- **WHEN** `Exp_Basic` 构建 `model_dict`
- **THEN** `model_dict` MUST 包含新模型名称到新模型模块的映射

#### Scenario: 训练跑通
- **WHEN** 使用 `.venv` Python 运行低预算训练测试并选择新模型
- **THEN** 训练 MUST 完成至少一次 cross-validation 流程
- **AND** MUST 输出 accuracy、precision、recall、macro F1 和 ROC AUC 指标
