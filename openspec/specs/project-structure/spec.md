# 项目结构规范

## Purpose

本文档用于说明该 fMRI 脑疾病分类项目的仓库结构、模块职责和实验运行边界。本项目实际设计并提出的模型为 `models/DeCI.py` 中的 `DeCI` 模型，以及 `layers/DeCI_Layer.py` 中的 `DeCI_Block`；其余 `models/` 与 `layers/` 中的模型或组件主要用于性能对照实验。

## Requirements

### Requirement: 规范范围

本规范 SHALL 覆盖以下内容：

- 顶层仓库结构。
- 实验执行流程。
- DeCI 主模型与对照模型代码之间的边界。
- 数据加载、训练、脚本、日志和结果汇总模块的职责。

本文档仅描述现有项目结构，不定义新功能，也不改变运行行为。

#### Scenario: 阅读项目结构规范

- **WHEN** 读者查看本文档
- **THEN** MUST 能理解顶层仓库结构、实验执行流程、模型边界、数据加载、训练、脚本、日志和结果汇总模块的职责

### Requirement: 顶层目录组织

项目 SHALL 将源代码、实验入口、OpenSpec 元数据、脚本和文档放置在清晰分离的顶层目录或文件中。

#### Scenario: 查看仓库根目录

- **WHEN** 开发者查看仓库根目录
- **THEN** 根目录 MUST 包含以下主要内容：`README.md`、`requirements.txt`、`run_cv.py`、`hrun.py`、`extract_re.py`、`data_provider/`、`exp/`、`models/`、`layers/`、`scripts/`、`utils/`、`openspec/` 和 `docs/`

### Requirement: 实验入口

系统 SHALL 使用 `run_cv.py` 作为配置和运行交叉验证实验的主入口。

#### Scenario: DeCI 脚本启动训练

- **WHEN** `scripts/DeCI/` 下的 shell 脚本调用 `python -u run_cv.py`
- **THEN** `run_cv.py` MUST 解析数据集、模型、优化、网络结构和运行时参数
- **AND** MUST 实例化 `exp/exp_classification_CV.py` 中的 `Exp_Main`
- **AND** MUST 执行重复的 k 折交叉验证训练流程

### Requirement: 实验生命周期

实验层 SHALL 负责模型构建、设备选择、数据获取、训练、验证、checkpoint 管理和指标汇报。

#### Scenario: 创建实验对象

- **WHEN** 构造 `Exp_Main(args)`
- **THEN** `Exp_Basic` MUST 选择运行设备
- **AND** MUST 根据 `args.model` 在 `models/` 中映射对应模型模块
- **AND** `Exp_Main._build_model()` MUST 创建可训练模型和一份初始权重副本，用于每个 fold 开始前重置模型

#### Scenario: 执行深度学习 k 折训练

- **WHEN** `args.Method` 为 `DL`
- **THEN** `Exp_Main.kf_train()` MUST 从 `data_provider.data_factory_CV.data_provider` 获取训练和验证 dataloader
- **AND** MUST 在每个 fold 前重置模型
- **AND** MUST 完成训练、验证、保存 checkpoint、加载最佳 checkpoint、汇报 fold 指标并返回平均指标

#### Scenario: 执行传统机器学习基线

- **WHEN** `args.Method` 选择 SVM 或 RF 相关行为
- **THEN** `Exp_Main.kf_ML()` MUST 执行传统机器学习对照流程
- **AND** 这些流程 MUST 先从时序 batch 中计算功能连接特征，再训练 scikit-learn 模型

### Requirement: 数据集加载器职责

`data_provider/data_loader_CV.py` 中的数据集类 SHALL 根据不同数据集的目录结构加载预处理后的 fMRI 样本和标签。

#### Scenario: 数据集类加载样本

- **WHEN** 实例化 `PPMI_Dataset`、`Abide_Dataset` 或 `ADNI_Dataset` 等数据集类
- **THEN** 数据集类 MUST 将样本数据填充到 `self.data`
- **AND** MUST 将与样本对齐的类别标签填充到 `self.labels`
- **AND** `__getitem__` MUST 按索引返回 `(data, label)`

### Requirement: 交叉验证 dataloader 构建

`data_provider/data_factory_CV.py` SHALL 创建分层 k 折训练和验证 dataloader。

#### Scenario: 调用数据提供函数

- **WHEN** 调用 `data_provider(args)`
- **THEN** MUST 根据 `args.data` 选择对应数据集类
- **AND** MUST 使用 `args.data_path`、`args.data_type`、`args.protocol` 和 `args.seq_len` 实例化数据集
- **AND** MUST 使用 `StratifiedKFold` 根据数据和标签创建 fold 索引
- **AND** MUST 返回对齐的训练 dataloader 列表和验证 dataloader 列表

### Requirement: 数据加载与时间长度对齐职责

项目 SHALL 通过 `data_provider/` 中的数据集类与 DataLoader 组织训练数据，并清晰区分“原始样本读取”和“batch 时间长度对齐”的职责。

#### Scenario: Dataset 读取原始时间序列

- **WHEN** 实例化 `Abide_Dataset`、`MDD_Dataset` 或其他时间序列数据集类
- **THEN** Dataset MUST 根据 `args.data_path`、`args.data_type` 和 `args.protocol` 匹配可用样本文件
- **AND** Dataset MUST NOT 因样本时间长度不等于 `args.seq_len` 而静默丢弃样本
- **AND** Dataset SHOULD 保留原始时间序列张量，供 collate 阶段统一对齐

#### Scenario: DataLoader 按 seq_len 对齐时间维

- **GIVEN** batch 中存在形状为 `[T, N]` 的时间序列样本
- **WHEN** `collate_fn` 使用 `args.seq_len` 组装 batch
- **THEN** 若 `T > args.seq_len`，系统 MUST 沿时间维截断到 `args.seq_len`
- **AND** 若 `T < args.seq_len`，系统 MUST 在时间维末尾补 0 到 `args.seq_len`
- **AND** 输出 batch 的时间序列形状 MUST 为 `[B, args.seq_len, N]`

#### Scenario: 相关矩阵保持图结构形状

- **GIVEN** batch 中包含样本级相关矩阵
- **WHEN** `collate_fn` 组装 batch
- **THEN** 系统 MUST 保持每个相关矩阵的 `[N, N]` 形状
- **AND** 系统 MUST NOT 对相关矩阵执行时间维截断或补零

### Requirement: DeCI 是项目提出的主模型

项目 SHALL 将 `models/DeCI.py` 和 `layers/DeCI_Layer.py` 视为实际设计的模型实现。

#### Scenario: 定位主模型实现

- **WHEN** 开发者需要查看项目提出的方法
- **THEN** MUST 从 `models/DeCI.py` 开始
- **AND** MUST 查看 `layers/DeCI_Layer.py` 以理解 DeCI 核心块、趋势提取和季节性提取逻辑

### Requirement: 其他模型作为对照基线

除 `models/DeCI.py` 以外的模型文件，以及除 DeCI 专用层实现以外的层文件，SHALL 被视为性能对照基线，除非它们被 DeCI 直接引用。

#### Scenario: 查看基线模型

- **WHEN** 开发者打开 `iTransformer.py`、`TimesNet.py`、`PatchTST.py`、`BrainNetTF.py`、`STAGIN.py`、`PSCRAttn.py`、`SimMVF.py` 或 `MVHO.py` 等模型文件
- **THEN** 这些文件 MUST 被理解为 benchmark 或对照实现
- **AND** 不应被描述为本项目的主设计架构

### Requirement: 脚本目录反映实验家族

`scripts/` 目录 SHALL 按照模型家族或对照类别组织复现实验命令。

#### Scenario: 使用批量启动器运行主模型

- **WHEN** 执行 `python hrun.py --opt 1`
- **THEN** MUST 选择 `scripts/DeCI/` 下的脚本作为主模型复现实验组

#### Scenario: 使用批量启动器运行基线

- **WHEN** 执行 `python hrun.py --opt 2`、`--opt 3`、`--opt 4`、`--opt 5` 或 `--opt 6`
- **THEN** 选中的脚本 MUST 分别对应 FC-based、dFC-based、general time-series、multi-view 或 attention-based 对照实验组

### Requirement: 训练日志

实验脚本 SHALL 将运行输出重定向到 `logs/<dataset>/<model>/` 风格的目录中。

#### Scenario: DeCI 数据集脚本运行

- **WHEN** DeCI shell 脚本成功运行
- **THEN** MUST 创建对应模型的日志目录
- **AND** MUST 将命令输出写入日志文件
- **AND** 日志文件名 SHOULD 记录 seed、batch size、learning rate、layer 数、dropout 和 `d_model` 等关键超参数

### Requirement: 结果提取

`extract_re.py` SHALL 从 `logs/` 树中汇总最佳日志指标。

#### Scenario: 实验完成后运行结果提取

- **WHEN** 在已有日志文件后执行 `python extract_re.py`
- **THEN** MUST 扫描数据集/模型日志目录
- **AND** MUST 提取 mean 和 std 指标
- **AND** MUST 将每个模型的最佳结果写入该数据集日志目录下的 `total.log`

### Requirement: OpenSpec 配置

项目 SHALL 将 OpenSpec 工作流配置保存在 `openspec/` 下。

#### Scenario: 检查 OpenSpec 状态

- **WHEN** 在没有活跃 change 的情况下执行 `openspec status`
- **THEN** 仓库 SHOULD 报告当前没有活跃变更

### Requirement: 文档位置

项目级 Markdown 文档 SHALL 放在 `docs/` 下。

#### Scenario: 读者查找结构文档

- **WHEN** 读者需要项目结构或模型设计说明
- **THEN** MUST 查看 `docs/`
- **AND** SHOULD 将 `docs/project-structure-spec.md` 与 `docs/deci-model-design-spec.md` 配合阅读

## 已知结构说明

- 当前 `data_provider/data_factory_CV.py` 中 `Taowu` 被映射到 `Neurocon_Dataset`。本文档只描述当前结构，不修改该行为。
- 当前源码中部分注释和数据集名称存在编码显示问题。文档在涉及精确匹配时保留代码中的表示方式。
- 项目假设预处理后的数据集放置在 `./dataset` 下，但数据文件本身不包含在仓库中。
