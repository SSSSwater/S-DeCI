# 项目结构规范文档

## 目的

本文档用于说明该 fMRI 脑疾病分类项目的仓库结构、模块职责和实验运行边界。本项目实际设计并提出的模型为 `models/DeCI.py` 中的 `DeCI` 模型，以及 `layers/DeCI_Layer.py` 中的 `DeCI_Block`；其余 `models/` 与 `layers/` 中的模型或组件主要用于性能对照实验。

## 范围

本文档覆盖以下内容：

- 顶层仓库结构。
- 实验执行流程。
- DeCI 主模型与对照模型代码之间的边界。
- 数据加载、训练、脚本、日志和结果汇总模块的职责。

本文档仅描述现有项目结构，不定义新功能，也不改变运行行为。

## 仓库顶层结构

### 要求：顶层目录组织

项目应当将源代码、实验入口、OpenSpec 元数据、脚本和文档放置在清晰分离的顶层目录或文件中。

#### 场景：查看仓库根目录

- **当** 开发者查看仓库根目录时
- **则** 根目录必须包含以下主要内容：
  - `README.md`：项目介绍、使用方法、引用信息和复现实验概览。
  - `requirements.txt`：Python 依赖列表。
  - `run_cv.py`：交叉验证训练主入口。
  - `hrun.py`：按实验组批量执行 shell 脚本的启动器。
  - `extract_re.py`：日志指标汇总工具。
  - `data_provider/`：数据集加载与交叉验证 dataloader 构建。
  - `exp/`：实验生命周期管理，包括模型构建、训练、验证与基线机器学习流程。
  - `models/`：模型定义目录，包含 DeCI 主模型与性能对照模型。
  - `layers/`：神经网络层实现目录，包含 DeCI 核心块与其他基线组件。
  - `scripts/`：面向不同数据集和模型族的实验 shell 脚本。
  - `utils/`：训练工具、评价指标和辅助函数。
  - `openspec/`：OpenSpec 配置与规范驱动流程元数据。
  - `docs/`：项目结构与模型设计文档。

## 核心运行流程

### 要求：实验入口

系统应当使用 `run_cv.py` 作为配置和运行交叉验证实验的主入口。

#### 场景：DeCI 脚本启动训练

- **当** `scripts/DeCI/` 下的 shell 脚本调用 `python -u run_cv.py` 时
- **则** `run_cv.py` 必须解析数据集、模型、优化、网络结构和运行时参数
- **并且** 必须实例化 `exp/exp_classification_CV.py` 中的 `Exp_Main`
- **并且** 必须执行重复的 k 折交叉验证训练流程。

### 要求：实验生命周期

实验层应当负责模型构建、设备选择、数据获取、训练、验证、checkpoint 管理和指标汇报。

#### 场景：创建实验对象

- **当** 构造 `Exp_Main(args)` 时
- **则** `Exp_Basic` 必须选择运行设备
- **并且** 必须根据 `args.model` 在 `models/` 中映射对应模型模块
- **并且** `Exp_Main._build_model()` 必须创建可训练模型和一份初始权重副本，用于每个 fold 开始前重置模型。

#### 场景：执行深度学习 k 折训练

- **当** `args.Method` 为 `DL` 时
- **则** `Exp_Main.kf_train()` 必须从 `data_provider.data_factory_CV.data_provider` 获取训练和验证 dataloader
- **并且** 必须在每个 fold 前重置模型
- **并且** 必须完成训练、验证、保存 checkpoint、加载最佳 checkpoint、汇报 fold 指标并返回平均指标。

#### 场景：执行传统机器学习基线

- **当** `args.Method` 选择 SVM 或 RF 相关行为时
- **则** `Exp_Main.kf_ML()` 必须执行传统机器学习对照流程
- **并且** 这些流程必须先从时序 batch 中计算功能连接特征，再训练 scikit-learn 模型。

## 数据提供模块结构

### 要求：数据集加载器职责

`data_provider/data_loader_CV.py` 中的数据集类应当根据不同数据集的目录结构加载预处理后的 fMRI 样本和标签。

#### 场景：数据集类加载样本

- **当** 实例化 `PPMI_Dataset`、`Abide_Dataset` 或 `ADNI_Dataset` 等数据集类时
- **则** 数据集类必须将样本数据填充到 `self.data`
- **并且** 必须将与样本对齐的类别标签填充到 `self.labels`
- **并且** `__getitem__` 必须按索引返回 `(data, label)`。

### 要求：交叉验证 dataloader 构建

`data_provider/data_factory_CV.py` 应当创建分层 k 折训练和验证 dataloader。

#### 场景：调用数据提供函数

- **当** 调用 `data_provider(args)` 时
- **则** 必须根据 `args.data` 选择对应数据集类
- **并且** 必须使用 `args.data_path`、`args.data_type`、`args.protocol` 和 `args.seq_len` 实例化数据集
- **并且** 必须使用 `StratifiedKFold` 根据数据和标签创建 fold 索引
- **并且** 必须返回对齐的训练 dataloader 列表和验证 dataloader 列表。

## 主模型边界

### 要求：DeCI 是项目提出的主模型

项目应当将 `models/DeCI.py` 和 `layers/DeCI_Layer.py` 视为实际设计的模型实现。

#### 场景：定位主模型实现

- **当** 开发者需要查看项目提出的方法时
- **则** 必须从 `models/DeCI.py` 开始
- **并且** 必须查看 `layers/DeCI_Layer.py` 以理解 DeCI 核心块、趋势提取和季节性提取逻辑。

### 要求：其他模型作为对照基线

除 `models/DeCI.py` 以外的模型文件，以及除 DeCI 专用层实现以外的层文件，应当被视为性能对照基线，除非它们被 DeCI 直接引用。

#### 场景：查看基线模型

- **当** 开发者打开 `iTransformer.py`、`TimesNet.py`、`PatchTST.py`、`BrainNetTF.py`、`STAGIN.py`、`PSCRAttn.py`、`SimMVF.py` 或 `MVHO.py` 等模型文件时
- **则** 这些文件必须被理解为 benchmark 或对照实现
- **并且** 不应被描述为本项目的主设计架构。

## 脚本与实验组

### 要求：脚本目录反映实验家族

`scripts/` 目录应当按照模型家族或对照类别组织复现实验命令。

#### 场景：使用批量启动器运行主模型

- **当** 执行 `python hrun.py --opt 1` 时
- **则** 必须选择 `scripts/DeCI/` 下的脚本作为主模型复现实验组。

#### 场景：使用批量启动器运行基线

- **当** 执行 `python hrun.py --opt 2`、`--opt 3`、`--opt 4`、`--opt 5` 或 `--opt 6` 时
- **则** 选中的脚本必须分别对应 FC-based、dFC-based、general time-series、multi-view 或 attention-based 对照实验组。

## 日志与结果汇总

### 要求：训练日志

实验脚本应当将运行输出重定向到 `logs/<dataset>/<model>/` 风格的目录中。

#### 场景：DeCI 数据集脚本运行

- **当** DeCI shell 脚本成功运行时
- **则** 必须创建对应模型的日志目录
- **并且** 必须将命令输出写入日志文件
- **并且** 日志文件名应记录 seed、batch size、learning rate、layer 数、dropout 和 `d_model` 等关键超参数。

### 要求：结果提取

`extract_re.py` 应当从 `logs/` 树中汇总最佳日志指标。

#### 场景：实验完成后运行结果提取

- **当** 在已有日志文件后执行 `python extract_re.py` 时
- **则** 必须扫描数据集/模型日志目录
- **并且** 必须提取 mean 和 std 指标
- **并且** 必须将每个模型的最佳结果写入该数据集日志目录下的 `total.log`。

## OpenSpec 与文档

### 要求：OpenSpec 配置

项目应当将 OpenSpec 工作流配置保存在 `openspec/` 下。

#### 场景：检查 OpenSpec 状态

- **当** 在没有活跃 change 的情况下执行 `openspec status` 时
- **则** 仓库应报告当前没有活跃变更。

### 要求：文档位置

项目级 Markdown 文档应当放在 `docs/` 下。

#### 场景：读者查找结构文档

- **当** 读者需要项目结构或模型设计说明时
- **则** 必须查看 `docs/`
- **并且** 将本文档与 `docs/deci-model-design-spec.md` 配合阅读。

## 已知结构说明

- 当前 `data_provider/data_factory_CV.py` 中 `Taowu` 被映射到 `Neurocon_Dataset`。本文档只描述当前结构，不修改该行为。
- 当前源码中部分注释和数据集名称存在编码显示问题。文档在涉及精确匹配时保留代码中的表示方式。
- 项目假设预处理后的数据集放置在 `./dataset` 下，但数据文件本身不包含在仓库中。
