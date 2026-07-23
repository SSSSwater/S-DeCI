## Purpose

定义 `S-DeCI` 模型的正式行为：它是基于现有 DeCI block 的独立模型文件，可在模块 1 输出后接入模块 2 时序因果图学习，并进一步接入模块 3/4 双曲路径。当前默认主路线为 `hgcn_hpec`：模块 1 提取 ALFF/fALFF 低频生理节点特征，模块 2 使用 Temporal NTS-NOTEARS 从历史 BOLD 预测未来 BOLD 并学习 $A_{\mathrm{lag}}$，模块 3 使用 Poincare HGCN 生成 $z_{\mathrm{global}}$，模块 4 使用 HPEC 多原型 energy 形成双曲原型证据。`lp_brain_hpec`、attention-guided temporal learner、旧静态 DAG 等路径只作为对照或 legacy/debug，不作为默认主路线。
## Requirements

### Requirement: S-DeCI 文档必须使用中文详细公式化描述默认主路线

S-DeCI 的 `docs/` 与 `openspec/specs/` Markdown 文档 SHALL 使用中文描述默认主路线，必要英文关键词可以保留。所有核心计算步骤 MUST 写成 LaTeX 公式，并写明输入输出维度、参考来源或原理、以及为什么在本项目中这样设计。

#### Scenario: 核心步骤必须包含公式、维度和来源

- **WHEN** 文档描述模块 1、模块 2、模块 3、模块 4 或联合损失
- **THEN** 每个核心步骤 MUST 包含 LaTeX 公式
- **AND** MUST 写清输入张量和输出张量的形状
- **AND** MUST 写清该步骤的来源或原理，例如 ALFF/fALFF、Granger causality、NOTEARS/NTS-NOTEARS、HGCN、Poincare Ball、HPEC 或 prototype learning
- **AND** MUST 写清该步骤为什么适合当前 fMRI 分类任务
- **AND** 若该步骤包含多个子计算，MUST 分别写出每个子计算的输入、公式、输出和形状，不能只写最终结果

#### Scenario: 公式与过程描述必须可直接阅读

- **WHEN** 文档描述任意默认主路线步骤
- **THEN** 公式 MUST 使用 Markdown LaTeX 语法，行内公式使用 `$...$`，独立公式使用 `$$...$$`
- **AND** 文档 MUST NOT 只用代码块、伪代码或普通文本替代核心数学公式
- **AND** 文档 MUST NOT 使用无法渲染的半截 LaTeX，例如缺少右花括号的 `\mathrm{...}`、未闭合的上下标或未成对的 `$$`
- **AND** 每个步骤 MUST 按“输入、计算过程、输出、来源或原理、设计原因”的顺序或等价结构说明
- **AND** 每个核心公式后 MUST 有中文解释，说明公式中关键变量代表什么、该公式来自什么原理、为什么该步骤需要出现在 S-DeCI 中
- **AND** 若某一步是工程控制、诊断量或消融开关而不是模型核心计算，文档 MUST 明确说明它不直接构成默认方法贡献
- **AND** 若某个 loss 权重默认为 `0`，文档 MUST 说明该项只作为可选诊断或消融项，不得写成默认训练贡献

#### Scenario: 后续 spec 文档必须按五段式描述可执行步骤

- **WHEN** 任意 S-DeCI 相关 proposal、design、tasks、主规范或 `docs/` Markdown 文档新增一个会改变 forward、loss、诊断量、可视化输出或训练指标的步骤
- **THEN** 该步骤 MUST 按“输入、计算过程、输出、来源或原理、设计原因”五段式描述
- **AND** 输入部分 MUST 写清张量符号、维度和语义，例如 `$X_{\mathrm{temp}}\in\mathbb{R}^{B\times T\times N}$`
- **AND** 计算过程 MUST 写出完整 Markdown LaTeX 公式，不能只写“融合”“投影”“加入损失”“计算能量”等概念词
- **AND** 输出部分 MUST 写清输出张量形状，以及该输出进入哪个后续模块或训练流程
- **AND** 来源或原理 MUST 写明来自哪类论文思想、数学原理、开源参考或工程稳定性原则
- **AND** 设计原因 MUST 写明为什么该公式适合当前 fMRI / S-DeCI 任务
- **AND** 若该步骤只用于 TensorBoard、控制台打印或图片保存，MUST 明确标注“不参与训练”
- **AND** 若该步骤参与训练，MUST 写明对应 loss 是否是未加权项，以及如何加权进入 $\mathcal{L}_{\mathrm{total}}$

#### Scenario: 默认主路线不得省略公式步骤

- **WHEN** 文档描述默认主路线中的任意数据流箭头、模块内部子步骤或 loss 项
- **THEN** 文档 MUST 写出当前符号下的完整 LaTeX 公式
- **AND** MUST NOT 使用“同理”“类似”“略”“见上文”替代核心公式
- **AND** MUST NOT 使用普通文本形式的 `A->B`、`sum(...)`、`lambda * loss` 替代 Markdown LaTeX
- **AND** 若某一步复用前文公式，仍 MUST 写清该步骤当前输入、当前输出和当前符号，例如 `$x_{b,t-\ell,i}$`、`$A_{\mathrm{lag}}^{(\ell)}[i,j]$` 和 `$\hat{x}_{b,t,j}$`
- **AND** 每个公式后 MUST 写明参考来源或原理，以及为什么该公式适合当前 fMRI / S-DeCI 设计
- **AND** 若文档描述的是后续可执行设计、可 apply 的变更或实验对照方案，也 MUST 写出完整公式、维度、来源或原理和设计原因；只有明确放在“裁剪清单”的历史分支 MAY 不展开公式

#### Scenario: 总数据流必须展开为逐步来源表

- **WHEN** 文档给出 S-DeCI 总流程图或总数据流
- **THEN** 文档 MUST 同时提供逐步表格或等价段落
- **AND** 每一步 MUST 包含输入形状、核心公式、输出形状、来源或原理、设计原因
- **AND** 表格 MUST 至少覆盖模块 1 低频生理特征、模块 2 时序因果图、分类图融合、模块 3 双曲图表示、模块 4 原型能量、最终预测
- **AND** 若某一步使用诊断量、TensorBoard 指标或可视化指标，文档 MUST 写明该量是否参与训练；参与训练的必须进入总 loss 公式，不参与训练的必须明确标注为诊断

#### Scenario: 速览表不得替代完整方法公式

- **WHEN** 文档使用流程表、模块图或总览公式概括默认主路线
- **THEN** 文档 MUST 明确说明该内容只是速览或索引
- **AND** 文档 MUST 另设完整计算链或对应模块小节，逐步写出从 $X$ 到 $\hat{Y}$ 的计算过程
- **AND** 完整计算链 MUST 至少写出以下公式：输入时间序列形状、ALFF/fALFF 或低频节点特征、模块 2 历史窗口预测、$A_{\mathrm{lag}}$ 和 $A_0$ 的来源、分类图 $A_{\mathrm{cls}}$、HGCN 图传播、`mean_std` readout、HPEC energy、最终 logits 和总 loss
- **AND** 每个公式后 MUST 写明参考来源或原理，例如 ALFF/fALFF、Granger causality、NOTEARS/NTS-NOTEARS、HGCN、Poincare Ball、HPEC、prototype learning 或 dual-view evidence fusion
- **AND** 每个公式后 MUST 写明为什么该步骤适合当前 fMRI / S-DeCI 设计

#### Scenario: 模块 1 描述子不得缺少定义

- **WHEN** 文档描述模块 1 的 ALFF/fALFF 节点特征
- **THEN** 文档 MUST 写出 $\mathrm{ALFF}$、$\mathrm{fALFF}$、$\mathrm{BandStd}$、$\mathrm{DomFreq}$、$\mathrm{TimeStd}$、$C^{\mathrm{desc}}$、$C^{\mathrm{band}}$ 和最终 $C$ 的 LaTeX 公式
- **AND** 文档 MUST 写明 $X_{\mathrm{temp}}$ 是否来自低频带通时间序列或等价时间序列
- **AND** 文档 MUST 说明模块 1 的设计来源是静息态 fMRI 低频生理信号，而不是普通黑盒特征投影

#### Scenario: 默认主路线每个 loss 必须有未加权与加权形式

- **WHEN** 文档描述默认训练目标
- **THEN** 文档 MUST 分别写出 $\mathcal{L}_{\mathrm{cls}}$、$\mathcal{L}_{\mathrm{pred}}$、$\mathcal{L}_{\mathrm{sparse}}$、$\mathcal{L}_{\mathrm{smooth}}$、$h(A_0)$ 和可选 $\mathcal{L}_{\mathrm{manifold}}$ 的未加权公式或组成
- **AND** 文档 MUST 写出这些项进入 $\mathcal{L}_{\mathrm{total}}$ 的加权方式
- **AND** 文档 MUST 区分未加权项和加权项；若写出 $\mathcal{L}^{0}_{\mathrm{distill}}$，MUST 明确它是额外消融项，默认不进入总损失
- **AND** 文档 MUST NOT 把同一个权重在子损失定义和总损失展开中重复乘两次
- **AND** 若某项默认权重为 `0`，文档 MUST 明确标注它不构成默认训练贡献
- **AND** 文档 MUST 说明每个 loss 的职责，避免多个 loss 重复优化同一目标

#### Scenario: 实验分支不得和默认路线平级

- **WHEN** 文档描述 `attn_nts_notears`、`static_feature DAG`、`dag_sampling`、`LP-Brain-HPEC`、复杂 readout 或额外监督 loss
- **THEN** MUST 明确标记为 experimental、legacy、debug 或消融对照
- **AND** MUST NOT 将这些分支写成默认主路线必须执行的步骤
- **AND** 默认主路线 MUST 保持为模块 1 低频生理特征、模块 2 Temporal NTS-NOTEARS、模块 3 Poincare HGCN `mean_std` readout、模块 4 HPEC 多原型能量证据融合

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

`S-DeCI` SHALL 在模块 1 产生低频生理节点特征和处理后的时间序列后接入模块 2 因果图学习。正式默认模块 2 输入 MUST 优先使用时间序列语义，而不是只使用静态 `[B, N, d_model]` 节点特征重构。

#### Scenario: 使用时间序列作为模块 2 主输入

- **WHEN** `S-DeCI` 执行 forward 并完成 DeCI block 分解
- **THEN** 模型 MUST 从模块 1 获得形状为 `[B, T, N]` 或可等价还原为该语义的低频时间序列
- **AND** 模块 2 MUST 用历史窗口预测未来时间点：

$$
\hat{x}_{t,j}
=
\sum_{\ell=1}^{L}
\sum_{i=1}^{N}
A_{\mathrm{lag}}^{(\ell)}[i,j]\,x_{t-\ell,i}
+
r_{t,j}.
$$

- **AND** $A_{\mathrm{lag}}^{(\ell)}[i,j]$ MUST 按 parent ROI $i$ 指向 child ROI $j$ 解释
- **AND** 该设计 MUST 以 Granger causality 和 Temporal NTS-NOTEARS 为原理来源，用时间先后关系减少静态 DAG 的方向不确定性

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

### Requirement: S-DeCI 默认模块 3/4 路径

`S-DeCI` 的默认主路线 SHALL 使用现有 Poincare HGCN-HPEC 路径。`lp_brain_hpec` 已在完整五折负向实验后退出当前可执行能力；若未来通过新 change 重新引入，MUST 按“输入、计算公式、输出、来源或原理、设计原因”完整展开，且不得和默认主路线混写。

#### Scenario: 默认使用 HGCN-HPEC 路径

- **GIVEN** `use_hyperbolic_modules34 == 1`
- **WHEN** `S-DeCI` 初始化
- **THEN** 默认配置 MUST 直接初始化 HGCN-HPEC 路径，不要求 `module34_arch` 选择器
- **AND** 模型 MUST 使用 Poincare HGCN readout 与 HPEC 多原型 energy evidence 作为默认模块 3/4 路径

#### Scenario: LP-Brain-HPEC 只保留历史证据

- **WHEN** 文档提到 `lp_brain_hpec`
- **THEN** MUST 明确标注它是已退出代码的负向实验
- **AND** MUST 说明其 MDD/AAL116 五折 Accuracy 为 62.63%、Macro-F1 为 59.43%、AUC 为 62.50%，且训练更慢
- **AND** 正式训练入口 MUST NOT 把它暴露为可选架构

### Requirement: S-DeCI 暴露模块 2 时序因果辅助损失

`S-DeCI` SHALL 在 forward 后暴露模块 2 的时序因果辅助损失和诊断量，使训练流程能够将 Temporal NTS-NOTEARS loss 纳入总 loss，同时不改变 `forward()` 的主返回值。设计依据是 Granger causality 的“过去预测未来”原则，以及 NOTEARS 的可微结构学习约束；因此默认路径不再把静态节点特征重构作为模块 2 的核心目标。

#### Scenario: 暴露时序因果学习 loss

- **WHEN** `S-DeCI` 开启模块 2 并完成一次 forward
- **THEN** 模型 MUST 能提供时序预测损失 $\mathcal{L}_{\mathrm{pred}}$、稀疏损失 $\mathcal{L}_{\mathrm{sparse}}$、lag 平滑损失 $\mathcal{L}_{\mathrm{smooth}}$ 和同时间片残余图约束 $h(A_0)$
- **AND** 模块 2 的预测公式 MUST 显式对应：

$$
\hat{x}_{t,j}
=
\sum_{\ell=1}^{L}
\sum_{i=1}^{N}
A_{\mathrm{lag}}^{(\ell)}[i,j]\,x_{t-\ell,i}
+
\rho_0
\sum_{i=1}^{N} A_0[i,j]\,\hat{x}^{\mathrm{lag}}_{t,i}.
$$

- **AND** 其中 $\hat{x}^{\mathrm{lag}}_{t,i}$ MUST 来自跨时间 lag 分支预测，不得使用真实当前目标 $x_{t,i}$ 作为 $A_0$ 输入
- **AND** 这些 loss MUST 能参与 PyTorch autograd 反向传播
- **AND** 旧静态 reconstruction loss MAY 仅作为 legacy/static-feature 对照路径保留，不得写入默认 S-DeCI 主路线说明

#### Scenario: forward 返回值不变

- **WHEN** 训练流程调用 `y_hat = model(x_enc)`
- **THEN** `S-DeCI.forward()` MUST 只返回分类输出 `y_hat`
- **AND** 模块 2 的辅助 loss MUST 通过模型属性或方法读取

#### Scenario: 暴露模块 2 诊断量

- **WHEN** `S-DeCI` 开启模块 2 并完成一次 forward
- **THEN** 模型 SHOULD 暴露 $A_{\mathrm{lag}}$、$A_{\mathrm{lag\_mean}}$、$A_0$、$A_{\mathrm{cls}}$、`a_lag_mass`、`a0_mass`、`directionality` 和 `graph_delta`
- **AND** 这些诊断量 SHOULD 用于 TensorBoard、控制台日志和最终 epoch 可视化
- **AND** 设计原因 MUST 写明：$A_{\mathrm{lag}}$ 是跨时间方向图，$A_0$ 只用于描述同一时间片残余依赖，$A_{\mathrm{cls}}$ 是传给模块 3/GCN 的分类图

### Requirement: S-DeCI 模块 2/3 联合损失遵守设计文档

`S-DeCI` SHALL 在启用模块 2 和模块 3 时使用由分类损失、模块 2 时序预测损失、稀疏损失、lag 平滑损失和 $A_0$ 弱 DAG 约束组成的联合训练目标。旧静态 reconstruction loss 只属于 legacy/static-feature 路径，不得作为默认 temporal 路径描述。

#### Scenario: 计算联合训练 loss

- **GIVEN** `S-DeCI` 已启用模块 2 和模块 3
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 等价于：

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{cls}}
+
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0).
$$

- **AND** 模块 2 预测项 MUST 来自历史时间窗预测未来时间点，而不是静态节点特征自编码
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

### Requirement: S-DeCI 使用 HPEC energy 融合分类 loss

`S-DeCI` SHALL 在启用模块 4 时使用 HPEC energy/prototype 证据参与最终分类，并保留模块 2 时序因果辅助损失组成联合训练目标。

#### Scenario: 计算模块 4 联合 loss

- **GIVEN** `S-DeCI` 已启用模块 2、模块 3 和模块 4
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 等价于：

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{cls}}(\hat{Y},y)
+
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{dag}}h(A_0)
+
\mathcal{L}_{\mathrm{manifold}}.
$$

- **AND** $\hat{Y}$ MUST 包含 HPEC energy/prototype 形成的双曲证据
- **AND** $\mathcal{L}_{\mathrm{cls}}(\hat{Y},y)$ MUST 使用最终融合 logits 的交叉熵：

$$
\mathcal{L}_{\mathrm{cls}}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{\exp(\hat{Y}_{b,y_b})}
{\sum_{k=1}^{K}\exp(\hat{Y}_{b,k})}.
$$

- **AND** 模块 2 时序预测项 MUST 来自历史窗口预测未来时间点：

$$
\hat{x}_{b,t,j}
=
f_{\theta,j}
\left(
x_{b,t-1,:},\ldots,x_{b,t-L,:};
A_{\mathrm{lag}},A_0
\right),
\qquad
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{|\Omega|}
\sum_{(b,t,j)\in\Omega}
\ell_{\mathrm{pred}}
\left(
\hat{x}_{b,t,j},x_{b,t,j}
\right).
$$

- **AND** 可选流形结构正则 MUST 只约束双曲表示和 prototype 的有效几何区域，例如：

$$
\mathcal{L}_{\mathrm{manifold}}
=
\lambda_z\mathcal{L}_{z\_\mathrm{radius}}
+
\lambda_p\mathcal{L}_{\mathrm{proto\_sep}}
+
\lambda_E\mathcal{L}_{\mathrm{energy}}.
$$

- **AND** 总 loss MUST NOT 使用真实因果矩阵监督
- **AND** 分类 loss MUST 能通过模块 4、模块 3 和 `A_learned` 回传到模块 2 因果图学习参数
- **AND** 设计原因 MUST 写明：HPEC energy 不再“替换掉”全部分类路径，而是作为双曲原型证据进入最终 logits；这样训练目标、指标口径和模型解释路径保持一致。

#### Scenario: 模型提供 primary loss

- **GIVEN** 模块 4 已完成一次 forward
- **WHEN** 训练流程需要分类 loss
- **THEN** `S-DeCI` MUST 能提供 HPEC primary/classification loss
- **AND** 训练流程 MUST 在该 loss 可用时优先使用它，而不是外部普通 criterion

### Requirement: S-DeCI 使用 HPEC 证据融合预测与指标

`S-DeCI` SHALL 在启用默认 HGCN-HPEC 模块 4 时，将 HPEC energy/prototype evidence 转换为校准后的双曲证据增量，并与欧氏局部结构证据 logits 融合后计算预测类别和指标。纯 energy-based prediction 只作为历史实验口径保留。

#### Scenario: 默认融合预测

- **GIVEN** `energy_matrix` 的形状为 `[B, classes]`
- **AND** 欧氏局部结构 logits $\ell^{\mathrm{base}}\in\mathbb{R}^{B\times K}$ 已可用
- **WHEN** 模块 4 产生 HPEC 证据
- **THEN** HPEC logits MUST 先由能量得到：

$$
\ell^{\mathrm{hyper}}_{b,k}
=
-E_{b,k}
+\lambda_{\mathrm{evi}}\bar{s}_{b,k}.
$$

- **AND** 最终 logits MUST 由欧氏局部结构 logits 与校准后的 HPEC 双曲证据增量融合得到：

$$
\hat{Y}_b
=
\ell^{\mathrm{base}}_b
+\lambda_{\mathrm{hyp}}g_b r^{\mathrm{hyper}}_b.
$$

- **AND** 预测类别和概率类指标 MUST 使用 $\hat{Y}$：

$$
\hat{y}_b=\operatorname*{argmax}_{k}\hat{Y}_{b,k},
\qquad
p_{b,k}
=
\frac{\exp(\hat{Y}_{b,k})}{\sum_{r=1}^{K}\exp(\hat{Y}_{b,r})}.
$$

- **AND** 文档 MUST 说明该融合不是主从式关系，而是欧氏局部结构视角与双曲层级原型视角的 dual-view evidence fusion；设计原因是保留模块 3/4 的双曲原型叙事，同时避免纯 energy-only 在小样本训练中造成校准偏移

#### Scenario: forward 返回兼容训练流程

- **WHEN** `S-DeCI.forward()` 在模块 4 启用时返回主输出
- **THEN** 返回值 MUST 能被现有训练、验证和测试流程收集
- **AND** 默认主路线的指标计算 MUST 使用最终融合 logits 或其概率
- **AND** 只有显式 energy-only 实验路径 MAY 优先使用模型缓存的 HPEC energy prediction/probability

### Requirement: S-DeCI 在模块 2 关闭时使用样本相关矩阵

`S-DeCI` SHALL 在 `use_causal_module2=0` 且 `use_hgcn_module3=1` 时，使用输入 batch 对应的样本相关系数矩阵作为模块 3 adjacency。

#### Scenario: 模块 2 关闭且模块 3 开启

- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hgcn_module3 == 1`
- **WHEN** `S-DeCI.forward()` 接收到 `correlation_matrix`
- **THEN** 模型 MUST 将 `correlation_matrix` 传入模块 3
- **AND** 模型 MUST NOT 初始化或调用模块 2 因果学习器
- **AND** 模型 MUST NOT 计算模块 2 时序预测损失、稀疏损失、lag 平滑损失或 $A_0$ DAG auxiliary loss

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

### Requirement: attention-guided 模块 2 不属于默认主路线

`S-DeCI` MAY 保留 `causal_graph_method == "attn_nts_notears"` 作为实验对照，但默认主路线 MUST 使用 `causal_graph_method == "nts_notears"`。若文档只在裁剪清单中提到 attention-guided 方法，可只说明负收益或未稳定收益；若把它写成后续可执行方案，则 MUST 写出 attention 权重、图抽取、预测、损失和输出维度的完整 LaTeX 公式。

#### Scenario: 默认不启用 attention-guided learner

- **GIVEN** 默认训练入口构建 `S-DeCI`
- **WHEN** `S-DeCI` 初始化
- **THEN** 默认 `causal_graph_method` MUST 为 `nts_notears`
- **AND** 默认文档 MUST NOT 将 attention-guided 模块 2 写成必须执行的计算步骤

#### Scenario: attention-guided 方案需要完整公式

- **WHEN** 后续变更重新提出 attention-guided temporal causal learner
- **THEN** proposal、design、tasks 或主设计文档 MUST 写明输入 $X_{\mathrm{temp}}\in\mathbb{R}^{B\times T\times N}$、attention score、attention probability、$A_{\mathrm{lag}}$ 抽取、预测 $\hat{x}_{b,t,j}$、损失项和输出图形状
- **AND** 每个公式后 MUST 写明来源或原理，以及为什么它比默认 Temporal NTS-NOTEARS 更适合当前实验

### Requirement: S-DeCI 支持模块 4 多 prototype 参数

`S-DeCI` SHALL 在启用模块 4 HPEC 时支持每类多 prototype 配置，并将相关超参数传递给 HPEC 层。

#### Scenario: 传入多 prototype 配置

- **GIVEN** 用户通过训练入口设置 `hpec_prototypes_per_class`
- **WHEN** `S-DeCI` 初始化模块 4
- **THEN** 模型 MUST 将 `hpec_prototypes_per_class` 传递给 HPEC 模块 4
- **AND** 模型 MUST 支持配置 `hpec_proto_temperature`
- **AND** 模型 MUST 支持配置 `hpec_energy_loss_weight`、`hpec_teacher_distill_weight`、`hpec_z_radius_loss_weight` 和 `hpec_prototype_separation_loss_weight`
- **AND** 模型 MUST 支持配置 `hpec_trainable_prototypes`、`hpec_use_sinkhorn_ema` 或等价 prototype 更新控制参数

#### Scenario: 模块 4 关闭时不创建多 prototype

- **GIVEN** `use_hpec_module4 == 0`
- **WHEN** `S-DeCI` 初始化
- **THEN** 模型 MUST NOT 初始化多 prototype HPEC 层
- **AND** 新增 prototype loss MUST 不参与训练

### Requirement: S-DeCI 暴露 HPEC prototype loss 与诊断

`S-DeCI` SHALL 在 forward 后基于 label 计算并暴露 HPEC energy、半径约束、prototype separation 和可选 distillation 校准相关 loss；只有对应权重大于 0 时，训练流程才应将这些辅助项加入总 loss。

#### Scenario: 计算 HPEC prototype loss

- **GIVEN** `S-DeCI` 已启用模块 3 和模块 4
- **AND** 模块 4 已完成一次 forward
- **WHEN** 训练流程调用模型的 label-aware loss 计算方法
- **THEN** 模型 MUST 能计算 HPEC final CE
- **AND** 当对应权重大于 0 时 MUST 能计算 energy loss、可选 distillation 校准、半径约束和 prototype separation
- **AND** 这些 HPEC auxiliary loss MUST 能与 HPEC primary loss 和模块 2 auxiliary loss 一起反向传播

#### Scenario: 总 loss 保持联合训练结构

- **GIVEN** 模块 2、模块 3 和模块 4 均已启用
- **WHEN** 训练流程计算总 loss
- **THEN** 默认总 loss MUST 包含：

$$
\mathcal{L}_{\mathrm{cls}}
+
\mathcal{L}_{\mathrm{module2}}
+
\mathcal{L}_{\mathrm{manifold}}.
$$

- **AND** HPEC energy MUST 作为模块 4 forward 中的类别证据进入 logits；除非文档和入口配置明确启用额外 energy loss，否则它 MUST NOT 被写成默认额外监督项
- **AND** 当对应权重大于 0 时 MAY 额外包含可选 distillation 校准、半径约束和 prototype separation；这些项默认不构成主路线贡献
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
- **AND** 总 loss MUST 包含模块 2 时序预测、$A_0$ DAG、$A_{\mathrm{lag}}/A_0$ 稀疏和 lag 平滑 auxiliary loss
- **AND** 若 HPEC prototype 稳定项或 distillation 权重大于 0，总 loss MUST 包含对应 HPEC auxiliary loss
- **AND** 模块 2 的时序预测项 MUST 对应：

$$
\hat{x}_{t,j}
=
\sum_{\ell=1}^{L}
\sum_{i=1}^{N}
A_{\mathrm{lag}}^{(\ell)}[i,j]x_{t-\ell,i}
+
\sum_{i=1}^{N}A_0[i,j]\hat{x}^{\mathrm{lag}}_{t,i}.
$$

- **AND** 设计原因 MUST 写明：该项来源于 Granger causality / Temporal NTS-NOTEARS 的历史预测未来原则，不是静态节点特征 reconstruction

#### Scenario: GCN fallback 且模块 2 启用时使用分类 loss 加因果辅助项
- **GIVEN** `use_causal_module2 == 1`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 GCN fallback 分类 loss
- **AND** 总 loss MUST 包含模块 2 时序预测、$A_0$ DAG、$A_{\mathrm{lag}}/A_0$ 稀疏和 lag 平滑 auxiliary loss
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss

### Requirement: S-DeCI 协调标准视图与独立原型更新

`S-DeCI` SHALL 在保持 `forward()` 返回标准最终 logits 的前提下，向训练循环提供优化器更新后的可靠 prototype 更新接口。已完成负向实验的互补遮挡分支不得进入当前正式 forward 或总 loss。

#### Scenario: 新增机制只在兼容模块组合中运行
- **GIVEN** 模块 3/4 未同时启用
- **WHEN** 用户启用可靠 TP prototype 或多阶因果编码
- **THEN** 系统 MUST 拒绝该不兼容组合或明确跳过
- **AND** MUST 不影响 GCN fallback 路径

#### Scenario: 互补学习不进入当前训练
- **WHEN** 系统计算标准 forward 与总 loss
- **THEN** MUST NOT 执行遮挡互补模块 3/4 前向
- **AND** MUST NOT 加入 Poincare 双视图一致性、InfoNCE 或 masked CE
- **AND** 历史负向结果 MUST 保留在模型修改证据台账

#### Scenario: GCN fallback 且模块 2 禁用时只使用分类 loss
- **GIVEN** `use_causal_module2 == 0`
- **AND** `use_hyperbolic_modules34 == 0`
- **WHEN** 训练流程计算总 loss
- **THEN** 总 loss MUST 包含 GCN fallback 分类 loss
- **AND** 总 loss MUST NOT 包含模块 2 auxiliary loss
- **AND** 总 loss MUST NOT 包含 HPEC 或 prototype loss

