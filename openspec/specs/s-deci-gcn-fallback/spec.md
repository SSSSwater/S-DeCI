# S-DeCI GCN fallback 规范

## Purpose

定义 `S-DeCI` 在模块 3/4 双曲路径禁用时的 Euclidean GCN fallback 行为。该路径用于消融实验和故障定位：模块 1 仍提供节点特征，模块 2 启用时使用 Temporal NTS-NOTEARS 学到的分类图 $A_{\mathrm{cls}}$ 作为 GCN adjacency；模块 2 禁用时使用数据集中已有的样本相关矩阵作为 adjacency。设计原因是保留“图结构 + 节点特征”的分类能力，同时排除 HGCN/HPEC 对性能的影响。

默认 fallback 保留因果图方向，分别执行入边与出边传播：

$$
H_{\mathrm{in}}^{(l)}=\tilde{A}_{\mathrm{in}}^{\top}H^{(l)},
\qquad
H_{\mathrm{out}}^{(l)}=\tilde{A}_{\mathrm{out}}H^{(l)}.
$$

$$
H^{(l+1)}
=\operatorname{GELU}\!\left(
\operatorname{LN}
\left[
H_{\mathrm{in}}^{(l)}\|H_{\mathrm{out}}^{(l)}\|
\left(H_{\mathrm{out}}^{(l)}-H_{\mathrm{in}}^{(l)}\right)
\right]W^{(l)}
\right).
$$

其中 $A[i,j]$ 表示 $i\rightarrow j$，$\tilde{A}_{\mathrm{in}}$ 按 child 入边归一化，$\tilde{A}_{\mathrm{out}}$ 按 parent 出边归一化。设计仍基于 GCN 邻域消息传递，但不再用对称归一化抹去因果方向；出入消息差显式描述 ROI 的驱动者/接收者角色。

默认图级读出为：

$$
r_b=
\left[
\frac{1}{N}\sum_i H_{b,i}
\;\middle\|\;
\sqrt{\frac{1}{N}\sum_i(H_{b,i}-\mu_b)^2}
\right].
$$

均值表示全脑平均状态，标准差保留脑区响应的离散程度；相较纯 mean，它可减少不同样本图在池化后变成近似常数的问题。

## Requirements
### Requirement: GCN fallback 分类路径

系统 SHALL 在 `S-DeCI` 的模块 3/4 联合禁用时，提供普通 Euclidean GCN fallback 分类路径，用于替代 HGCN + HPEC。

#### Scenario: 使用模块 2 因果矩阵作为 GCN adjacency
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **AND** `use_causal_module2 == 1`
- **WHEN** `S-DeCI.forward()` 完成模块 1 特征提取和模块 2 因果图学习
- **THEN** GCN fallback MUST 使用模块 2 输出的 `A_cls` 或等价的 `A_lag_mean` 融合图作为 adjacency
- **AND** GCN fallback MUST 使用模块 1 输出的 `[B, N, d_model]` 节点特征作为输入
- **AND** 模型 MUST 输出与现有分类训练流程兼容的 logits 或二分类分数
- **AND** 该 adjacency 的默认来源 MUST 对应连续有向门控：

$$
G_{i,j}=\frac{\bar A_{\mathrm{lag}}[i,j]}
{\max_r \bar A_{\mathrm{lag}}[r,j]+\epsilon},
\qquad
A_{\mathrm{cls},b}=|S_b|\odot\left[\alpha+(1-\alpha)G\right].
$$

- **AND** 其中 $\alpha$ 为 `module2_graph_residual_alpha`，默认 `0.10`
- **AND** $G$ MUST 来自稀疏 `A_lag_mean`，不得使用几乎处处非零的稠密 `A_effective > 0` 作为二值 mask
- **AND** $G$ MUST 保持可微，使分类 loss 能回传至模块 2

#### Scenario: 有向传播与 mean-std 读出
- **GIVEN** GCN fallback 使用模块 2 的有向分类图
- **WHEN** 执行图传播和图级读出
- **THEN** 默认路径 MUST 分别编码入边聚合、出边聚合和出入差异
- **AND** 默认 readout MUST 拼接节点 hidden 的 mean 与标准差
- **AND** 系统 MAY 通过参数关闭有向传播或切换旧 readout，用于消融对照

#### Scenario: 共享分类头反事实诊断
- **GIVEN** fallback 同时接收图 readout 与 FC embedding
- **WHEN** 记录 graph-only 和 FC-only 分支指标
- **THEN** 两个分支 MUST 复用已经训练的主分类器
- **AND** graph-only MUST 将 FC embedding 置零
- **AND** FC-only MUST 将图 readout 置零
- **AND** 系统 MUST NOT 使用未训练的随机独立分类头报告分支性能

#### Scenario: 使用样本相关矩阵作为 GCN adjacency
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **AND** `use_causal_module2 == 0`
- **WHEN** `S-DeCI.forward()` 接收到 `correlation_matrix`
- **THEN** GCN fallback MUST 使用 `correlation_matrix` 作为 batch 级 adjacency
- **AND** GCN fallback MUST NOT 初始化或调用模块 2 causal graph learner
- **AND** 总 loss MUST NOT 包含模块 2 temporal prediction、sparse、smooth 或 DAG auxiliary loss

#### Scenario: 缺少 adjacency 时清晰失败
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **AND** `use_causal_module2 == 0`
- **WHEN** `S-DeCI.forward()` 未接收到 `correlation_matrix`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 说明 GCN fallback 需要样本相关矩阵或启用模块 2

### Requirement: GCN fallback 中间量缓存

系统 SHALL 缓存 GCN fallback 的关键中间量，供训练诊断、heatmap 和 t-SNE 可视化使用。

#### Scenario: 缓存 GCN fallback 表征
- **GIVEN** `use_hyperbolic_modules34 == 0`
- **WHEN** GCN fallback 完成 forward
- **THEN** 模型 MUST 缓存实际使用的 adjacency
- **AND** 模型 MUST 缓存 GCN hidden、readout feature 和最终分类输出
- **AND** 这些缓存 MUST 不改变 `S-DeCI.forward()` 的主返回值

#### Scenario: 可视化区分 GCN 与 HGCN 路径
- **GIVEN** 显式启用中间量可视化
- **WHEN** 当前 fold 训练结束
- **THEN** 系统 MUST 能保存 GCN fallback 的 adjacency 和 readout 可视化
- **AND** 文件名或标题 MUST 能区分 `gcn_fallback` 与 `hgcn_hpec`
