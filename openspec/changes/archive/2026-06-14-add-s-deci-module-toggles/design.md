## Context

`S-DeCI` 当前已经包含模块 1 的 DeCI/Cycle 特征、模块 2 的 causal graph learning、模块 3 的 HGCN readout，以及模块 4 的 HPEC energy/prototype 分类。已有代码还支持在模块 2 关闭时读取样本级相关矩阵作为模块 3 的 adjacency。

这次变更的核心不是替换已有模型，而是让每个模块有清晰、可复现的启用/禁用路径，便于消融实验：

- 模块 1 关闭时，仍要产生 `[B, N, d_model]` 节点特征，但来源改为原始时间序列投影。
- 模块 2 关闭时，继续使用数据集中已有的样本相关矩阵。
- 模块 3 与模块 4 作为一组关闭时，不再使用 HGCN/HPEC，而是退化为 Euclidean GCN 分类路径。

## Goals / Non-Goals

**Goals:**

- 提供稳定的模块开关参数，并在 `run_cv.py` 和测试脚本中可配置。
- 保证任意支持的组合都有明确的 feature、adjacency、loss 和可视化来源。
- 模块 3/4 禁用时提供普通 GCN fallback，使模型仍可利用模块 1 输出特征和模块 2/相关矩阵图结构完成分类。
- 默认行为保持向后兼容，默认仍可走当前已经验证过的 `S-DeCI` 全模块路径。

**Non-Goals:**

- 不修改 `docs/` 下的原始设计参考文档。
- 不重新设计模块 2 的 causal graph learning 算法。
- 不将模块 3 与模块 4拆成新的独立实验路径；本变更要求二者在训练入口上作为一组联合启用/禁用。
- 不改变数据集文件格式，只复用已存在的时间序列和相关矩阵读取逻辑。

## Decisions

1. 模块开关使用显式布尔/整型 CLI 参数。

   推荐新增或统一使用：

   - `use_deci_module1`: 控制模块 1 的 DeCI/Cycle 分解是否启用。
   - `use_causal_module2`: 沿用已有模块 2 开关。
   - `use_hyperbolic_modules34`: 作为模块 3/4 的联合总开关。

   如果实现中为了兼容已有参数仍保留 `use_hgcn_module3` 与 `use_hpec_module4`，训练入口必须把它们与 `use_hyperbolic_modules34` 归一到同一语义，避免出现 HGCN 开启但 HPEC 关闭、或 HPEC 开启但 HGCN 关闭的歧义状态。

2. 模块 1 禁用时使用 raw projection。

   输入 `x_enc` 形状保持 `[B, T, N]`。禁用模块 1 时，模型先转为 `[B, N, T]`，再用 `nn.Linear(seq_len, d_model)` 或等价投影得到 `[B, N, d_model]`。该路径不调用 DeCI block，不计算高频、trend、seasonal 或 residual，但对后续模块暴露同名的节点特征缓存，便于训练与可视化代码复用。

3. 模块 2 禁用时继续使用 sample correlation adjacency。

   该行为已经存在，本变更只把它纳入统一开关矩阵。模块 2 禁用时不得初始化或调用 causal graph learner，也不得把 reconstruction、DAG、L1 loss 加入总损失。若后续图路径需要 adjacency，则必须从 batch 的 `correlation_matrix` 获取。

4. 模块 3/4 禁用时使用 GCN fallback。

   GCN fallback 使用 Euclidean graph convolution，不进入 Poincare Ball，不计算 HPEC energy/prototype loss。其输入为模块 1 输出的 `[B, N, d_model]` 节点特征；adjacency 来源为模块 2 的 `A_learned`，或模块 2 禁用时的 sample correlation matrix。GCN 输出经过 readout 和分类头得到训练 logits。

5. loss 根据启用路径选择。

   - HGCN/HPEC 路径启用：沿用现有 HPEC primary loss、多 prototype loss、模块 2 auxiliary loss 的联合结构。
   - GCN fallback 路径启用：使用普通分类 loss；若模块 2 启用，则继续叠加模块 2 auxiliary loss；若模块 2 禁用，则只使用分类 loss。
   - 模块 1 关闭不改变 loss 类型，只改变节点特征来源。

6. 可视化根据当前路径保存可解释的中间量。

   模块 1 开启时可视化 Cycle/seasonal feature；模块 1 关闭时可视化 raw projected feature。HGCN/HPEC 路径可继续保存 `z_global`、prototype、energy 等；GCN fallback 路径保存 GCN hidden/readout/logits，以及实际使用的 adjacency。

## Risks / Trade-offs

- [Risk] 开关组合过多，容易造成参数不一致。 → 在参数解析后做归一化与校验，特别是模块 3/4 必须联合启用/禁用。
- [Risk] 模块 1 关闭后 raw projection 特征分布不同，可能需要不同学习率或正则。 → 默认先保持同一训练流程，必要时通过 CLI 调整超参数。
- [Risk] GCN fallback 与 HGCN/HPEC 的输出接口不同，训练流程可能分支过多。 → 模型内部缓存统一命名，训练流程优先读取模型提供的 primary loss/prediction，否则回退到普通 criterion。
- [Risk] 模块 2 禁用但未提供相关矩阵时 GCN fallback 无 adjacency。 → forward 必须清晰报错，提示需要 sample correlation matrix 或启用模块 2。

## Migration Plan

1. 在模型配置中加入模块开关，默认保持当前全模块启用路径。
2. 在 `S_DeCI.py` 内实现 raw projection 分支、开关归一化和 GCN fallback 分支。
3. 在训练入口和测试脚本中暴露中文 help，并把参数传入 experiment/model。
4. 增加或更新低预算训练验证，覆盖全模块、模块 1 关闭、模块 2 关闭、模块 3/4 关闭的关键组合。
5. 如需回滚，将默认值恢复为全模块启用，并移除 GCN fallback 调用路径；已有 HGCN/HPEC 与模块 2 逻辑不应受影响。

## Open Questions

- GCN fallback 是否复用已有图卷积层还是新增轻量层，取决于当前 `layers/` 中是否已有合适实现。
- 模块 3/4 关闭时的默认 readout 使用 mean pooling 还是 attention pooling，可在实现时优先选择更简单且稳定的 mean pooling。
