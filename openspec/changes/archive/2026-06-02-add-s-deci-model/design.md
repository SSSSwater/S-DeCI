## Context

`docs/新模块设计.md` 规划了一个多阶段系统，其中模块 1 负责基于 DeCI 前端进行时序信号解耦，目标是从原始 BOLD fMRI 信号中得到去除 Drift 噪声后的 Cycle 高频神经表征。当前仓库已有 `models/DeCI.py` 和 `layers/DeCI_Layer.py`，其中 `DeCI_Block` 已实现 trend/drift 与 seasonal/cycle 的逐层分解和分类头。

本次变更只实现模块 1 的可训练落地版本：新增一个模型文件，复用现有 `DeCI_Block`，不修改原始 DeCI 模型；为了先跑通训练，分类阶段仅使用 Cycle/seasonal logits，不使用 Drift/trend logits，也不实现模块 2-4。

## Goals / Non-Goals

**Goals:**
- 在 `models/` 下新增独立模型文件，名称为 `S_DeCI.py`，对外模型名为 `S-DeCI`。
- 复用现有 `layers.DeCI_Layer.DeCI_Block`，不复制或新建 block。
- 从 `models/DeCI.py` 复制必要的前端结构，包括可选输入归一化、`Variate_Embedding` 和 `deci_blocks`。
- forward 中保留 Cycle/Drift decomposition 流程，但分类输出只聚合 seasonal/cycle logits。
- 保持输出形状兼容现有 `Exp_Main`：二分类输出 `[B, 1]` 并 sigmoid，多分类输出 `[B, classes]` logits。
- 将新模型注册到 `models/__init__.py` 和 `Exp_Basic.model_dict`，支持 `run_cv.py --model S-DeCI`。
- 使用现有 `.venv` 和训练测试脚本或等价命令确认训练跑通。

**Non-Goals:**
- 不直接修改 `models/DeCI.py` 的主模型逻辑。
- 不新增或修改 `DeCI_Block`。
- 不实现因果邻接矩阵、DAG loss、Hyperbolic GCN、Fréchet mean、HPEC 原型分类或联合损失。
- 不引入 `geoopt` 等模块 2-4 才需要的依赖。
- 不追求论文级性能，只验证模块 1 结构和训练链路。

## Decisions

1. 新模型命名为 `S-DeCI`。

   原因：名称明确表达“Seasonal/Cycle 分支 + DeCI 前端”的阶段性目标，且不会与现有 DeCI 或其他 baseline 混淆。Python 文件名使用合法标识符形式 `S_DeCI.py`。

   备选方案：随机命名或命名为 DeCIv2。随机命名不利于后续维护，DeCIv2 容易暗示替代原 DeCI。

2. 新模型复制 `models/DeCI.py` 的顶层结构，但复用 `DeCI_Block`。

   原因：这样能最小化实现风险，并保持与现有 DeCI 输入、归一化、embedding 和 block 堆叠行为一致；同时满足“不直接在 DeCI 上修改”的约束。

   备选方案：从零重写模块 1。这样更干净，但容易引入与现有 DeCI 不一致的训练行为。

3. forward 继续调用 `trend, seasonal, res = deci_block(res)`，但只收集和聚合 `seasonal`。

   原因：现有 `DeCI_Block` 已同时计算 trend 和 seasonal；为了保留 decomposition 机制且丢弃 Drift/trend 分类贡献，只需忽略 trend logits。

   备选方案：修改 block 让其只返回 seasonal。该方案会影响现有 DeCI block 复用范围，并违反“不新建 DeCI block”的轻量约束。

4. 暂时将 `sum(seasonals)` 作为分类输出。

   原因：这是当前最小可训练版本，直接验证 Cycle/seasonal 分支能否接入现有 loss 和指标路径。

   备选方案：输出 `(B, N, 64)` 的 Cycle 特征并另建分类器。该方案更贴近后续模块接口，但会额外引入分类头设计；本次先以训练跑通为优先。

5. 训练验证使用已有低预算脚本覆盖。

   原因：项目已有 `test_training_smoke.py` 和 `test_matai_small_sample.py`，均支持通过 CLI 覆盖 `--model`、`--d-model`、`--layer` 等参数，可直接用于新模型训练验证。

## Risks / Trade-offs

- [Risk] 只用 seasonal logits 可能短期性能低于 DeCI 全量 trend+seasonal 融合 -> 本次目标是模块 1 训练链路跑通，不以最终性能为判断标准。
- [Risk] 继续计算 trend 但不参与分类会有额外计算开销 -> 这是复用 `DeCI_Block` 的代价，后续如需优化可在新 block 或新前端中处理。
- [Risk] 文档中模块 1 目标提到输出 `(B, N, 64)` 特征，而本次训练模型输出分类 logits -> 设计中保留 seasonal/cycle 中间表征路径，但当前阶段先用 seasonal logits 分类；后续模块 2 可再暴露特征输出接口。
- [Risk] 新模型注册遗漏会导致 `run_cv.py --model S-DeCI` 失败 -> tasks 中必须包含模型导入和 `model_dict` 注册验证。

## Migration Plan

1. 新增 `models/S_DeCI.py`，复制并调整 DeCI 顶层模型。
2. 在 `models/__init__.py` 导入新模型。
3. 在 `exp/exp_basic.py` 的 `model_dict` 注册 `S-DeCI`。
4. 使用 `.venv` 编译检查新模型、模型注册文件和实验入口。
5. 使用低预算训练命令验证新模型可以在现有数据集上完成训练。

## Open Questions

- 后续模块 2 是否需要新模型直接返回 Cycle 特征 `C`，还是通过单独 feature extractor API 暴露；本次不解决。
