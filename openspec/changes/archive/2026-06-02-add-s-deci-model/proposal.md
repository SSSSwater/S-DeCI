## Why

`docs/新模块设计.md` 将“模块 1：DeCI 前端时序信号解耦”定义为后续因果图、双曲 GCN 和 HPEC 分类之前的前端特征提取阶段。为了先验证该前端模块能在现有训练框架中跑通，需要新增一个独立模型文件，而不是直接修改现有 `models/DeCI.py`。

本次变更先只实现模块 1，并暂时使用 Cycle/seasonal 特征完成分类，从而在不影响 DeCI 原始实现的前提下验证新模块结构、模型注册和训练链路。

## What Changes

- 在 `models/` 下新增一个独立模型文件，模型名称可临时命名，不能直接修改 `models/DeCI.py` 的主逻辑。
- 新模型可从现有 DeCI 模型复制基础结构，并按 `docs/新模块设计.md` 的模块 1 要求做适当修改。
- 复用现有 `layers/DeCI_Layer.py` 中的 `DeCI_Block`，不新建 DeCI block。
- 保留 DeCI 的 `Channel-Independence` 时序嵌入和 Cycle/Drift decomposition 前端思路。
- 将模块 1 的前端输出聚焦到 Cycle/seasonal 特征；当前阶段丢弃 Drift/trend 特征的分类贡献。
- 最终分类先仅使用 Cycle/seasonal 特征，确保输出仍兼容现有 `Exp_Main` 的二分类和多分类训练路径。
- 将新模型注册到现有模型字典，使其可通过 `run_cv.py --model <new_model_name>` 训练。
- 使用已有训练测试脚本或等价低预算命令验证新模型训练能跑通。

## Capabilities

### New Capabilities
- `s-deci-model`: 独立的 `S-DeCI` 模型实现，复用 DeCI block，仅以 Cycle/seasonal 特征完成分类，并可接入现有训练流程。

### Modified Capabilities

## Impact

- 影响文件：新增 `models/<new_model_name>.py`，更新 `models/__init__.py` 和 `exp/exp_basic.py` 中的模型导入/注册。
- 影响系统：新增一个可训练模型选项，不改变现有 `DeCI`、数据加载、训练循环或 loss 逻辑。
- 依赖：不新增 `geoopt` 或模块 2-4 的依赖；本次只实现模块 1。
- 回滚方案：删除新增模型文件，并移除模型注册项；现有 DeCI 和其他模型保持不变。
