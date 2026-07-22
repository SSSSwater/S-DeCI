## Why

当前模块 3/4 在 MDD 完整 5-fold 中表现不稳定：HGCN 路径可以接近 GCN fallback，但 HPEC final head 容易让训练集继续变好、测试集下降，说明双曲 prototype 分类没有稳定承接模块 2 学到的有向因果图。`修改方案.md` 提出的 Lorentz 有向图卷积、Lorentz-to-Poincare bridge 和 MAC/HBR-HPEC 思路，能够让模块 3/4 更直接地利用非对称因果拓扑，同时通过几何安全半径降低中心塌缩和边界 NaN 风险。

## What Changes

- 在 S-DeCI 中新增可切换的模块 3/4 路径：`lp_brain_hpec`，保留现有 HGCN/HPEC 路径作为回退和消融对照。
- 模块 3 新增 Lorentz lifting、Directed Lorentz GCN 和 tangent-space readout，使模块 2 的有向因果图以入边/出边两套聚合参与双曲表示学习。
- 模块 4 新增 Lorentz-to-Poincare stereographic bridge、MAC 半径裁剪和 HBR 半径惩罚，并继续使用 HPEC energy/prototype 分类。
- 不直接照搬“全程 float64”作为默认训练策略；改为提供可配置的几何计算精度，默认只在关键流形操作中使用更稳定的 dtype，避免显存和速度不可控。
- 新建项目实现说明文档，记录 LP-Brain-HPEC 数据流、张量形状、损失项、默认参数和回滚方式；不修改 `docs/新模块设计.md` 等原始参考文档。
- 更新训练脚本和默认测试脚本，支持选择 `module34_arch=lp_brain_hpec`，并在 `result.xlsx` 与 TensorBoard 中记录该路径的诊断量。

## Capabilities

### New Capabilities

无。该方案是对现有 S-DeCI 模块 3/4 能力的结构升级，不新增独立业务能力。

### Modified Capabilities

- `module3-hgcn-readout`: 增加 LP-Brain-HPEC 的 Lorentz directed graph readout 路径，用入边/出边聚合替代仅 Poincare HGCN 的单一路径。
- `module4-hpec-classification`: 增加 Lorentz-to-Poincare bridge、MAC 半径裁剪和 HBR 半径惩罚，并保持 HPEC energy/prototype final classifier。
- `s-deci-model`: 增加 `module34_arch` 级别的路径选择、缓存和可视化逻辑，使新模块 3/4 与模块 1/2 主流程衔接。
- `training-test-scripts`: 增加 CLI、best config 脚本、TensorBoard/result 记录和完整 5-fold 对比要求。

## Impact

- 影响代码：
  - `models/S_DeCI.py`
  - `layers/` 下新增 Lorentz directed GCN / bridge / MAC-HBR-HPEC 相关模块
  - `run_cv.py`
  - `test_mdd_best_config.py`，必要时同步 `test_abide_best_config.py`
  - `exp/exp_classification_CV.py`
- 影响依赖：
  - 复用现有 `torch`、`geoopt`，不新增大型依赖。
- 影响训练：
  - 需要至少完成 MDD 5-fold 50epoch 对比：现有 HGCN/HPEC、GCN fallback、`lp_brain_hpec`。
  - 完整结果继续写入 `result.xlsx`，TensorBoard 记录 loss、指标、半径、入出边聚合强度、MAC/HBR 诊断。
- 回滚方案：
  - 将 `module34_arch` 设置回现有 `hgcn_hpec`。
  - 或关闭模块 3/4，继续使用 GCN fallback 消融路径。
  - 新增代码应保持参数默认兼容，不破坏现有训练脚本。
