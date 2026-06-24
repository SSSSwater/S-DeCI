## Why

当前 `S-DeCI` 的 HPEC 模块 4 仍是每个类别一个 prototype。单一 prototype 容易把同一诊断类别内的不同病理连接模式压成一个中心，导致类内多样性表达不足，也会让 t-SNE 中训练/测试样本与 prototype 的关系不够清晰。

参考用户提供的论文笔记《Prototypical Representation Learning for Multi-Site Domain Generalization in Schizophrenia Diagnosis》，本变更希望把 HPEC 分类从“每类单 prototype”扩展为“每类多个 prototype”，并加入与多 prototype 匹配相关的损失项，使特征既能靠近同类 prototype 分布，也能与异类 prototype 拉开。

## What Changes

- 将模块 4 的类别 prototype 从形状 `[classes, hidden_dim]` 改为 `[classes, prototypes_per_class, hidden_dim]`。
- 新增可配置参数，例如 `hpec_prototypes_per_class`、`hpec_proto_temperature`、`lambda_hpec_mle`、`lambda_hpec_pcl`、`lambda_hpec_pal`。
- 参考论文笔记引入多 prototype 相关损失：
  - `L_mle`：基于样本与同类/异类 prototype 相似度的最大似然损失。
  - `L_pcl`：prototype contrastive loss，用于约束同类 prototype 与异类 prototype 的结构关系。
  - `L_pal`：prototype alignment loss，使样本靠近同类中最匹配的 prototype。
- 保留原有 HPEC energy 分类路径，但 energy 需要从多 prototype 聚合为类别级能量或类别级 score。
- 最终 t-SNE 可视化继续显示 train/test 样本，并显示每个类别的多个 prototype。
- 默认保持保守配置，允许通过 `hpec_prototypes_per_class=1` 或关闭新增 loss 权重回退到接近当前行为。

## Capabilities

### New Capabilities
- `multi-prototype-hpec-classification`: 定义 HPEC 模块 4 的每类多 prototype、prototype assignment、prototype losses、类别能量聚合与可视化行为。

### Modified Capabilities
- `s-deci-model`: `S-DeCI` 模型需要支持模块 4 多 prototype 参数、loss 暴露、预测输出和中间量缓存。
- `training-test-scripts`: 训练入口和测试脚本需要支持多 prototype 超参数，并打印新增 prototype loss。
- `tensor-visualization-helper`: S-DeCI 可视化需要能展示多 prototype 矩阵以及 t-SNE 中的多个 prototype 点。

## Impact

- 影响模块范围：
  - `layers/hpec_energy_layer.py`
  - `models/S_DeCI.py`
  - `exp/exp_classification_CV.py`
  - `run_cv.py`
  - `test_training_smoke.py`
  - `test_matai_small_sample.py`
  - 新增实现说明文档，原 `docs/新模块设计.md` 不直接修改。
- 训练行为影响：
  - 启用多 prototype 后，分类 loss 将包含 HPEC energy loss 与 prototype-related loss 的组合。
  - prototype 数量增加会提高少量显存与计算开销，尤其是在 t-SNE 可视化和 loss 计算中。
- API/参数影响：
  - 新增命令行参数，但不移除已有参数。
  - `use_hpec_module4=0` 时不受影响。
- 回滚方案：
  - 设置 `hpec_prototypes_per_class=1`，并将 `lambda_hpec_mle=0`、`lambda_hpec_pcl=0`、`lambda_hpec_pal=0`，即可回退到接近当前单 prototype HPEC 行为。
  - 如需代码级回滚，可恢复 `layers/hpec_energy_layer.py` 和 `models/S_DeCI.py` 中的单 prototype 实现，同时移除新增参数和训练日志字段。
