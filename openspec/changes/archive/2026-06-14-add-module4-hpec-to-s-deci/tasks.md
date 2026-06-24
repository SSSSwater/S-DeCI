## 1. 参考实现与接口确认

- [x] 1.1 阅读 `docs/新模块设计.md` 中模块 4/HPEC 的损失函数、输入输出和训练目标描述
- [x] 1.2 阅读 `reference/HPEC-main/` 中 prototype 初始化、Poincare Ball、angle/energy 和 entailment loss 的实现
- [x] 1.3 确认当前 `models/S_DeCI.py`、`layers/hyperbolic_gcn_layer.py`、`exp/exp_classification_CV.py` 的模块 3 输出、loss 汇总和指标计算路径

## 2. HPEC 核心层实现

- [x] 2.1 新增 `layers/hpec_energy_layer.py`，实现 HPEC prototype、aperture/psi、angle、energy matrix、energy loss 和 prediction 工具
- [x] 2.2 实现 hyperspherical separation 或等价初始化，并按 `hpec_prototype_radius` 投影到 Poincare Ball
- [x] 2.3 为 HPEC 数学函数添加 eps/clamp 稳定处理，避免 `acos`、`asin`、除法和 norm 在边界处产生 NaN
- [x] 2.4 添加 HPEC 层的中文注释，并确保运行时不从 `reference/` import

## 3. S-DeCI 模块 4 接入

- [x] 3.1 在 `models/S_DeCI.py` 中新增 `use_hpec_module4` 及相关 HPEC 超参数读取逻辑
- [x] 3.2 在模块 4 启用时要求模块 3 可用，并使用 `z_global` 作为 HPEC 默认输入
- [x] 3.3 在 forward 中计算并缓存 prototype、angle matrix、psi/aperture、energy matrix、prediction、probability 和 `Loss_HPEC`
- [x] 3.4 新增或扩展模型方法，使训练流程能读取 HPEC primary/classification loss、energy-based prediction 和 probability
- [x] 3.5 保留 `use_hpec_module4=0` 时模块 3 线性分类头和模块 3 关闭时 Cycle/seasonal logits 分类路径
- [x] 3.6 为新增模块 4 逻辑添加中文注释

## 4. 训练流程与指标

- [x] 4.1 修改 `exp/exp_classification_CV.py`，在模型提供 primary loss 时优先使用模型 loss，否则回退到外部 criterion
- [x] 4.2 保持总 loss 为 `Loss_HPEC + alpha * Loss_Recon + lambda * Loss_DAG + gamma * L1`，并只执行一次 `backward()`
- [x] 4.3 修改训练、验证和测试指标收集逻辑，使模块 4 启用时使用 `argmin(energy)` prediction 和 `softmax(-energy)` probability
- [x] 4.4 按 `print_metric_every` 打印 total loss、HPEC loss、模块 2 辅助 loss、reconstruction loss、DAG loss、L1 loss 以及 accuracy、precision、recall、macro F1、ROC AUC

## 5. 参数入口、脚本与文档

- [x] 5.1 在 `run_cv.py` 中新增模块 4 HPEC 参数，help 文本使用中文描述
- [x] 5.2 更新 `test_training_smoke.py` 和 `test_matai_small_sample.py`，支持显式开启模块 4 并传入 HPEC 超参数
- [x] 5.3 新增 `docs/S-DeCI模块4-HPEC实现说明.md`，说明调用方式、关键参数、loss 构成、可视化输出和回退方式
- [x] 5.4 确认不直接修改 `models/DeCI.py` 的主模型逻辑

## 6. 可视化

- [x] 6.1 扩展 S-DeCI 中间量缓存或可视化调用，保存 HPEC prototype、angle matrix、psi/aperture、energy matrix、prediction 和 label 对照
- [x] 6.2 确保 train/test heatmap 文件名能区分 fold、数据划分和模块 4 内容
- [x] 6.3 最终 epoch t-SNE 使用 `logmap0(z_global)` 或等价 HPEC 输入表示，并用 marker 区分 train/test、颜色区分真实 label
- [x] 6.4 确保测试集 label 只用于 forward 后绘图标注，不输入模型

## 7. 验证

- [x] 7.1 运行 HPEC 层 shape/loss/prediction 的最小验证，确认输出形状和梯度可用
- [x] 7.2 运行 `S-DeCI` 模块 4 forward 验证，确认 `energy_matrix`、HPEC loss、prediction 和 probability 缓存存在
- [x] 7.3 运行低预算 cross-validation 训练，确认 HPEC loss、模块 2 auxiliary loss 和指标能正常打印
- [x] 7.4 验证分类 loss 能通过模块 4/3 回传到模块 2 因果图学习参数
- [x] 7.5 显式开启可视化运行一个低预算 fold，确认 HPEC heatmap 和最终 epoch train/test t-SNE 文件生成
- [x] 7.6 运行 `openspec validate add-module4-hpec-to-s-deci` 和必要的 Python 编译或冒烟测试


