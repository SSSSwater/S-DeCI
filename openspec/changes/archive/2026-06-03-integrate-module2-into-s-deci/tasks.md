## 1. 模块 2 可复用组件

- [x] 1.1 新建正式可复用的模块 2 因果学习文件，例如 `layers/causal_graph_learner.py`
- [x] 1.2 从 `module_2_test/` 迁移或复用 `CausalGraphLearner`、DAG penalty、阈值化邻接矩阵等核心逻辑
- [x] 1.3 确保正式模型不从 `module_2_test/` 直接 import 生产训练所需组件
- [x] 1.4 保留 `module_2_test/` 独立测试目录原有用途，不破坏其合成数据和训练检查脚本

## 2. DeCI block 与 S-DeCI 特征接入

- [x] 2.1 为 `DeCI_Block` 增加可选返回 trend/seasonal feature 的能力，并保持默认返回行为不变
- [x] 2.2 修改 `models/S_DeCI.py`，在 forward 中获得 `[B, N, d_model]` 的 Cycle/seasonal feature
- [x] 2.3 在多层 block 场景下聚合 Cycle/seasonal feature，默认与 seasonal logits 求和语义保持一致
- [x] 2.4 保持 `S-DeCI.forward()` 主返回值仍为分类输出 `y_hat`
- [x] 2.5 在 `models/S_DeCI.py` 新增关键逻辑处添加中文注释，说明模块 2 输入、分类约束、loss 缓存和可视化触发

## 3. S-DeCI 模块 2 损失与诊断量

- [x] 3.1 在 `S-DeCI` 初始化中按配置创建模块 2 因果图学习器
- [x] 3.2 在 forward 中将聚合后的 Cycle/seasonal feature 输入模块 2 并缓存 `A_learned`、`C_hat` 等中间量
- [x] 3.3 计算并缓存 reconstruction、归一化 DAG acyclicity、归一化 L1 sparsity loss
- [x] 3.4 提供 `get_aux_loss()` 或等价方法，使训练流程能读取总辅助 loss
- [x] 3.5 支持通过配置关闭模块 2 或将其辅助 loss 权重设为 `0`

## 4. 训练流程集成

- [x] 4.1 修改 `exp/exp_classification_CV.py`，训练阶段在分类 loss 后读取并叠加模型辅助 loss
- [x] 4.2 兼容普通模型和 `DataParallel` 模型，只有模型提供辅助 loss 时才叠加
- [x] 4.3 保持验证阶段默认分类 loss 和分类指标逻辑不变
- [x] 4.4 在 `run_cv.py` 或测试脚本参数中补充模块 2 相关默认配置

## 5. 中间量可视化

- [x] 5.1 在 `S-DeCI` 中提供显式可视化方法或配置触发逻辑，默认不执行可视化
- [x] 5.2 使用 `utils.tensor_visualization.visualize_tensors` 可视化 Cycle/seasonal feature、`C_hat`、重构误差、`A_learned` 和阈值化邻接矩阵
- [x] 5.3 支持指定可视化保存目录或保存路径，并让文件名能区分不同中间量
- [x] 5.4 限制训练中可视化保存频率，避免默认每个 batch 保存图片

## 6. 验证与回归测试

- [x] 6.1 运行 S-DeCI forward smoke test，验证分类输出形状保持兼容
- [x] 6.2 验证模块 2 输入输出形状为 `C: [B, N, d_model]`、`A_learned: [N, N]`、`C_hat: [B, N, d_model]`
- [x] 6.3 验证辅助 loss 可反向传播，且 `A_learned` 参数能收到梯度
- [x] 6.4 验证显式开启可视化时能生成 heatmap 图片，默认关闭时不保存图片
- [x] 6.5 使用 `.venv` 运行低预算 `S-DeCI` 训练测试，确认 `run_cv.py --model S-DeCI` 路径能跑通
- [x] 6.6 确认 `models/DeCI.py` 主模型逻辑未被直接修改
