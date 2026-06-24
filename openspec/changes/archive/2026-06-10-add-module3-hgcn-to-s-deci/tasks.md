## 1. 依赖与参考代码确认

- [x] 1.1 检查 `.venv` 中是否可 import `geoopt`，若缺失则补充依赖并更新 `requirements.txt`
- [x] 1.2 阅读 `docs/新模块设计.md` 的模块 3 章节，确认 Backclip、Poincare 投影、HGCN、Fréchet readout 的输入输出语义
- [x] 1.3 阅读 `reference/HPEC-main/` 中 `geoopt`、Poincare Ball 和黎曼优化相关调用方式
- [x] 1.4 阅读 `reference/Differentiable-Frechet-Mean-master/` 中 Fréchet mean/readout 相关实现
- [x] 1.5 确认正式实现不从 `reference/` 目录直接 import 运行时组件

## 2. 模块 3 HGCN 层实现

- [x] 2.1 在 `layers/` 下新增模块 3 HGCN/双曲 readout 层文件
- [x] 2.2 实现 Backclip 或等价限幅逻辑，输入输出保持 `[B, N, d_model]`
- [x] 2.3 使用 `geoopt` Poincare Ball 实现 `expmap0` 投影、`logmap0` 和必要的 `projx` 稳定化
- [x] 2.4 实现特征升维，将 `d_model` 映射到可配置的 `hgcn_hidden_dim`，默认 `128`
- [x] 2.5 使用模块 2 的 `A_learned` 实现 HGCN 图传播，输出 `H_gcn: [B, N, hgcn_hidden_dim]`
- [x] 2.6 实现可微 Fréchet readout 或设计中声明的可微切空间均值 readout，输出 `z_global: [B, hgcn_hidden_dim]`
- [x] 2.7 缓存 `C_clipped`、Poincare 投影结果、`H_gcn`、`z_global` 和 `logmap0(z_global)` 等诊断量
- [x] 2.8 在 HGCN 层关键数学逻辑处添加中文注释，必要英文关键词保留

## 3. S-DeCI 模块 3 接入

- [x] 3.1 修改 `models/S_DeCI.py`，按配置初始化模块 3
- [x] 3.2 将模块 1 聚合后的 Cycle feature `C` 和模块 2 输出的 `A_learned` 输入模块 3
- [x] 3.3 使用 `z_global` 或 `logmap0(z_global)` 作为当前阶段分类依据
- [x] 3.4 保持 `S-DeCI.forward()` 主返回值仍为分类输出
- [x] 3.5 在 `use_hgcn_module3=0` 时保留原 Cycle/seasonal logits 分类路径
- [x] 3.6 确保当前阶段不新增模块 4、HPEC 原型角度损失或能量分类器
- [x] 3.7 在模块 3 初始化、数据流、`z_global` 分类和缓存逻辑处添加中文注释

## 4. 联合损失与梯度路径

- [x] 4.1 保持分类 loss 使用 `z_global` 分类头输出计算
- [x] 4.2 保持模块 2 reconstruction、归一化 DAG、归一化 L1 sparsity loss 参与总 loss
- [x] 4.3 确保总 loss 结构为 `Loss_cls(z_global,label) + alpha*Loss_Recon + lambda*Loss_DAG + gamma*L1`
- [x] 4.4 确保训练 loss 不使用 `A_true`、`A_structure_true` 或任何真实因果矩阵监督
- [x] 4.5 确保不提供阻断分类 loss 回传到模块 2 因果图的配置开关
- [x] 4.6 验证 `Loss_cls(z_global,label)` 的梯度能回传到模块 2 因果图学习参数
- [x] 4.7 检查 `exp/exp_classification_CV.py` 训练循环对模块 2/3 联合 loss 的读取和反传仍然正确

## 5. 参数入口与配置

- [x] 5.1 在 `run_cv.py` 增加模块 3 相关 CLI 参数
- [x] 5.2 在 `test_training_smoke.py` 增加模块 3 默认参数和 CLI 参数
- [x] 5.3 在 `test_matai_small_sample.py` 增加模块 3 默认参数和 CLI 参数
- [x] 5.4 支持配置 `use_hgcn_module3`、`hgcn_hidden_dim`、HGCN 层数、曲率和 Backclip 半径或等价参数
- [x] 5.5 确认默认 `hgcn_hidden_dim=128`
- [x] 5.6 确认常规训练默认不保存模块 3 可视化图片

## 6. 中间量可视化

- [x] 6.1 扩展 `S-DeCI` 可视化方法或新增模块 3 可视化方法
- [x] 6.2 可视化 `C_clipped`、Poincare 投影结果或 `H0`、`H_gcn`、`z_global` 或 `logmap0(z_global)`
- [x] 6.3 可视化 `A_learned` 和 `A_learned - A_learned.T`，体现模块 2 到模块 3 的衔接
- [x] 6.4 复用 `utils.tensor_visualization.visualize_tensors`，保持 3D 张量只显示 Batch0 或指定 batch
- [x] 6.5 验证显式开启可视化时能生成 heatmap 图片
- [x] 6.6 验证默认配置下训练、验证、推理不保存模块 3 heatmap 图片

## 7. 文档与说明

- [x] 7.1 不修改原始 `docs/新模块设计.md`
- [x] 7.2 新建模块 3 实现说明文档，例如 `docs/S-DeCI模块3实现说明.md`
- [x] 7.3 在说明文档中记录当前阶段不实现模块 4
- [x] 7.4 在说明文档中记录 `z_global` 分类、联合 loss、可视化入口和 Fréchet readout 实现方式

## 8. 验证与回归测试

- [x] 8.1 运行 HGCN 层 forward shape 测试，验证 `H_gcn` 和 `z_global` 形状
- [x] 8.2 运行 `S-DeCI` forward smoke test，验证二分类输出 `[B, 1]`
- [x] 8.3 运行 `S-DeCI` 多分类 forward smoke test，验证输出 `[B, classes]`
- [x] 8.4 运行反向传播测试，确认分类 loss 能回传到模块 2 因果图参数
- [x] 8.5 运行低预算 `test_training_smoke.py`，确认 `S-DeCI` + 模块 2 + 模块 3 训练能跑通
- [x] 8.6 运行显式可视化测试，确认模块 2/3 中间量图片保存成功
- [x] 8.7 运行 `py_compile` 覆盖新增 HGCN 层、`models/S_DeCI.py`、训练入口和测试脚本
- [x] 8.8 确认 `models/DeCI.py` 主模型逻辑未被本次变更直接修改
