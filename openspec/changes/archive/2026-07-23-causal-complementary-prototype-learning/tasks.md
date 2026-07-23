## 1. 可靠 TP 多原型 EMA

- [x] 1.1 实现 `reliable_tp_ema`，按预测正确和置信度筛选标准训练样本。
- [x] 1.2 在 `optimizer.step()` 后以无梯度 EMA 更新每类多个 Poincare prototype。
- [x] 1.3 保留 `sinkhorn_ema` 和 `none` 对照，并输出可靠比例、分配、位移和未更新统计。

## 2. 多阶因果可达性编码

- [x] 2.1 从 `abs(A_cls[parent,child])` 构造行归一化有向转移矩阵。
- [x] 2.2 以 1 至 L 阶转置传播使 child 聚合 causal parent 特征，并通过 hop gate 和残差 scale 注入模块 3。
- [x] 2.3 记录各 hop gate、编码范数与残差范数；关闭时不计算额外矩阵幂。

## 3. 互补学习实验

- [x] 3.1 历史实现因果显著性遮挡、标准/互补共享前向、Poincare 一致性、InfoNCE 与 masked CE。
- [x] 3.2 完成 MDD/AAL116 5-fold、至少 50 epoch 的同划分消融和 TensorBoard 检查。
- [x] 3.3 确认互补组合未超过可靠 TP EMA 基线，InfoNCE 接近 `log(batch_size)`，不作为正向能力保留。

## 4. 负向路径清理

- [x] 4.1 从正式模型 forward 移除互补遮挡与第二次模块 3/4 前向。
- [x] 4.2 从正式总 loss 和训练入口移除互补一致性、InfoNCE、masked CE 及其参数。
- [x] 4.3 在证据台账中保留完整负向原因，避免后续重复引入。
- [x] 4.4 从 `models/S_DeCI.py` 和 `exp/exp_classification_CV.py` 移除遗留互补缓存、t-SNE 收集分支、epoch accumulator 与 `Complementary/*` TensorBoard 映射。
- [x] 4.5 移除仅有单折证据的 `epoch_reliable_frechet_ema` 参数、实现分支和回归用例；如未来需要保留，必须先通过独立 change 完成固定划分的完整五折对照。

## 5. 规范与验证

- [x] 5.1 更新 proposal、design、delta specs 和主规范，区分正式保留、可选消融与已移除能力。
- [x] 5.2 核对 `forward()` 返回兼容、prototype 仅训练期更新、验证/测试不读取标签参与预测。
