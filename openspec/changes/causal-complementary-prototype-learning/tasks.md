## 1. 参数与兼容基础

- [x] 1.1 在 `run_cv.py`、四个最佳配置脚本和 `sweep_hparam.py` 中新增可靠 prototype、互补视图、多阶因果编码的中文 CLI 参数，并保持既有配置可运行。
- [x] 1.2 在 `S_DeCI` 中集中归一化新开关，校验模块 3/4 关闭时的非法组合，默认关闭训练期互补和多阶编码。
- [x] 1.3 补充模型缓存结构，使标准/互补输出、视图 loss、显著性和 prototype 更新统计可由训练循环读取，且 `forward()` 返回值不变。

## 2. 可靠 TP 多原型更新

- [x] 2.1 在 `layers/hpec_energy_layer.py` 实现 `hpec_prototype_update_mode` 及 `reliable_tp_ema` 初始化逻辑，保留 `sinkhorn_ema` 和 `none` 回退。
- [x] 2.2 实现标准 logits 的 true-positive/置信度筛选、可选标准-互补切空间一致性权重和类内 energy winner prototype 分配。
- [x] 2.3 在无梯度上下文实现切空间可靠样本均值、anchor 混合、EMA、半径壳约束和 Poincare 投影；样本不足时保持原 prototype。
- [x] 2.4 在训练循环的 `optimizer.step()` 后调用独立 prototype 更新接口，确保验证、测试、推理不更新，并使该更新不进入总 loss 或第二次 backward。
- [x] 2.5 输出可靠 TP 比例、类别/原型分配数、EMA 位移、未更新计数和 assignment entropy 到模型诊断与 TensorBoard。
- [x] 2.6 为三种 prototype 更新模式编写张量级单元/快速验证，覆盖形状、无 NaN、无梯度移动和测试期冻结。

## 3. 因果显著性互补视图

- [x] 3.1 从标准模块 3 节点表征和 `A_cls[parent, child]` 计算 detach 后的语义/拓扑显著性，并实现可配置融合。
- [x] 3.2 实现 warm-up、渐进比例和 Gumbel-top-k ROI 遮挡；遮挡仅作用于模块 3 输入节点特征，不改动模块 1/2 时间序列和 `A_cls`。
- [x] 3.3 在训练期以共享模块 3/4 参数执行互补分支，复用标准 `A_cls`，并在评估/推理期跳过该分支。
- [x] 3.4 实现可选 Poincare 双视图距离一致性项，按权重写入总 loss；未启用时保持零贡献。
- [x] 3.5 添加 `Complementary/*` TensorBoard 标量及最后 epoch 标准/互补 t-SNE、显著性 ROI 诊断输出。
- [x] 3.6 运行单 fold 短训，验证互补分支不重复模块 2、不改变标准预测接口，且关闭开关时计算路径与现有模型一致。

## 4. 多阶因果可达性编码

- [x] 4.1 在模块 3 或独立轻量层实现从 `abs(A_cls)` 构造有向行归一化 `P`，并以 `(P^l)^T` 聚合 parent 到 child 的 1 至 L 阶节点信息。
- [x] 4.2 实现可学习 hop gate 与残差 scale，将多阶编码注入 HGCN 输入但不写回 `A_cls`。
- [x] 4.3 添加 `CausalReachability/*` 诊断，记录每 hop gate、编码范数和最终残差范数。
- [x] 4.4 验证 2D 全局图和 3D 样本图均可运行，关闭开关时不计算矩阵幂且输出与原 HGCN 接口兼容。

## 5. 训练记录与设计说明

- [x] 5.1 在 `exp/exp_classification_CV.py` 将新增 loss、prototype 更新统计和多阶诊断写入每 fold、汇总 TensorBoard run。
- [x] 5.2 扩展 `sweep_hparam.py`，使其可扫描新增开关及数值参数，并在结束后以清晰表格汇总完整实验指标和训练时长。
- [x] 5.3 新建 `docs/S-DeCI因果显著性互补原型学习设计说明.md`，用中文说明数据流、公式、BrainCL 来源、开关、回退和诊断，不修改既有 `docs/` 参考设计。
- [x] 5.4 更新相关 OpenSpec 主规范增量所需的中文描述，保持默认路线和 legacy 路线边界清晰。

## 6. 分层实验与验收

- [x] 6.1 运行静态检查和单 fold/短 epoch smoke，验证默认关闭、Sinkhorn legacy、可靠 TP EMA、互补视图和多阶编码均无 NaN 且可完成 backward。
- [x] 6.2 在固定 MDD/AAL116 划分上完成 5-fold、至少 50 epoch 的 `sinkhorn_ema` 与 `reliable_tp_ema` 对比，并记录指标与训练时长。
- [x] 6.3 在同一完整设置下依次比较可靠 TP EMA、互补视图无一致性项、互补视图加一致性项及多阶因果编码，写入 `result.xlsx`。
- [x] 6.4 检查 TensorBoard、因果图、标准/互补 t-SNE、prototype 更新统计和测试集指标，判断是否存在 prototype 坍缩、少数 ROI 依赖或训练测试显著背离。
- [x] 6.5 仅在完整 5-fold 结果稳定改善 Accuracy、Macro-F1 或 AUC 且训练成本可接受时更新各数据集默认配置；否则保留为显式实验开关并报告回退配置。
