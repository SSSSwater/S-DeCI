## 1. 现有实现核对

- [x] 1.1 阅读 `修改方案.md`、`models/S_DeCI.py`、`layers/hyperbolic_gcn_layer.py` 和 `layers/hpec_energy_layer.py`，确认当前模块 3/4 的输入输出、缓存和 loss 接入点。
- [ ] 1.2 核对当前 result/TensorBoard 中 HGCN/HPEC 与 GCN fallback 的表现，记录本变更要对比的 baseline。
- [x] 1.3 确认 `geoopt` 中可复用的 Lorentz / Poincare 接口；若接口不满足需求，则实现最小必要的 Lorentz exp/log/inner product 工具函数。

## 2. LP-Brain-HPEC 层实现

- [x] 2.1 新增 `layers/lp_brain_hpec_layer.py` 或等价文件，定义 Lorentz lifting、Directed Lorentz GCN、Lorentz tangent readout、Lorentz-to-Poincare bridge、MAC/HBR-HPEC 子模块。
- [x] 2.2 实现 Lorentz lifting：输入 `[B,N,d_in]`，输出 `[B,N,D+1]`，并保证 time-like 维度和 manifold constraint 数值稳定。
- [x] 2.3 实现 Directed Lorentz GCN：按 `A[parent, child]` 语义拆分入边 `A[:, i]` 与出边 `A[i, :]`，分别聚合后用可学习或可配置系数融合。
- [x] 2.4 支持 `[N,N]` 全局 adjacency 和 `[B,N,N]` 样本级 adjacency，错误形状给出清晰报错。
- [x] 2.5 实现 Lorentz tangent readout：先用 Lorentz 对数映射得到切空间节点表示，再按均值或 attention 权重聚合，最后用 Lorentz 指数映射得到 `[B,D+1]` 图级 Lorentz embedding；完整公式见 `design.md` 的 “Lorentz tangent readout” 小节。
- [x] 2.6 实现 stereographic bridge：按显式公式从 Lorentz 映射到 Poincare，输出 `[B,D]`。
- [x] 2.7 实现 MAC 半径裁剪，记录低半径/高半径 clip 比例和裁剪后半径分布。
- [x] 2.8 实现 HBR loss，支持 `hbr_loss_weight == 0` 时完全关闭。
- [x] 2.9 提供 `module34_geo_dtype` 配置，支持 `auto`、`float32`、`float64`，并处理输出 dtype 与主模型兼容。
- [x] 2.10 添加简洁中文注释，说明 Lorentz lifting、有向入/出边聚合、bridge、MAC/HBR 和 HPEC energy 的数据流。

## 3. S-DeCI 接入

- [x] 3.1 在 `models/S_DeCI.py` 中新增 `module34_arch` 参数，支持 `hgcn_hpec` 和 `lp_brain_hpec`。
- [x] 3.2 当 `module34_arch == "lp_brain_hpec"` 且模块 3/4 启用时，初始化 LP-Brain-HPEC 路径；现有 HGCN/HPEC 路径保持可回退。
- [x] 3.3 将模块 2 解析后的 learned/effective causal graph 传入 LP-Brain-HPEC，并保持 `A[parent, child]` 方向语义，不默认对称化。
- [x] 3.4 当模块 2 关闭时，LP-Brain-HPEC 使用 batch sample correlation matrix；缺失时清晰报错。
- [x] 3.5 在 LP-Brain-HPEC 路径下，保持 HPEC energy/prototype 为最终分类依据，不能默认退回 GCN fallback logits。
- [x] 3.6 将 HBR loss、HPEC energy loss、teacher distill、prototype separation 等与现有 primary loss / auxiliary loss 体系合并。
- [x] 3.7 缓存 Lorentz 节点表示、Lorentz graph embedding、Poincare bridge embedding、MAC 后 embedding、半径、clip 比例、HBR loss、energy matrix 和 prototype assignment。
- [x] 3.8 更新中间量可视化，使 LP-Brain-HPEC 路径可保存关键 heatmap/t-SNE，且测试 label 不进入 forward。

## 4. 训练入口与记录

- [x] 4.1 在 `run_cv.py` 中新增 `module34_arch`、Lorentz 层数/hidden dim、入出边平衡、`module34_geo_dtype`、MAC/HBR 参数，help 使用中文。
- [x] 4.2 在 `test_mdd_best_config.py` 中补齐 LP-Brain-HPEC 参数透传，并保持当前默认路径可回退。
- [x] 4.3 必要时同步 `test_abide_best_config.py`，确保不会因新增参数缺失而初始化失败。
- [x] 4.4 更新 `exp/exp_classification_CV.py` 的 loss/diagnostics 收集，支持记录 Lorentz constraint、MAC clip ratio、MAC radius、HBR loss、入/出边聚合强度。
- [x] 4.5 更新 TensorBoard `CompareTrend` 和每 fold scalar，加入 LP-Brain-HPEC 诊断量。
- [x] 4.6 更新 `result.xlsx` 写入参数列，至少记录 `module34_arch`、核心 MAC/HBR 参数和最终指标。

## 5. 文档与规范同步

- [x] 5.1 新建中文实现说明文档，描述 LP-Brain-HPEC 数据流、张量形状、采纳/适配 `修改方案.md` 的地方、默认参数和回滚方式。
- [x] 5.2 不覆盖 `docs/新模块设计.md` 等原始参考文档。
- [x] 5.3 如实现中对 proposal/spec 有调整，更新本 change 的 specs 和主规范，保持中文描述。

## 6. 测试与实验

- [x] 6.1 运行 `py_compile` 覆盖新增 layer、`models/S_DeCI.py`、`run_cv.py`、`test_mdd_best_config.py` 和实验流程。
- [x] 6.2 编写或运行随机张量 forward/backward 检查：`[B,N,D]` 节点特征、`[N,N]` 与 `[B,N,N]` adjacency 均无 NaN/Inf，并能反向更新参数。
- [x] 6.3 运行 MDD 1fold 低预算 smoke test，确认 LP-Brain-HPEC 路径训练、指标、TensorBoard 和可视化不报错。
- [ ] 6.4 运行 MDD AAL116 完整 5-fold 50epoch，对比 `gcn_fallback`、`hgcn_hpec` 和 `lp_brain_hpec`。
- [ ] 6.5 将完整 5-fold final-epoch 指标写入 `result.xlsx`，并用 TensorBoard 检查训练/测试趋势和几何诊断。
- [ ] 6.6 若 LP-Brain-HPEC 未优于现有路径，如实记录结果，并保持默认参数回退到当前更稳路径。
