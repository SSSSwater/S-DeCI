## 1. 参数入口与默认配置

- [x] 1.1 在 `run_cv.py` 中新增模块 1 去噪、模块 2 稳定性、模块 3 edge dropout/半径正则、模块 4 prototype 正则相关参数，help 使用中文说明。
- [x] 1.2 更新 `test_abide_best_config.py`，使默认配置启用 `S-DeCI` 模块 1、模块 2、模块 3、模块 4，并使用 ABIDE-120。
- [x] 1.3 保留 GCN fallback 参数作为显式消融开关，但确保 ABIDE 默认不走 fallback。
- [x] 1.4 确认新增参数写入 `SimpleNamespace` experiment args，并保持 MDD 与其他脚本旧行为兼容。

## 2. 模块 1 去噪增强

- [x] 2.1 在数据或模型输入路径中实现训练期随机时间窗裁剪，验证/测试使用确定性裁剪。
- [x] 2.2 在模块 1 输入或模块 1 内部实现 `temporal dropout` 和 `ROI dropout`，仅在 training 模式启用。
- [x] 2.3 在 `S-DeCI` 中缓存模块 1 去噪输出，供模块 2、可视化和诊断读取。
- [x] 2.4 实现可选 denoising auxiliary loss，并用 `module1_denoise_loss_weight` 控制是否加入总 loss。

## 3. 模块 2 稳定因果学习

- [x] 3.1 调整模块 2 输入，使 `temporal_sem` 优先使用模块 1 去噪后的时序或节点表示。
- [x] 3.2 实现 `lambda_causal_stability` 控制的因果图稳定性 loss，支持增强视图或等价稳定图约束。
- [x] 3.3 降低 ABIDE 默认样本残差图自由度，配置更保守的 `sample_graph_delta_scale`、L1 和 deviation 正则。
- [x] 3.4 在模块 2 诊断中输出 graph mass、方向性、stability loss 和样本残差图强度。
- [x] 3.5 确保模块 2 loss 不使用真实因果矩阵监督。

## 4. 模块 3 双曲投影正则

- [x] 4.1 在模块 3 adjacency 处理前实现训练期 `causal_edge_dropout`，推理期不丢边。
- [x] 4.2 缓存并日志输出 `z_global` 半径、`z_tangent` 范数或等价双曲诊断。
- [x] 4.3 实现可选 `lambda_hgcn_radius_reg` 半径正则，并接入总 loss。
- [x] 4.4 确认 HPEC 分类梯度默认不 detach，能够通过模块 3 回传到模块 2 因果图参数。

## 5. 模块 4 HPEC 与多原型正则

- [x] 5.1 实现或完善训练 fold 内 prototype warm-start，禁止使用验证/测试样本。
- [x] 5.2 增加 prototype 半径约束与类间 margin 约束，避免 HPEC energy 不稳定。
- [x] 5.3 增加多 prototype diversity 约束，避免同类 prototype collapse。
- [x] 5.4 输出每个样本的 prototype assignment、prototype-level energy 和类别级 energy 诊断。
- [x] 5.5 调整 ABIDE 默认的 `lambda_hpec_mle`、`lambda_hpec_pcl`、`lambda_hpec_pal` 为温和抗过拟合设置。

## 6. 训练流程、日志和可视化

- [x] 6.1 更新 `exp/exp_classification_CV.py` 的 loss 汇总，加入模块 1、模块 2、模块 3、模块 4 新增 loss 项。
- [x] 6.2 更新 epoch 日志，按类别打印 total loss、HPEC loss、模块 1 去噪 loss、模块 2 因果 loss、模块 3 正则、prototype loss、train/test 指标。
- [x] 6.3 更新中间量可视化，显式开启时保存模块 1 去噪输出、模块 2 因果图、模块 3 双曲表示、模块 4 prototype energy。
- [x] 6.4 确保最终 t-SNE 区分 train/test 样式、label 颜色，并能显示 prototype。

## 7. 文档

- [x] 7.1 新建中文实现说明文档，记录 ABIDE 完整四模块抗过拟合设计、默认参数、回滚方案和测试命令。
- [x] 7.2 不修改 `docs/新模块设计.md` 原始参考文档。
- [x] 7.3 在说明文档中记录 GCN fallback 仅作为消融对照，不作为 ABIDE 主路径。

## 8. 验证

- [x] 8.1 运行 `python -m py_compile` 检查修改过的 Python 文件。
- [x] 8.2 运行 `test_abide_best_config.py --iterations 1 --max-folds 1 --train-epochs <适中值>`，确认完整四模块训练跑通。
- [x] 8.3 对比旧 GCN fallback 结果，报告完整四模块路径的 accuracy、precision、recall、macro F1 和 ROC AUC。
- [x] 8.4 显式开启一次 `--visualize-causal 1`，确认 heatmap 和 t-SNE 文件生成且不把测试 label 输入模型。
- [x] 8.5 如完整四模块默认表现仍过拟合，记录训练/验证差异和下一轮需要调整的模块内参数。
