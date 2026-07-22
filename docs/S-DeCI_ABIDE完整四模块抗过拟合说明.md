# S-DeCI ABIDE 完整四模块抗过拟合说明

本文档记录 `improve-abide-full-sdeci-anti-overfit` 变更的工程实现。原始设计参考仍以 `docs/新模块设计.md` 为准，本文只说明本次 ABIDE-120 训练入口和实现细节。

## 主路径

`test_abide_best_config.py` 默认使用完整 S-DeCI：

- 模块 1：启用 DeCI/Cycle，并在训练期加入随机时间窗、temporal dropout、ROI dropout 与可选 denoising auxiliary loss。
- 模块 2：启用 temporal SEM 因果图学习，输入优先来自模块 1 增强后的时序，并保留 DAG、稀疏、平滑和样本残差图正则。
- 模块 3：启用 HGCN 双曲 readout，使用模块 2 输出的因果图作为 adjacency，并在训练期支持 causal edge dropout。
- 模块 4：启用 HPEC 多 prototype energy loss，支持 trainable prototype / EMA 更新，并加入 teacher distill、`z_global` 半径约束与 prototype separation 诊断。

GCN fallback 仍保留，但只作为显式消融对照。ABIDE 默认配置不关闭模块 2，也不关闭模块 3/4。

## 默认参数

ABIDE 专用脚本默认使用 `seq_len=120`、`use_deci_module1=1`、`use_causal_module2=1`、`use_hyperbolic_modules34=1`、`use_hpec_module4=1`，并关闭站点对抗分支。

新增抗过拟合参数包括：

- `--module1-random-crop 1`
- `--module1-temporal-dropout 0.08`
- `--module1-roi-dropout 0.05`
- `--module1-denoise-loss-weight 0.02`
- `--lambda-causal-stability 0.002`
- `--temporal-sample-graph-delta-scale 0.01`
- `--lambda-sample-graph-l1 0.0001`
- `--lambda-sample-graph-deviation 0.001`
- `--causal-edge-dropout 0.15`
- `--hpec-teacher-distill-weight 1.0`
- `--hpec-z-radius-loss-weight 0.1`
- `--hpec-prototype-separation-loss-weight 0.2`
- `--hpec-trainable-prototypes 1`

## 诊断输出

训练日志会按类别输出 loss 和指标，包括模块 1 denoise loss、模块 2 temporal SEM / DAG / stability loss、模块 3 双曲表示诊断、模块 4 HPEC final CE / energy / teacher distill / 半径约束 / prototype separation，以及 train/test accuracy、precision、recall、macro F1、ROC AUC。

显式设置 `--visualize-causal 1` 时，会保存 train/test 中间量 heatmap 和最终 epoch 的 train/test t-SNE。t-SNE 使用 train/test 不同 marker、label 不同颜色，并显示 HPEC prototype。

## 回滚方式

若需要回到旧行为，可以将新增正则和增强关闭：

```powershell
--module1-random-crop 0 --module1-temporal-dropout 0 --module1-roi-dropout 0 `
--module1-denoise-loss-weight 0 --lambda-causal-stability 0 `
--causal-edge-dropout 0 --hpec-teacher-distill-weight 0 `
--hpec-z-radius-loss-weight 0 --hpec-prototype-separation-loss-weight 0
```

若只做 GCN fallback 消融，显式设置：

```powershell
--use-causal-module2 0 --use-hyperbolic-modules34 0 --use-sample-correlation-when-module2-disabled 1
```

该路径仅用于对照，不作为 ABIDE 主方案。

## 测试命令

快速语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile models\S_DeCI.py layers\hyperbolic_gcn_layer.py layers\hpec_energy_layer.py exp\exp_classification_CV.py run_cv.py test_abide_best_config.py data_provider\data_factory_CV.py
```

完整四模块 smoke test：

```powershell
.\.venv\Scripts\python.exe test_abide_best_config.py --iterations 1 --max-folds 1 --train-epochs 3 --visualize-causal 0 --print-metric-every 1
```
