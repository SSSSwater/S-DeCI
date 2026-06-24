# S-DeCI 模块4 HPEC 实现说明

本文档说明 `S-DeCI` 中模块 4 HPEC energy 分类的实现和调用方式。原始 `docs/新模块设计.md` 保持不变，本文件用于记录落地实现细节。

## 功能概览

模块 4 接在模块 3 之后：

1. 模块 1 输出 `Cycle/seasonal feature C`。
2. 模块 2 学习因果图 `A_learned` 并重构 `C_hat`。
3. 模块 3 使用 `C` 和 `A_learned` 经过 HGCN 得到双曲中心点 `z_global`。
4. 模块 4 使用 `z_global` 与类别 `prototype` 计算 HPEC angle、`psi/aperture` 和 `energy_matrix`。
5. 推理时使用 `argmin(energy_matrix)` 作为预测类别。

启用模块 4 时，普通线性分类 loss 被替换为：

```text
Loss_total =
    Loss_HPEC(z_global, label)
  + alpha * Loss_Recon(C, C_hat)
  + lambda * Loss_DAG(A_learned)
  + gamma * L1(A_learned)
```

其中真实因果矩阵不会作为监督信号参与训练。

## 关键文件

- `layers/hpec_energy_layer.py`：HPEC prototype 初始化、angle、psi、energy、loss 和 prediction。
- `models/S_DeCI.py`：模块 4 初始化、forward 接入、中间量缓存、HPEC primary loss。
- `exp/exp_classification_CV.py`：优先使用模型内部 HPEC loss，指标使用 `argmin(energy)` 和 `softmax(-energy)`。
- `run_cv.py`：模块 4 参数入口。
- `test_training_smoke.py`、`test_matai_small_sample.py`：根目录测试脚本参数入口。

## 常用参数

- `--use_hpec_module4`：是否启用模块 4，启用时需要 `--use_hgcn_module3 1`。
- `--hpec_prototype_radius`：类别 prototype 半径，默认 `0.3`。
- `--hpec_cone_k`：HPEC cone aperture 的 `K` 参数，默认 `0.1`。
- `--hpec_margin`：非真实类别 energy 的 margin，默认 `1.0`。
- `--hpec_trainable_prototypes`：prototype 是否参与训练，默认 `0`。
- `--hpec_init_steps`：hyperspherical separation 初始化步数。
- `--hpec_eps`：数值稳定用 eps。

根目录测试脚本使用横线风格参数，例如：

```bash
python test_training_smoke.py --use-hpec-module4 1 --hpec-init-steps 50
```

`run_cv.py` 使用下划线风格参数，例如：

```bash
python run_cv.py --model S-DeCI --use_hgcn_module3 1 --use_hpec_module4 1
```

## 可视化

显式设置 `visualize_causal=1` 后，每个 fold 训练完成会保存 train/test 中间量 heatmap，并在最终 epoch 后保存 train/test 联合 t-SNE。

模块 4 新增可视化内容包括：

- HPEC prototypes
- HPEC angle matrix
- HPEC psi/aperture
- HPEC energy matrix
- `softmax(-energy)` probability
- predicted labels
- ground truth labels

测试集真实 label 只用于 forward 之后绘图标注，不会作为模型输入。

## 回退方式

设置 `--use_hpec_module4 0` 即可回退到模块 3 的 `logmap0(z_global)` 线性分类头；若同时设置 `--use_hgcn_module3 0`，则回退到原 Cycle/seasonal logits 分类路径。
