## Context

当前 `S-DeCI` 已具备模块 1 的 Cycle/seasonal feature、模块 2 的因果图学习和模块 3 的 HGCN 双曲 readout。模块 3 输出 `z_global: [B, hgcn_hidden_dim]`，并缓存 `logmap0(z_global)` 作为线性分类头输入。`docs/新模块设计.md` 的模块 4 要求进一步引入 HPEC：构建类别原型，计算 `z_global` 与各原型之间的角度能量，并用 HPEC energy loss 替换当前普通分类 loss。

参考实现来自 `reference/HPEC-main/`，其中核心包括：

- `entailment.py` 的 `func_psi`、`func_angle_vec`、`Entailment_loss` 和 `predict_entailment`。
- `models.py` 中将 hyperspherical prototypes 归一化、缩放到 `prototypes_ray` 后用 `PoincareBall.expmap0` 投影到双曲空间。
- `utils.py` 中通过 `SeparationLoss` 学习最大间隔原型初始化。

正式实现不得在运行时从 `reference/` import；需要将必要公式迁移到项目代码中，并添加中文注释。

## Goals / Non-Goals

**Goals:**

- 在 `S-DeCI` 中新增模块 4 HPEC energy 分类路径。
- 使用模块 3 的 `z_global` 或 `logmap0(z_global)` 与类别原型计算 energy matrix `[B, classes]`。
- 启用模块 4 时，将分类 loss 从普通 MSE/CE 线性分类损失替换为 HPEC energy loss。
- 保持联合损失：

```text
Loss_total =
    Loss_HPEC(z_global, label)
  + alpha * Loss_Recon(C, C_hat)
  + lambda * Loss_DAG(A_learned)
  + gamma * L1(A_learned)
```

- 让 `Loss_HPEC` 通过模块 4、模块 3 和 `A_learned` 回传到模块 2 因果图参数。
- 为 HPEC 原型、角度、energy、预测结果和相关中间量提供缓存与可视化。
- 保留 `use_hpec_module4=0` 回退到当前模块 3 线性分类路径。

**Non-Goals:**

- 不修改原始 `docs/新模块设计.md`，只新增实现说明文档。
- 不修改 `models/DeCI.py` 的原始模型逻辑。
- 不直接依赖 `reference/` 目录作为运行时 import。
- 不引入真实因果矩阵监督。
- 不在本次变更中重新设计模块 2 或模块 3 的核心结构。

## Decisions

### 1. HPEC 公式迁移到 `layers/`，模块装配保留在 `S-DeCI`

新增文件建议为 `layers/hpec_energy_layer.py`，包含：

- `HPECOutput`
- `HPECPrototypeEnergy`
- `hpec_angle`
- `hpec_aperture`
- `hpec_energy_loss`
- `predict_hpec`

`models/S_DeCI.py` 负责按配置初始化模块 4、调用模块 4、缓存输出、向训练循环暴露 HPEC loss。

选择原因：HPEC 的角度/孔径/energy 是可复用数学组件，适合放在 `layers/`；`S-DeCI` 保持数据流编排，便于继续调试模块 1-4 的衔接。

备选方案：全部写入 `models/S_DeCI.py`。实现更快，但会让模型文件承担过多数学细节，后续难以单独测试。

### 2. HPEC 输入默认使用双曲点 `z_global`

模块 4 默认接收 `z_global` 与双曲原型 `prototypes`，使用 reference 中 Poincare Ball 坐标上的角度公式计算：

```text
angle = Xi(prototype_k, z_global)
psi = aperture(prototype_k, K)
energy = max(0, angle - psi)
```

同时缓存 `z_tangent = logmap0(z_global)`，用于 t-SNE 和诊断。若数值不稳定，可提供 `hpec_input_space=tangent` 作为调试回退，但默认严格走 HPEC 的 Poincare 表示。

选择原因：`docs/新模块设计.md` 明确要求用 `z_global` 与 HPEC 原型计算角度能量；reference 公式也是在 Poincare Ball 坐标上计算。

### 3. 原型初始化采用 hyperspherical separation 后投影到 Poincare Ball

实现项目内的原型初始化：

1. 用 `classes x hgcn_hidden_dim` 随机向量初始化。
2. 在欧氏单位球面上用 separation loss 优化，使类别原型尽量分离。
3. 将原型归一化并缩放到 `hpec_prototype_radius`。
4. 用模块 3 相同曲率的 Poincare Ball `expmap0` 投影得到双曲原型。

默认原型固定不训练，提供 `hpec_trainable_prototypes` 开关。若启用训练，可使用普通 Adam 优化底层参数并每次 forward 投影，也可后续扩展为 `geoopt.ManifoldParameter`。

选择原因：HPEC reference 默认使用固定原型；固定原型更适合先验证模块 4 loss 是否稳定，也减少优化器复杂度。

### 4. HPEC loss 替换分类 criterion，但保留训练循环总 loss 结构

当前训练循环会执行：

```python
y_hat = model(x_enc)
loss = criterion(y_hat, label)
aux_loss = model.get_aux_loss()
total_loss = loss + aux_loss
total_loss.backward()
```

模块 4 启用后需要让 `criterion` 不再计算普通 MSE/CE。设计方案：

- `S-DeCI.forward()` 返回可用于指标计算的 prediction scores，建议返回 `energy_scores = -energy`，形状仍为 `[B, classes]`；二分类兼容时可提供 `[B, 1]` 概率或保留 energy-based prediction 分支。
- `S-DeCI` 缓存 `hpec_loss`。
- `Exp_Main` 在检测到 `model.get_cls_loss()` 或 `model.get_primary_loss()` 可用时，优先使用模型内部分类 loss；否则使用外部 criterion。

更明确的训练逻辑：

```python
y_hat = model(x_enc, label=label_for_loss)  # 或 forward 后调用模型方法
cls_loss = model.get_primary_loss() or criterion(y_hat, label)
aux_loss = model.get_aux_loss()
total_loss = cls_loss + aux_loss
total_loss.backward()
```

为了避免改变所有模型接口，本次建议采用 `S-DeCI.set_current_labels(label)` 之前不优雅；更干净的做法是让训练循环在 `model(x_enc)` 后调用：

```python
primary_loss = model.compute_primary_loss(label)
```

该方法仅对 `S-DeCI` 模块 4 生效，其他模型继续走 criterion。

选择原因：HPEC loss 需要 label 和原型 energy，放在模型内部最容易访问缓存的 energy matrix，同时不要求普通模型改变 forward 签名。

### 5. 推理预测使用 energy 最小原则

模块 4 输出 `energy_matrix: [B, classes]`，预测类别为：

```text
pred = argmin(energy_matrix, dim=1)
```

为了兼容现有二分类指标逻辑，训练/验证流程需要在 `model` 暴露 HPEC prediction/probability 时优先使用该 prediction，而不是对 sigmoid 输出用 `0.5` 阈值。建议增加：

- `model.get_latest_prediction()`
- `model.get_latest_probabilities()`
- 或在 `Exp_Main` 中识别 `latest_hpec_output`

选择原因：HPEC 是 energy classifier，不能简单解释为 sigmoid 概率；推理规则应严格遵守 `argmin(E)`。

### 6. 可视化扩展到 HPEC energy 和原型

显式开启 `visualize_causal=1` 时，现有 train/test heatmap 继续保存模块 2/3 中间量，并新增：

- HPEC prototypes
- HPEC angle matrix
- HPEC aperture/psi
- HPEC energy matrix
- HPEC predicted label
- Ground truth label

最终 epoch t-SNE 默认继续使用 `logmap0(z_global)`，可同时记录 HPEC prediction。t-SNE 图中 train/test 用 marker 区分，label 用颜色区分。

### 7. 参数入口

新增参数建议：

- `use_hpec_module4`: 是否启用模块 4。
- `hpec_prototype_radius`: 原型半径，参考 HPEC `prototypes_ray`，默认 `0.3`。
- `hpec_cone_k`: HPEC 孔径函数参数 `K`，默认 `0.1`。
- `hpec_margin`: negative energy margin，默认 `1.0`。
- `hpec_trainable_prototypes`: 是否训练原型，默认 `0`。
- `hpec_init_steps`: 原型 separation 初始化步数，默认低预算可用 `500` 或 `1000`。
- `hpec_eps`: 数值稳定 clamp。

`use_hpec_module4=1` 应要求 `use_hgcn_module3=1`，因为模块 4 的输入来自模块 3 的 `z_global`。

## Risks / Trade-offs

- [Risk] HPEC 角度公式在点接近原点或边界时数值不稳定 → 对 norm、acos 输入和 asin 输入执行 eps clamp，并继续使用模块 3 Backclip/projx。
- [Risk] 原型训练与普通 Adam/Geoopt 参数混用复杂 → 默认固定原型，后续再开启 trainable prototypes。
- [Risk] 二分类指标路径原本假设 sigmoid 概率 → 为 HPEC 增加 energy-based prediction/probability 获取逻辑，避免错误阈值。
- [Risk] HPEC loss 初期可能较大，压过模块 2 auxiliary loss 或反过来被 auxiliary loss 淹没 → 保留现有 loss 权重打印，并新增 HPEC loss/energy 打印。
- [Risk] `func_angle_vec` reference 代码中距离项形状写法不够直观 → 迁移时按 pairwise `[B, K, D]` 方式重写公式，并用 shape test 验证。

## Migration Plan

1. 新增 `layers/hpec_energy_layer.py`，迁移 HPEC angle、aperture、energy loss、prediction 与原型初始化。
2. 修改 `models/S_DeCI.py`，新增模块 4 初始化、forward 调用、HPEC loss 缓存和 prediction 缓存。
3. 修改 `exp/exp_classification_CV.py`，支持模型自定义 primary/classification loss 与 energy-based prediction。
4. 更新 `run_cv.py`、`test_training_smoke.py`、`test_matai_small_sample.py` 的 HPEC 参数入口。
5. 扩展可视化和 t-SNE，保存 HPEC energy/prototype/预测相关中间量。
6. 新增 `docs/S-DeCI模块4-HPEC实现说明.md`，不修改原始 `docs/新模块设计.md`。
7. 运行 shape、loss、gradient、低预算训练和可视化验证。

回滚方式：设置 `use_hpec_module4=0` 回退到模块 3 线性分类路径；如需彻底回滚，删除 HPEC 层、参数入口和 `S-DeCI` 中模块 4 分支即可。

## Open Questions

- HPEC 原型默认固定是否足够，还是需要首版就支持 trainable prototypes？建议首版固定，保留开关。
- 二分类输出是否需要构造类 1 概率用于 ROC AUC？建议使用 `softmax(-energy)` 得到 `[B, classes]` 概率，再取正类概率。
- HPEC loss 的 negative margin 是否沿用 reference 默认 `gamma=1.0`？建议先沿用，并通过参数暴露。
