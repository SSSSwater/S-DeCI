## 1. 现状确认与接口设计

- [x] 1.1 阅读 `layers/hpec_energy_layer.py`，确认当前单 prototype 的初始化、energy、loss 和 prediction 数据结构
- [x] 1.2 阅读 `models/S_DeCI.py`，确认模块 4 的初始化、forward、loss 缓存和可视化缓存路径
- [x] 1.3 阅读 `exp/exp_classification_CV.py`，确认 HPEC primary loss、aux loss、指标和 t-SNE 的读取方式
- [x] 1.4 新增 `docs/S-DeCI多原型HPEC实现说明.md`，记录多 prototype 参数、loss 构成、可视化和回滚方式，不直接修改原 `docs/新模块设计.md`

## 2. HPEC 多 prototype 核心层

- [x] 2.1 扩展 `layers/hpec_energy_layer.py`，让 prototype 形状支持 `[classes, hpec_prototypes_per_class, hidden_dim]`
- [x] 2.2 保留 `hpec_prototypes_per_class=1` 的单 prototype 兼容路径
- [x] 2.3 计算 `energy_per_proto: [B, classes, K]`、类别级 `energy_matrix: [B, classes]`、prediction 和 probability
- [x] 2.4 使用 soft-min 或等价可微方式将 prototype-level energy 聚合为类别级 energy
- [x] 2.5 缓存 prototype-level energy、prototype similarity 或必要诊断量，供 `S-DeCI` 可视化使用
- [x] 2.6 为多 prototype 初始化、energy 聚合和数值稳定逻辑添加中文注释

## 3. Prototype Loss 实现

- [x] 3.1 在 HPEC 层或相关 helper 中实现 `L_mle`，支持温度参数 `hpec_proto_temperature`
- [x] 3.2 实现 `L_pcl`，区分同类 prototype 与异类 prototype，并避免过强类内坍缩
- [x] 3.3 实现 `L_pal`，让样本靠近真实类别下最匹配的 prototype
- [x] 3.4 暴露未加权的 `hpec_mle_loss`、`hpec_pcl_loss`、`hpec_pal_loss`
- [x] 3.5 暴露加权后的 prototype auxiliary loss，用于加入训练总 loss
- [x] 3.6 确认三个 prototype loss 均可参与 PyTorch autograd 反向传播

## 4. S-DeCI 接入

- [x] 4.1 在 `models/S_DeCI.py` 中读取 `hpec_prototypes_per_class`、`hpec_proto_temperature`、`lambda_hpec_mle`、`lambda_hpec_pcl`、`lambda_hpec_pal`
- [x] 4.2 将多 prototype 参数传入 `HPECPrototypeEnergy`
- [x] 4.3 在 `compute_primary_loss(labels)` 或独立 label-aware 方法中计算并缓存 prototype loss
- [x] 4.4 将 prototype loss 写入 `latest_aux_losses`，供训练循环统一读取
- [x] 4.5 扩展 `visualize_causal_intermediates`，显示多 prototype 和 prototype-level energy
- [x] 4.6 为新增 `S-DeCI` 多 prototype 接入逻辑添加中文注释

## 5. 训练入口与日志

- [x] 5.1 在 `run_cv.py` 中新增多 prototype 参数，help 文本使用中文描述
- [x] 5.2 在 `test_training_smoke.py` 中新增多 prototype CLI 参数并传入 experiment args
- [x] 5.3 在 `test_matai_small_sample.py` 中新增多 prototype CLI 参数并传入 experiment args
- [x] 5.4 扩展 `exp/exp_classification_CV.py`，将 prototype auxiliary loss 加入总 loss
- [x] 5.5 扩展训练日志，打印 `hpec_mle_loss`、`hpec_pcl_loss`、`hpec_pal_loss` 和 prototype auxiliary loss
- [x] 5.6 确认 `lambda_hpec_mle=0`、`lambda_hpec_pcl=0`、`lambda_hpec_pal=0` 时训练日志和总 loss 稳定

## 6. 多 prototype 可视化

- [x] 6.1 扩展最终 epoch t-SNE，将 train/test embedding 和所有 prototype embedding 拼接后一起降维
- [x] 6.2 在 t-SNE 中用 marker 区分 train/test/prototype，并用颜色区分 prototype 所属类别
- [x] 6.3 确认 `hpec_prototypes_per_class=1` 和 `>1` 时 t-SNE 都能生成
- [x] 6.4 确认 heatmap 标题能标明 prototype 的类别数、每类 prototype 数和 hidden 维度

## 7. 验证

- [x] 7.1 运行 `python -m py_compile` 覆盖 `layers/hpec_energy_layer.py`、`models/S_DeCI.py`、`exp/exp_classification_CV.py`、`run_cv.py` 和两个测试脚本
- [x] 7.2 运行 HPEC 多 prototype 最小 shape/loss 验证，确认输出形状和 loss 标量正确
- [x] 7.3 运行低预算训练，设置 `use_hpec_module4=1`、`hpec_prototypes_per_class=2`，确认至少一个 fold 一个 epoch 跑通
- [x] 7.4 运行低预算训练，设置 `hpec_prototypes_per_class=1`，确认单 prototype 回退路径跑通
- [x] 7.5 显式开启 `visualize_causal`，确认 heatmap 和带多 prototype 的 t-SNE 文件生成
- [x] 7.6 运行 `openspec validate add-multi-prototype-hpec-loss`
