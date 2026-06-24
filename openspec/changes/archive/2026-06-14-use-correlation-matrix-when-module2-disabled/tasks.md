## 1. 数据加载与样本对齐

- [x] 1.1 梳理现有 `data_provider/data_loader_CV.py` 中各数据集的时间序列与 FC 文件加载逻辑
- [x] 1.2 新增按同一 subject/protocol 查找 correlation matrix 的 helper，兼容 `sub-xxx_<protocol>_correlation_matrix.mat` 和 `sub-xxx_xxx_features_sub_correlation_matrix.mat`
- [x] 1.3 为 `.mat` 读取增加 key 选择与清晰错误提示，优先读取 `data`
- [x] 1.4 在需要 sample correlation fallback 时，让 Dataset 缓存并返回 `(x_enc, label, correlation_matrix)`
- [x] 1.5 校验 correlation matrix 形状为 `[N, N]`，缺失或不匹配时给出包含样本路径的错误

## 2. DataLoader 与训练 batch 兼容

- [x] 2.1 修改 `data_provider/data_factory_CV.py` 的 collate 逻辑，使其兼容二元组和三元组 batch
- [x] 2.2 在 `exp/exp_classification_CV.py` 新增统一 batch unpack helper
- [x] 2.3 修改 train、val、可视化 batch 选择和 t-SNE embedding 收集逻辑，将 correlation matrix 移动到正确 device
- [x] 2.4 确保测试集 label 仍只用于 loss/metric/可视化，不作为模型输入

## 3. 模块 3 batch adjacency 支持

- [x] 3.1 修改 `layers/hyperbolic_gcn_layer.py`，使 HGCN 图传播支持 `[N, N]` 和 `[B, N, N]` adjacency
- [x] 3.2 扩展 adjacency 归一化逻辑，支持 batch adjacency 的 self-loop、`row`、`sym`、`none`
- [x] 3.3 新增 sample correlation 负值处理模式，至少支持 `abs`、`positive`、`raw`，默认 `abs`
- [x] 3.4 对 adjacency 中 NaN/Inf 做稳定处理，并对错误 shape 抛出清晰异常

## 4. S-DeCI 回退路径

- [x] 4.1 修改 `models/S_DeCI.py`，让 `forward` 支持可选 `correlation_matrix=None`
- [x] 4.2 当 `use_causal_module2=0` 且 `use_hgcn_module3=1` 时，将 `correlation_matrix` 传入模块 3
- [x] 4.3 模块 2 关闭时不初始化/调用因果学习器，且 `get_aux_loss()` 返回 `None`
- [x] 4.4 模块 2 开启时保持使用 `A_learned`，不得被输入的 correlation matrix 替代
- [x] 4.5 缓存 sample correlation adjacency，并在模块 4 启用时继续支持 HPEC energy loss
- [x] 4.6 为新增回退路径添加中文注释

## 5. 参数入口与文档

- [x] 5.1 在 `run_cv.py` 中新增 sample correlation fallback 和负值处理模式参数，help 使用中文
- [x] 5.2 在 `test_training_smoke.py` 和 `test_matai_small_sample.py` 中新增对应参数
- [x] 5.3 新增实现说明文档，说明模块 2 关闭时如何使用相关矩阵、文件命名要求和回退方式
- [x] 5.4 确认不修改原始 `docs/新模块设计.md` 和 `models/DeCI.py`

## 6. 可视化

- [x] 6.1 扩展 `S-DeCI` 中间量可视化，在模块 2 关闭路径显示 sample correlation adjacency
- [x] 6.2 保持模块 2 开启路径继续显示 `A_learned`、二值邻接和 reconstruction 相关中间量
- [x] 6.3 确保 train/test heatmap 和最终 t-SNE 在三元组 batch 下仍可生成

## 7. 验证

- [x] 7.1 运行数据加载最小验证，确认同一 TS 样本能加载对应 correlation matrix
- [x] 7.2 运行模块 3 最小 shape 验证，覆盖 `[N, N]` 和 `[B, N, N]` adjacency
- [x] 7.3 运行 `S-DeCI` forward/backward 验证，覆盖 `use_causal_module2=0,use_hgcn_module3=1`
- [x] 7.4 运行低预算训练，确认模块 2 关闭时 sample correlation fallback 能跑通并打印指标
- [x] 7.5 显式开启可视化运行低预算 fold，确认 sample correlation heatmap 和 t-SNE 生成
- [x] 7.6 运行 `openspec validate use-correlation-matrix-when-module2-disabled` 和必要的 Python 编译测试

