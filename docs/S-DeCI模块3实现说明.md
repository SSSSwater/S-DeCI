# S-DeCI 模块 3 实现说明

## 当前实现范围

本次实现将模块 3 接入 `S-DeCI`：模块 1 输出的 Cycle/seasonal feature `C` 作为节点特征，模块 2 学到的 `A_learned` 作为有向图拓扑，经过 Backclip、Poincare Ball 投影、HGCN 图传播和 readout 得到 `z_global`。

当前阶段不实现模块 4，不加入 HPEC 原型、角度能量分类器或额外 prototype loss。分类直接使用 `logmap0(z_global)` 的欧氏切空间表示，通过线性分类头输出。

## 数据流

1. `S-DeCI` backbone 提取每层 seasonal feature。
2. 按 `causal_feature_source` 聚合为 `C: [B, N, d_model]`。
3. 模块 2 根据 `C` 学习共享因果图 `A_learned: [N, N]`，并计算 `C_hat`、DAG loss 和 L1 sparsity loss。
4. 模块 3 接收 `C` 和未 detach 的 `A_learned`，输出：
   - `C_clipped: [B, N, d_model]`
   - `H0: [B, N, d_model]`
   - `H_gcn: [B, N, hgcn_hidden_dim]`
   - `z_global: [B, hgcn_hidden_dim]`
   - `logmap0(z_global): [B, hgcn_hidden_dim]`
5. `hgcn_classifier(logmap0(z_global))` 生成分类输出。

## Loss 构成

启用模块 3 后，训练总 loss 保持为：

```text
Loss_total =
    Loss_cls(z_global, label)
  + alpha * Loss_Recon(C, C_hat)
  + lambda * Loss_DAG(A_learned)
  + gamma * L1(A_learned)
```

训练中不使用 `A_true`、`A_structure_true` 或任何真实因果矩阵监督。分类 loss 会通过 HGCN 中使用的 `A_learned` 回传到模块 2 因果图参数。

## Fréchet Readout 实现方式

`layers/hyperbolic_gcn_layer.py` 中的 `TangentFrechetReadout` 使用可微切空间均值作为当前工程实现：

1. 对 `H_gcn` 执行 `logmap0`。
2. 在节点维度求均值或加权均值。
3. 使用 `expmap0` 投回 Poincare Ball，得到 `z_global`。
4. 再缓存 `logmap0(z_global)` 供分类和可视化使用。

这种方式保留端到端梯度，并且比逐样本黎曼迭代 Fréchet mean 更适合当前 116 节点交叉验证训练。后续如需严格复刻 reference 中的 Differentiable-Frechet-Mean，可在该 readout 内部替换实现。

## 可视化入口

显式设置 `visualize_causal=1` 时，每个 fold 训练结束后会调用 `S-DeCI.visualize_causal_intermediates()` 保存 heatmap。默认不保存图片。

可视化内容包括模块 2 的 `C`、`C_hat`、重构误差、`A_learned`、方向差分矩阵，以及模块 3 的 `C_clipped`、`H0`、`H_gcn`、`z_global`、`logmap0(z_global)` 和归一化邻接矩阵。

3D 张量由 `utils.tensor_visualization.visualize_tensors` 只显示 Batch0 或指定 batch，并在副标题中显示维度提示。
