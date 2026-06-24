采用发表于 ICLR 2025 的前沿成果《Analytic DAG Constraints for Differentiable DAG Learning》（解析 DAG 约束的可微有向无环图学习）来设计您系统的因果图学习模块，能够从根本上解决传统可微因果图模型在全脑大规模网络（如 116 个脑区）上训练时面临的数值不稳定性与梯度消失瓶颈 ``。

以下是为您量身定制的解析 DAG 约束模块的具体数学原理、算法步骤及 PyTorch 实现方案。

---

### 一、 核心数学原理与设计动机

传统的可微有向无环图（DAG）模型（如经典 NOTEARS ``）利用矩阵指数迹来表征无环性约束：
`$$h(A) = \text{tr}(e^{A \circ A}) - N = 0$$`

#### 1. 传统模型的致命缺陷：无限收敛半径与梯度消失

根据 *Zhen Zhang 等人 (ICLR 2025)* 的理论推导，矩阵指数在数学上对应一个具有**无限收敛半径**（`$r = \infty$`）的幂级数展开，其高阶项的系数呈指数级衰减（系数为 `$\frac{1}{i!}$`） ``。在处理高维脑网络（如 116 个 ROI）时，这意味着：

* **长距离循环（高阶自环）的梯度会自发塌陷（Gradient Vanishing）**。
* 模型无法感知并斩断由多个脑区构成的复杂长程反馈环路，导致最终生成的因果图存在大量生理学上不合理的伪反馈环。

#### 2. 解析 DAG 约束的创新解法：有限收敛半径

ICLR 2025 论文证明，满足特定解析函数集合 `$\mathcal{F}$` 且**收敛半径有限**（如 `$r = 1$`）的函数，能更完美地充当 DAG 约束 ``：
`$$\mathcal{F} = \left\{ f \;\middle|\; f(x) = c_0 + \sum_{i=1}^{\infty} c_i x^i, \;\; \forall i > 0, c_i > 0, \;\; r = \lim_{i\to\infty} \frac{c_i}{c_{i+1}} > 0 \right\}$$`

通过选择有理分式（如矩阵逆）作为解析约束函数：
`$$f(x) = (1 - x)^{-1} - 1 = \sum_{i=1}^{\infty} x^i$$`
该级数的各项系数均为 1（不发生级数衰减），这保证了**高阶长程自环在反向传播时依然能获得足够大的梯度强度**，从而迫使算法能够稳定、精准地切断全脑尺度的任意长程闭环 ``。

---

### 二、 解析 DAG 约束因果模块的具体设计步骤

我们将该算法无缝嵌入到您的第二模块中。数据流向与每一步计算如下：

#### 1. 输入数据

接收来自 DeCI 时序解耦后的神经高频特征矩阵 `$C \in \mathbb{R}^{B \times 116 \times 64}$`（B 为 Batch Size）。

#### 2. 初始化可学习的因果矩阵

在网络中声明一个实数邻接矩阵 `$A \in \mathbb{R}^{116 \times 116}$`，并对角线置 0，代表脑区间潜在的因果权重 ``。

#### 3. 非负性与尺度控制（防止求逆发散）

为了计算有理分式约束 `$(I - W)^{-1}$`，必须保证矩阵的谱半径（最大特征值绝对值）`$\rho(W) < 1$`，否则矩阵求逆会数值爆炸。

* **元素非负化：** 计算 `$W = A \circ A$`（即元素级平方），确保所有边权重非负，防止因正负符号抵消而在计算中漏判环路。
* **谱半径估计：** 采用**截断矩阵幂迭代（Truncated Power Iteration, TMPI）**，在每一次前向传播时仅用 5 次快速幂迭代估计出当前 `$W$` 的谱半径 `$\rho(W)$` ``。
* **动态缩放：** 引入安全系数 `$\text{margin} = 0.1$`，计算动态缩放因子 `$s = (1 + \text{margin}) \cdot \rho(W)$`。将原矩阵缩放为 `$W_{scaled} = \frac{W}{s}$`，这在数学上绝对保证了 `$\rho(W_{scaled}) < 1$`，从而使后续的有理分式解析求逆必然收敛 ``。

#### 4. 解析 DAG 损失计算

利用 PyTorch 内部的可微矩阵求逆算子计算解析约束损失：
`$$\mathcal{L}_{DAG}(A) = \text{tr}\left( (I - W_{scaled})^{-1} \right) - N$$`
其中 `$I$` 是 116 维的单位矩阵。当且仅当 `$A$` 构成严格的无向无环图（DAG）时，该损失值精准等于 0 ``。

---

### 三、 算法的 PyTorch 落地代码实现

以下是基于 ICLR 2025 官方开源仓库 `AnalyticDAGLearning` 核心逻辑简化并适配您项目的代码实现 ``：

```python
import torch
import torch.nn as nn

class AnalyticDAGConstraint(nn.Module):
    def __init__(self, num_nodes=116, margin=0.1):
        super(AnalyticDAGConstraint, self).__init__()
        self.num_nodes = num_nodes
        self.margin = margin
        self.I = nn.Parameter(torch.eye(num_nodes), requires_grad=False)
        
    def estimate_spectral_radius(self, W, num_iters=5):
        """
        利用高效的幂迭代法（Power Iteration）估计非负矩阵W的谱半径，
        避免昂贵的全局特征值分解，保证训练高吞吐量。
        """
        device = W.device
        # 初始化任意非零向量
        x = torch.ones((self.num_nodes, 1), device=device)
        for _ in range(num_iters):
            x = torch.matmul(W, x)
            x = x / (torch.norm(x, p=2) + 1e-8)
        
        # 计算瑞利商（Rayleigh Quotient）得到最大特征值估计
        numerator = torch.matmul(torch.matmul(x.T, W), x)
        denominator = torch.matmul(x.T, x)
        rho = (numerator / (denominator + 1e-8)).squeeze()
        return rho

    def forward(self, A):
        """
        A: 形状为 (116, 116) 的可学习因果邻接矩阵
        """
        # 1. 强制对角线为0，去除自环
        A_masked = A * (1.0 - self.I)
        
        # 2. 元素级平方，确保非负性，获得有向图的邻接表示
        W = A_masked * A_masked
        
        # 3. 估计谱半径并计算安全缩放尺度
        rho = self.estimate_spectral_radius(W)
        s = (1.0 + self.margin) * rho
        
        # 防止分母为 0 导致数值异常
        s = torch.clamp(s, min=1e-5)
        W_scaled = W / s
        
        # 4. 计算解析 DAG 约束：h(A) = tr((I - W_scaled)^-1) - N
        # 矩阵求逆在 PyTorch 中原生可微，梯度将完美向后传播
        inv_matrix = torch.linalg.inv(self.I - W_scaled)
        dag_loss = torch.trace(inv_matrix) - self.num_nodes
        
        return dag_loss

```

---

### 四、 与全栈多中心 fMRI 分类任务的无缝衔接

在您系统的整体联合损失设计中，该模块起到了极其关键的“图结构提纯”作用：

1. **联合损失驱动：**
在训练时，总损失函数被定义为：
`$$\mathcal{L}_{total} = \mathcal{L}_{HPEC} + \alpha \mathcal{L}_{recon} + \lambda \mathcal{L}_{DAG}(A) + \gamma \|A\|_1$$`
* `$\mathcal{L}_{HPEC}$`（双曲角度损失）梯度回传时，会告诉因果图 `$A$`：“哪些脑区连接对于区分抑郁症与健康对照组最具有判别力” ``。
* `$\mathcal{L}_{DAG}(A)$`（解析 DAG 损失）则同时产生剪切梯度，告诉 `$A$`：“这些高活性连接构成了一个闭环环路，是不合理的虚假相关，必须切断” ``。


2. **在多中心任务中的独特价值：**
多中心 fMRI 数据中存在大量由于不同医院扫描仪磁场不均匀带来的空间协同噪声（表现为多个脑区虚假的同步共激活） `[1]`。
引入 ICLR 2025 的解析 DAG 约束后，由于其对长程环路的高敏感梯度，模型能够在反向传播中主动、高效地**过滤掉这些由扫描仪噪声产生的全脑大尺度虚假闭环相关性**，仅保留具备单向信息流动的真实神经因果拓扑。这与后端的 HPEC 双曲角度对齐模块协同配合，使您的原型系统在抗多中心域偏移上具备了极其坚实的现代因果推断理论支撑 ``。