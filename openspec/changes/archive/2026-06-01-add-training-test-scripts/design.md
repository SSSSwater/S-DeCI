## Context（背景）

本仓库通过 `run_cv.py`、`Exp_Main` 以及 `data_provider/` 下的数据加载器训练 fMRI 分类模型。现有 shell 脚本面向完整 benchmark 复现实验，包含较长 epoch、多折交叉验证、重定向日志，以及固定的数据集/模型组合和 GPU/server 假设。它们适合复现实验结果，但对于每次代码变更后的快速验证来说过重。

本次变更将引入两个根目录 Python 脚本：一个快速冒烟测试，用来确认训练链路可以端到端执行；另一个小样本数据集测试，重点关注 Mātai 这类数据集的训练表现和指标输出。两个脚本都应使用 `dataset/` 下已经拷贝好的本地数据集、`.venv/` 中配置好的运行环境，以及当前项目源码；`reference/` 暂不参与。

## Goals / Non-Goals（目标 / 非目标）

**Goals（目标）：**
- 提供一个可在每次变更后运行的根目录训练冒烟测试。
- 提供一个根目录 Mātai 小样本训练测试，并输出可参考的验证指标。
- 复用项目现有训练代码，不重复实现模型、dataloader、loss 或 metric 逻辑。
- 通过低 epoch、低 fold、CPU 兼容和临时 checkpoint 等默认值，让运行预算清晰可控。
- 当数据集、导入、模型构建、训练、验证或指标聚合出错时，给出明确失败信息。

**Non-Goals（非目标）：**
- 不复现论文级 benchmark 结果。
- 不修改数据集内容、`reference/` 源码或现有 benchmark shell 脚本。
- 不引入新的测试框架依赖。
- 不替换 `run_cv.py` 或现有 experiment 类。

## Decisions（设计决策）

1. 使用根目录独立 Python 脚本，而不是 pytest 测试。

   原因：需求明确要求在根目录创建两个可重复运行的 `.py` 文件。独立脚本也更符合当前仓库风格，并避免把 pytest 变成强制依赖。

   备选方案：在 `tests/` 下创建 pytest 风格测试。优点是更便于测试发现，但会引入当前项目尚未使用的新约定和依赖预期。

2. 构造 `argparse.Namespace`，直接调用 `Exp_Main.kf_train()`。

   原因：`run_cv.py` 固定了 5 次外层迭代和 CPU affinity 行为，这对日常检查来说太慢，也带有服务器环境假设。直接构造参数可以复用真实训练路径，同时控制 folds、epochs、device、workers、checkpoints 和清理策略。

   备选方案：通过子进程执行 `python run_cv.py`。这样更接近 CLI 真实入口，但当前入口会固定重复训练五次，并基于 GPU index 设置 CPU affinity，难以保持快速稳定。

3. 默认使用 CPU 兼容、低预算配置，并提供 CLI 覆盖项。

   原因：变更后的验证应该能在没有 GPU 的机器上运行，并尽快完成。默认值应使用轻量模型/配置以及很少的 epochs/folds；如果需要更深入检查，可以通过参数提高预算。

   备选方案：默认使用 DeCI benchmark 配置。它更接近论文脚本，但对于快速回归检查来说更慢、更脆弱。

4. 使用本地临时 checkpoint 目录，并默认删除生成的模型权重。

   原因：重复验证不应污染 benchmark 日志，也不应留下大量 checkpoint 文件。项目已有 `del_weight` 支持，因此脚本应隔离生成产物。

   备选方案：写入现有 `checkpoints/` 和 `logs/` 路径。实现更简单，但会让日常测试产物更杂，也更难清理。

## Risks / Trade-offs（风险 / 权衡）

- 小规模运行通过，但长时间 benchmark 仍可能失败 -> 保持脚本可配置，让维护者在发布验证时提高 epochs/folds/model size。
- 直接使用 `Exp_Main` 可能漏掉 `run_cv.py` CLI wrapper 的回归 -> 参数构造尽量贴近 CLI 默认值，同时在脚本命名和输出中明确它覆盖的是训练链路，而不是完整 CLI 行为。
- Mātai 路径或 Unicode 拼写在不同环境中可能不同 -> 脚本应解析本地 `dataset/Mātai` 目录；缺失时输出明确错误，并列出可用数据集目录。
- 很小的 fold 数可能导致指标波动 -> 小样本脚本应把指标作为冒烟/效果信号输出，而不是强制论文级阈值。
