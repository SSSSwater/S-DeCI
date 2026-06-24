## Why（为什么）

当前项目主要依赖完整 benchmark shell 脚本来验证训练流程，这些脚本运行时间较长，也不适合在每次代码变更后做快速回归检查。现在数据集已经放在 `dataset/` 下，虚拟环境也已配置在 `.venv/` 中，因此需要两个稳定的根目录 Python 测试脚本，用来在后续每次变更后快速确认训练链路仍然可用。

## What Changes（变更内容）

- 新增一个根目录 Python 冒烟测试脚本，用最小训练配置跑通一次训练；当导入、数据集加载、模型构建、checkpoint 处理或指标计算链路出错时，能够清晰失败。
- 新增一个根目录 Python 小样本数据集测试脚本，重点验证 Mātai 这类小样本数据集的训练效果，默认使用较少 epoch/fold，并输出简洁指标。
- 两个脚本都提供 CLI 参数，方便后续变更时用不同模型、数据集、脑区协议或运行预算复用。
- 训练默认参数和数据集配置可参考 `scripts\DeCI` 中现有脚本，尤其是各数据集对应的 `data_path`、`protocol`、`channel`、`seq_len`、`classes`、`loss`、`d_model` 等参数。
- 脚本使用本地路径（`dataset/`、`.venv/` 运行环境假设、临时 checkpoints/logs），不依赖 `reference/` 源码目录。

## Capabilities（能力）

### New Capabilities（新增能力）
- `training-test-scripts`: 根目录 Python 测试脚本，用于可重复的最小训练验证和小样本数据集训练效果检查。

### Modified Capabilities（修改能力）

## Impact（影响）

- 影响文件：仓库根目录新增两个 Python 脚本。
- 影响系统：本地训练工作流、通过 `data_provider` 的数据集加载、通过 `exp.exp_classification_CV` 的实验执行，以及 checkpoint/log 清理行为。
- 依赖：不新增强制依赖，继续使用现有 `requirements.txt` 和已配置的 `.venv/`。
- 回滚方案：删除新增的两个根目录 Python 脚本；现有训练脚本和模型代码无需变更。
