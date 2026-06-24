## 1. 共享训练测试工具

- [x] 1.1 梳理当前 `run_cv.py`、`Exp_Main` 和 data provider 默认参数，确定构造最小 experiment 参数 namespace 所需字段。
- [x] 1.2 在新增脚本中定义通用辅助逻辑，包括仓库根目录解析、数据集路径校验、确定性 seed、指标格式化和清晰错误输出。
- [x] 1.3 确保生成的 checkpoints 使用测试专用目录，并默认启用模型权重清理。

## 2. 根目录训练冒烟测试脚本

- [x] 2.1 创建根目录 Python 冒烟测试脚本，通过 `Exp_Main.kf_train()` 运行一次低预算 cross-validation 训练。
- [x] 2.2 添加 CLI 参数，覆盖 dataset、model、protocol、sequence length、channel count、classes、folds、epochs、batch size、learning rate、device usage、workers 和 checkpoint directory。
- [x] 2.3 配置无需 GPU 假设即可本地运行的安全默认值，并在成功时打印 accuracy、precision、recall、macro F1 和 ROC AUC。
- [x] 2.4 验证当所选数据集路径或训练阶段失败时，冒烟测试会以非零退出码退出，并输出清晰错误信息。

## 3. 根目录 Mātai 小样本训练脚本

- [x] 3.1 创建根目录 Python 脚本，默认面向本地 `dataset/Mātai` 数据集，并使用与 Mātai 兼容的参数。
- [x] 3.2 添加 CLI 覆盖项，用于调整运行预算和核心 experiment 设置，同时保留低成本默认配置。
- [x] 3.3 打印简洁验证指标和解析后的训练配置，便于后续变更比较小样本训练行为。
- [x] 3.4 验证当无法解析 `dataset/Mātai` 时，脚本会报告期望数据集路径，并列出可用数据集目录。

## 4. 验证

- [x] 4.1 使用已配置的 `.venv` Python 运行训练冒烟测试，并确认训练完成。
- [x] 4.2 使用已配置的 `.venv` Python 运行 Mātai 小样本训练测试，并确认训练完成。
- [x] 4.3 记录精确运行命令，以及值得关注的运行耗时或指标输出，供后续维护参考。

## 5. 运行记录

- `.\.venv\Scripts\python.exe -m py_compile test_training_smoke.py test_matai_small_sample.py` 通过。
- `.\.venv\Scripts\python.exe test_training_smoke.py` 通过；默认配置为 Taowu + TSMixer + CPU + 2 fold + 1 epoch，指标：accuracy 0.5500，precision 0.5725，recall 0.5500，macro_f1 0.5320，roc_auc 0.6200。
- `.\.venv\Scripts\python.exe test_matai_small_sample.py` 通过；默认配置参考 `scripts\DeCI\Mātai.sh`，为 Mātai + DeCI + CPU + 5 fold + 10 epoch + batch size 16 + layer 2 + d_model 64，指标：accuracy 0.6667，precision 0.6264，recall 0.6343，macro_f1 0.5951，roc_auc 0.6686。
- 缺失数据集路径验证通过：两个脚本均会非零退出，并输出期望路径和可用数据集目录。
