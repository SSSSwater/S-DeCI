## 1. 新模型文件

- [x] 1.1 新增 `models/S_DeCI.py`，以 `models/DeCI.py` 为基础复制必要结构。
- [x] 1.2 在新模型中复用 `layers.DeCI_Layer.DeCI_Block`，不得新建或复制 DeCI block。
- [x] 1.3 保留可选输入归一化、`Variate_Embedding` 和 `deci_blocks` 堆叠逻辑。
- [x] 1.4 在 forward 中执行 Cycle/Drift decomposition，但最终输出只聚合 seasonal/Cycle logits。
- [x] 1.5 保持二分类输出 `[B, 1]` 并 sigmoid，多分类输出 `[B, classes]` logits。

## 2. 模型注册

- [x] 2.1 更新 `models/__init__.py`，导入 `S_DeCI` 模块。
- [x] 2.2 更新 `exp/exp_basic.py`，在 `model_dict` 中注册 `'S-DeCI': S_DeCI`。
- [x] 2.3 确认 `run_cv.py --model S-DeCI` 和训练测试脚本 CLI 能选择新模型。

## 3. 约束检查

- [x] 3.1 确认 `models/DeCI.py` 的主 `Model.forward()` 行为不因该变更而改变。
- [x] 3.2 确认本次不实现模块 2-4，不新增 `geoopt`、DAG loss、Hyperbolic GCN、HPEC 原型或联合损失。
- [x] 3.3 确认新模型没有新增替代性的 DeCI block 类。

## 4. 训练验证

- [x] 4.1 使用 `.venv` Python 编译检查新模型、模型注册文件和实验基础类。
- [x] 4.2 使用低预算训练命令或 `test_training_smoke.py --model S-DeCI` 验证训练跑通。
- [x] 4.3 记录训练命令和指标输出，便于后续 verify/archive。

## 5. 验证记录

- 编译检查：
  - `.\\.venv\\Scripts\\python.exe -m py_compile models\\S_DeCI.py exp\\exp_basic.py`
  - `.\\.venv\\Scripts\\python.exe -m py_compile models\\__init__.py models\\S_DeCI.py exp\\exp_basic.py`
- 模型注册探针：
  - 确认 `Exp_Basic.model_dict` 可通过 `'S-DeCI'` 选择 `models.S_DeCI`。
- 训练跑通命令：
  - `.\\.venv\\Scripts\\python.exe test_training_smoke.py --model S-DeCI --d-model 16 --layer 1 --train-epochs 1 --kfold 2 --batch-size 8 --num-workers 0`
- 训练输出指标：
  - `accuracy: 0.5000`
  - `precision: 0.2500`
  - `recall: 0.5000`
  - `macro_f1: 0.3333`
  - `roc_auc: 0.5250`
