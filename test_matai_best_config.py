"""Mātai 最佳配置入口：瘦包装，复用 test_mdd_best_config 的共享解析器与训练主流程。

只在此处预置 Mātai 的数据集与训练默认值，其余模型/模块超参数全部沿用共享入口。
完整可调项见：python test_mdd_best_config.py --help
"""
import sys
from pathlib import Path

from test_mdd_best_config import main as run_best_config


def _prepend_defaults():
    root = Path(__file__).resolve().parent
    defaults = [
        "--data", "Mātai",
        "--data-path", str(root / "dataset" / "Mātai"),
        "--protocol", "AAL116",
        "--channel", "116",
        "--seq-len", "200",
        "--classes", "2",
        "--kfold", "5",
        "--max-folds", "5",
        "--iterations", "1",
        "--causal-vis-dir", "outputs/matai_best_config_causal",
        "--tensorboard-run-name", "matai_best_config",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]


if __name__ == "__main__":
    _prepend_defaults()
    run_best_config()
