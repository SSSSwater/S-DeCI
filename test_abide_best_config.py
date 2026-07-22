"""ABIDE 最佳配置入口：瘦包装，复用 test_mdd_best_config 的共享解析器与训练主流程。

只在此处预置 ABIDE 的数据集与训练默认值，其余模型/模块超参数全部沿用共享入口
（当前最佳配置）。如需改超参数，直接在命令行追加，例如：
    python test_abide_best_config.py --train-epochs 60
完整可调项见：python test_mdd_best_config.py --help
"""
import sys
from pathlib import Path

from test_mdd_best_config import main as run_best_config


def _prepend_defaults():
    root = Path(__file__).resolve().parent
    defaults = [
        "--data", "Abide",
        "--data-path", str(root / "dataset" / "Abide"),
        "--protocol", "AAL116",
        "--channel", "116",
        "--seq-len", "180",
        "--classes", "2",
        "--kfold", "5",
        "--max-folds", "5",
        "--iterations", "1",
        "--train-epochs", "50",
        "--batch-size", "32",
        "--module1-feature-mode", "alff",
        "--module1-alff-time-weight", "0.5",
        "--use-causal-module2", "1",
        "--causal-learning-target", "temporal_sem",
        "--causal-graph-method", "nts_notears",
        "--classification-graph-source", "blend",
        "--module2-sample-correlation-blend", "0.75",
        "--sample-correlation-mode", "abs",
        "--module1-temporal-dropout", "0.03",
        "--module1-roi-dropout", "0.02",
        "--gcn-fallback-hidden-dim", "32",
        "--gcn-fallback-dropout", "0.3",
        "--gcn-fallback-use-graph-stats", "1",
        "--gcn-fallback-graph-stats-input", "raw",
        "--gcn-fallback-readout-mode", "mean_std",
        "--use-fc-readout-branch", "1",
        "--fc-readout-mode", "both",
        "--fc-readout-dropout", "0.7",
        "--weight-decay", "0.01",
        "--use-hyperbolic-modules34", "1",
        "--use-hgcn-module3", "1",
        "--use-hpec-module4", "1",
        "--module34-branch-ce-loss-weight", "0.2",
        "--hpec-energy-loss-weight", "0.2",
        "--hpec-prototype-ce-loss-weight", "0.1",
        "--hyperbolic-residual-fusion-mode", "dual_consensus",
        "--hyperbolic-logit-residual-weight", "0.5",
        "--causal-vis-dir", "outputs/abide_best_config_causal",
        "--tensorboard-run-name", "abide_best_config",
    ]
    # 预置默认值放在最前，命令行显式参数在后覆盖。
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]


if __name__ == "__main__":
    _prepend_defaults()
    run_best_config()
