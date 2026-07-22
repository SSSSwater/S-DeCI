#!/usr/bin/env bash
# LP-Brain-HPEC performance repair ablation.
# Root cause fixed in models/S_DeCI.py: FC injection was hard-disabled for the LP path
# (the proven FC biomarker never reached the classifier -> raw acc 0.578 / AUC 0.588).
# This script verifies the repair: bare LP -> +FC inject -> +FC-base/LP-residual fusion.
# Results go to result_lp_repair_tmp.xlsx (not the main result.xlsx).
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
COMMON="--use-hyperbolic-modules34 1 --module34-arch lp_brain_hpec \
 --d-model 32 --dropout 0.5 --batch-size 16 --loss bce \
 --use-fc-readout-branch 1 --train-epochs 40 --iterations 1 \
 --use-tensorboard 0 --print-metric-every 0 --result-file result_lp_repair_tmp.xlsx"

echo "=== SMOKE: LP + FC inject, 1 fold / 4 ep (shape/manifold sanity) ==="
$PY test_mdd_best_config.py $COMMON --max-folds 1 --train-epochs 4 --hgcn-fc-inject-weight 1.0 \
  2>&1 | grep -E "Final-epoch Test Avg|Error|Traceback|nan" | head -5

echo "=== L0: LP bare (FC inject=0) -> reproduce ~0.578 baseline ==="
$PY test_mdd_best_config.py $COMMON --hgcn-fc-inject-weight 0.0 \
  2>&1 | grep -E "Final-epoch Test Avg|Error|Traceback"

echo "=== L1: LP + FC inject 1.0 (FC biomarker into LP z_tangent) ==="
$PY test_mdd_best_config.py $COMMON --hgcn-fc-inject-weight 1.0 \
  2>&1 | grep -E "Final-epoch Test Avg|Error|Traceback"

echo "=== L2: LP + FC inject 1.0 + FC-base/LP-residual fusion (resid 0.25) ==="
$PY test_mdd_best_config.py $COMMON --hgcn-fc-inject-weight 1.0 --hyperbolic-logit-residual-weight 0.25 \
  2>&1 | grep -E "Final-epoch Test Avg|Error|Traceback"

echo "=== DONE (compare L0 vs L1 vs L2 vs gcn_fallback+FC baseline 0.711) ==="
