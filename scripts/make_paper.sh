#!/usr/bin/env bash
# The paper's tables, in-text numbers and figures, from logs/paper and output/paper, into
# output/paper/summary/. Needs scripts/train_*.sh and scripts/eval_*.sh to have run.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
OUT=output/paper/summary

"${PYTHON}" -m src.paper.tables --logs logs/paper --output output/paper --out "${OUT}"
for figure in tradeoff ablation clean_cost; do
  "${PYTHON}" -m "src.paper.${figure}" "${OUT}/fig_${figure}.pdf"
done
echo "Wrote ${OUT}/"
