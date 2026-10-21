#!/usr/bin/env bash
# Usage: CHECKPOINT=outputs/plrm_llama_gsm8k_nl/best_val_accuracy bash scripts/evaluate.sh {gsm8k|gsm8k_nl|commonsenseqa}
# Reports accuracy and mean tokens (latent steps + generated answer tokens) on data/<task>/test.jsonl.
# For gsm8k_nl it also evaluates the OOD sets data/{svamp,gsm_hard,multiarith}/test.jsonl.
set -euo pipefail

TASK=${1:?task: gsm8k, gsm8k_nl or commonsenseqa}
shift
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a trained PLRM checkpoint}
DATA_ROOT=${DATA_ROOT:-data}

datasets=("$TASK")
if [[ "$TASK" == "gsm8k_nl" ]]; then
  datasets+=(svamp gsm_hard multiarith)
fi

for name in "${datasets[@]}"; do
  echo "== $name"
  python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data-file "$DATA_ROOT/$name/test.jsonl" \
    --task "$TASK" \
    --decode-mode mode \
    --output-file "$CHECKPOINT/predictions_$name.jsonl" \
    "$@"
done
