#!/usr/bin/env bash
# Usage: CHECKPOINT=outputs/plrm_gpt2_hidden_branch/best_val_accuracy bash scripts/evaluate_hidden_branch.sh
# Hit Rate from one greedy rollout, TV score from NUM_SAMPLES stochastic prior rollouts per input.
set -euo pipefail

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a trained PLRM checkpoint}
DATA_DIR=${DATA_DIR:-data/hidden_branch}

python evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --data-file "$DATA_DIR/test.jsonl" \
  --task hidden_branch \
  --decode-mode mode \
  --num-samples "${NUM_SAMPLES:-15}" \
  --seed "${SEED:-42}" \
  --max-new-tokens 32 \
  --output-file "$CHECKPOINT/predictions_hidden_branch.jsonl" \
  "$@"
