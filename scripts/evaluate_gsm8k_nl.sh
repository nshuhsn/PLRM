#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-outputs/plrm_llama1b_gsm8k_nl/best_val_accuracy}
DATA_DIR=${DATA_DIR:-data/gsm8k_nl}

python evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --data-file "$DATA_DIR/test.jsonl" \
  --task gsm8k_nl \
  --decode-mode mode \
  --output-file "$CHECKPOINT/predictions_test.jsonl" \
  "$@"
