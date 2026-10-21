#!/usr/bin/env bash
# Usage: INIT_MODEL=/path/to/cot_sft_checkpoint bash scripts/train.sh {gpt2|llama} {gsm8k|gsm8k_nl|commonsenseqa|hidden_branch} [extra train.py args]
set -euo pipefail

BACKBONE=${1:?backbone: gpt2 or llama}
TASK=${2:?task: gsm8k, gsm8k_nl, commonsenseqa or hidden_branch}
shift 2

INIT_MODEL=${INIT_MODEL:?set INIT_MODEL to an explicit-CoT SFT checkpoint of the backbone}
DATA_DIR=${DATA_DIR:-data/$TASK}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/plrm_${BACKBONE}_${TASK}}
NPROC=${NPROC:-1}
SEED=${SEED:-11}

# Table 6: learning rate, epochs, effective batch, posterior tokens per latent, max input tokens.
case "$BACKBONE:$TASK" in
  gpt2:gsm8k)          LR=1e-3; EPOCHS=40; BATCH=128; PTL=8;  MAX_TOKENS=200 ;;
  gpt2:gsm8k_nl)       LR=1e-3; EPOCHS=40; BATCH=128; PTL=12; MAX_TOKENS=256 ;;
  gpt2:commonsenseqa)  LR=2e-3; EPOCHS=40; BATCH=128; PTL=24; MAX_TOKENS=1000 ;;
  gpt2:hidden_branch)  LR=1e-3; EPOCHS=40; BATCH=128; PTL=6;  MAX_TOKENS=1000 ;;
  llama:gsm8k)         LR=3e-4; EPOCHS=40; BATCH=256; PTL=7;  MAX_TOKENS=200 ;;
  llama:gsm8k_nl)      LR=3e-4; EPOCHS=40; BATCH=256; PTL=14; MAX_TOKENS=256 ;;
  llama:commonsenseqa) LR=3e-4; EPOCHS=10; BATCH=256; PTL=24; MAX_TOKENS=1000 ;;
  *) echo "unsupported setting: $BACKBONE $TASK" >&2; exit 2 ;;
esac
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-$((BATCH / NPROC))}
GRAD_ACCUM=$((BATCH / (PER_DEVICE_BATCH * NPROC)))

if [[ "$TASK" == "hidden_branch" ]]; then
  # Mixture prior with M strategy components (Table 5: M = 1, 2, 3, 4).
  METHOD_ARGS=(--num_prior_components "${NUM_COMPONENTS:-2}" --mode_balance_weight 1.0
               --posterior_mc_samples 1 --max_new_tokens 32)
else
  METHOD_ARGS=(--num_prior_components 1)
fi

DATA_ARGS=(--train_file "$DATA_DIR/train.jsonl")
EVAL_ARGS=(--eval_strategy no)
if [[ -f "$DATA_DIR/val.jsonl" ]]; then
  DATA_ARGS+=(--validation_file "$DATA_DIR/val.jsonl")
  if [[ "$TASK" != "hidden_branch" ]]; then
    EVAL_ARGS=(--eval_strategy epoch --per_device_eval_batch_size $((16 / NPROC)) --prediction_loss_only True)
  fi
fi

python -m torch.distributed.run --standalone --nproc_per_node "$NPROC" train.py \
  --model_name_or_path "$INIT_MODEL" \
  --data_name "$TASK" \
  "${DATA_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_latent 6 \
  --latent_dim 768 \
  --posterior_tokens_per_latent "$PTL" \
  --max_token_num "$MAX_TOKENS" \
  --model_max_length 1024 \
  --kl_factor 3.0 \
  --kl_warmup_ratio 0.8 \
  "${METHOD_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  --save_strategy epoch \
  --save_total_limit 2 \
  --save_safetensors False \
  --logging_steps 10 \
  --report_to none \
  --ddp_find_unused_parameters True \
  "$@"
