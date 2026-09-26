#!/usr/bin/env bash
set -euo pipefail

INIT_MODEL=${INIT_MODEL:?set INIT_MODEL to an explicit-CoT SFT checkpoint of the base model}
DATA_DIR=${DATA_DIR:-data/gsm8k_nl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/plrm_llama1b_gsm8k_nl}
NPROC=${NPROC:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-$((256 / NPROC))}
SEED=${SEED:-11}

torchrun --standalone --nproc_per_node "$NPROC" train.py \
  --model_name_or_path "$INIT_MODEL" \
  --train_file "$DATA_DIR/train.jsonl" \
  --validation_file "$DATA_DIR/val.jsonl" \
  --data_name gsm8k_nl \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --num_train_epochs 10 \
  --learning_rate 3e-4 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --per_device_eval_batch_size $((16 / NPROC)) \
  --gradient_accumulation_steps 1 \
  --num_latent 6 \
  --latent_dim 768 \
  --posterior_tokens_per_latent 14 \
  --posterior_mc_samples 2 \
  --num_prior_components 1 \
  --kl_factor 3.0 \
  --kl_warmup_ratio 0.8 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --save_safetensors False \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --prediction_loss_only True \
  --logging_steps 10 \
  --report_to none \
  --ddp_find_unused_parameters True \
  "$@"
