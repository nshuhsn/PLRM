# PLRM: Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories

Official implementation of **PLRM**

PLRM replaces explicit chain-of-thought (CoT) tokens with a short sequence of continuous latent thoughts and treats them as random variables. During training, a posterior over all latent steps is computed in a single parallel forward pass over the gold CoT. At inference, the model rolls out a learned prior without any CoT: taking the prior mean gives a deterministic trajectory, and sampling from it gives diverse reasoning trajectories for the same question.

## Repository layout

```
PLRM/
├── plrm/          # model, data, trainer, inference and metrics
├── scripts/       # data preparation, training and evaluation scripts
├── train.py
├── evaluate.py
└── requirements.txt
```

## Installation

```bash
conda create -n plrm python=3.12 -y
conda activate plrm
pip install -r requirements.txt
```

## Data

We train on GSM8K-Aug-NL and evaluate on the GSM8K test set. The following script downloads the sources and writes the JSONL files used in our experiments:

```bash
python scripts/prepare_gsm8k_nl.py --output-dir data/gsm8k_nl
```

Each line has the fields `example_id`, `question`, `cot` and `answer`.

## Training

PLRM is initialized from a backbone (`meta-llama/Llama-3.2-1B-Instruct` in our experiments) that has first been fine-tuned with explicit-CoT supervision on the same training data.

```bash
INIT_MODEL=/path/to/explicit_cot_sft_checkpoint \
NPROC=4 \
bash scripts/train_gsm8k_nl.sh
```

The model with the best validation accuracy is saved to `OUTPUT_DIR/best_val_accuracy/`. Default settings are defined in `plrm/config.py` and `scripts/train_gsm8k_nl.sh`.

## Evaluation

```bash
CHECKPOINT=outputs/plrm_llama1b_gsm8k_nl/best_val_accuracy \
bash scripts/evaluate_gsm8k_nl.sh
```

Predictions are written to `CHECKPOINT/predictions_test.jsonl`. Use `--decode-mode mode` (default) for deterministic decoding, or `--decode-mode sample` with different `--seed` values to obtain diverse reasoning trajectories.
