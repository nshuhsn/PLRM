# PLRM: Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories

Official implementation of **PLRM**.

PLRM replaces explicit chain-of-thought (CoT) tokens with a short sequence of continuous latent thoughts and treats them as random variables. During training, a posterior over all latent steps is computed in a single parallel forward pass over the gold CoT. At inference, the model rolls out a learned prior without any CoT: taking the prior mean gives a deterministic trajectory, and sampling from it gives diverse reasoning trajectories for the same question. A mixture prior with `M` strategy components models problems that admit several distinct reasoning trajectories.

## Repository layout

```
PLRM/
├── plrm/          # model, data, trainer, inference and metrics
├── scripts/       # training and evaluation scripts
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

Download the datasets and place them as JSONL files under `data/`:

```
data/
├── gsm8k/           train.jsonl  val.jsonl  test.jsonl    # GSM8K-Aug, test = GSM8K test
├── gsm8k_nl/        train.jsonl  val.jsonl  test.jsonl    # GSM8K-Aug-NL, test = GSM8K test
├── commonsenseqa/   train.jsonl  [val.jsonl] test.jsonl   # test = CommonsenseQA validation split
├── svamp/           test.jsonl                            # OOD
├── gsm_hard/        test.jsonl                            # OOD
├── multiarith/      test.jsonl                            # OOD
└── hidden_branch/   train.jsonl  val.jsonl  test.jsonl
```

`val.jsonl` is optional. When present, it is used to select the checkpoint saved in `best_val_accuracy/`.

**GSM8K, GSM8K-NL, CommonsenseQA and OOD sets.** One example per line:

```json
{"example_id": "0", "question": "...", "cot": "...", "answer": "18"}
```

- `cot` is only needed for training and validation.
- For CommonsenseQA, `question` must contain the answer choices, and `answer` is the choice letter:

```
Question: ...
Choices:
A: ...
B: ...
```

`python scripts/prepare_gsm8k_nl.py --output-dir data/gsm8k_nl` builds `data/gsm8k_nl/` from the public sources.

**HiddenBranch.** HiddenBranch is built from the open-world abductive setting of ProofWriter. Each input has two to five valid hidden facts, and each hidden fact has its own proof. The dataset is included in this repository:

```bash
unzip data/hidden_branch.zip -d data
```

| Split | Unique inputs | Valid branches |
| --- | ---: | ---: |
| Train | 21,466 | 52,734 |
| Validation | 3,179 | 7,712 |
| Test | 6,128 | 15,101 |

The files use the following format:

- `train.jsonl` has one line per branch (hidden fact), so inputs repeat:

  ```json
  {"example_id": "...-B1", "question": "Task: ...\nFacts: ...\nRules: ...\nQuestion: ...", "cot": "B1: Assume Anne is big. ...", "answer": "Anne is big."}
  ```

- `val.jsonl` and `test.jsonl` have one line per unique input, with every valid hidden fact listed:

  ```json
  {"example_id": "...", "question": "Task: ...", "answers": ["Anne is big.", "Anne is rough."]}
  ```

## Training

PLRM is initialized from the backbone (`openai-community/gpt2` or `meta-llama/Llama-3.2-1B-Instruct`) after it has been fine-tuned with explicit-CoT supervision on the same training data.

```bash
INIT_MODEL=/path/to/explicit_cot_sft_checkpoint NPROC=4 \
bash scripts/train.sh {gpt2|llama} {gsm8k|gsm8k_nl|commonsenseqa}

# HiddenBranch (GPT-2), M strategy components
INIT_MODEL=/path/to/explicit_cot_sft_checkpoint NUM_COMPONENTS=2 \
bash scripts/train.sh gpt2 hidden_branch
```

`scripts/train.sh` sets the learning rate, epochs, batch size and posterior tokens per latent for each backbone and dataset. The model is saved to `OUTPUT_DIR` (default `outputs/plrm_<backbone>_<task>`). If `val.jsonl` exists, the checkpoint with the best validation accuracy (Hit Rate for HiddenBranch) is also saved to `OUTPUT_DIR/best_val_accuracy/`.

## Evaluation

Accuracy and the average number of tokens (latent steps plus generated answer tokens). For `gsm8k_nl`, this also evaluates the three OOD sets:

```bash
CHECKPOINT=outputs/plrm_llama_gsm8k_nl/best_val_accuracy \
bash scripts/evaluate.sh {gsm8k|gsm8k_nl|commonsenseqa}
```

Hit Rate (one greedy rollout) and TV score (15 stochastic prior rollouts per input) on HiddenBranch:

```bash
CHECKPOINT=outputs/plrm_gpt2_hidden_branch/best_val_accuracy \
bash scripts/evaluate_hidden_branch.sh
```

Predictions are written next to the checkpoint as `predictions_<dataset>.jsonl`.
