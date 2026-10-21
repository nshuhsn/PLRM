import argparse
import json
from pathlib import Path

import torch

from plrm.config import TASKS
from plrm.data import gold_answers, read_jsonl
from plrm.inference import DECODE_MODES, iter_predictions, load_model
from plrm.metrics import is_correct, tv_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--task", choices=TASKS, help="default: the task the checkpoint was trained on")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--decode-mode", choices=DECODE_MODES, default="mode")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="stochastic prior rollouts per input for the TV score (hidden_branch)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = load_model(args.checkpoint, args.base_model, args.model_revision)
    task = args.task or model.data_name
    if task not in TASKS:
        raise ValueError(f"cannot infer the task from the checkpoint; pass --task ({', '.join(TASKS)})")
    rows = read_jsonl(args.data_file)
    answers = {row["example_id"]: gold_answers(row) for row in rows}
    decode = dict(task=task, max_new_tokens=args.max_new_tokens)

    predictions = {}
    for prediction in iter_predictions(
        model, rows, args.batch_size, decode_mode=args.decode_mode, teacher=args.teacher, **decode
    ):
        prediction["correct"] = is_correct(prediction["prediction"], answers[prediction["example_id"]], task)
        predictions[prediction["example_id"]] = prediction
    for _ in range(args.num_samples):
        for sample in iter_predictions(model, rows, args.batch_size, decode_mode="sample", **decode):
            predictions[sample["example_id"]].setdefault("samples", []).append(sample["prediction"])

    correct = sum(row["correct"] for row in predictions.values())
    total = len(predictions)
    tokens = [row["generated_token_count"] + row["latent_step_count"] for row in predictions.values()]
    result = {
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "task": task,
        "decode_mode": "teacher" if args.teacher else args.decode_mode,
        "n_correct": correct,
        "n_total": total,
        "hit_rate" if task == "hidden_branch" else "accuracy": correct / total,
        "mean_tokens": sum(tokens) / total,
    }
    if args.num_samples and task == "hidden_branch":
        result["num_samples"] = args.num_samples
        result["tv_score"] = sum(
            tv_score(row["samples"], answers[example_id]) for example_id, row in predictions.items()
        ) / total
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as handle:
            for prediction in predictions.values():
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
