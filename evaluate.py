import argparse
import json
from pathlib import Path

import torch

from plrm.config import TASKS
from plrm.data import read_jsonl
from plrm.inference import DECODE_MODES, iter_predictions, load_model
from plrm.metrics import is_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--task", choices=TASKS, default="gsm8k_nl")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--decode-mode", choices=DECODE_MODES, default="mode")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = load_model(args.checkpoint, args.base_model, args.model_revision)
    rows = read_jsonl(args.data_file)
    answers = {row["example_id"]: row["answer"] for row in rows}
    correct = 0
    predictions = []
    for prediction in iter_predictions(
        model, rows, args.batch_size, decode_mode=args.decode_mode,
        teacher=args.teacher, max_new_tokens=args.max_new_tokens,
    ):
        prediction["correct"] = is_correct(prediction["prediction"], answers[prediction["example_id"]], args.task)
        correct += prediction["correct"]
        predictions.append(prediction)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "decode_mode": "teacher" if args.teacher else args.decode_mode,
        "n_correct": correct,
        "n_total": len(predictions),
        "accuracy": correct / len(predictions),
    }, indent=2))


if __name__ == "__main__":
    main()
