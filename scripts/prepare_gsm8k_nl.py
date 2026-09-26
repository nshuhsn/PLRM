import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset


GSM8K_AUG_NL = ("zen-E/GSM8K-Aug-NL", "e20551f801620cc7dd968edc941126e493fa8202")
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
VALIDATION_URL = (
    "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/"
    "e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt"
)
VALIDATION_SHA256 = "04f9bddfe8ac4846571b1e2b612b289c95770b5012fdb34cc36a597b041afb87"

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"'])?\s+")
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
DISCOURSE_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:So|Then|Therefore|Thus|Next|Finally|Hence|In total|"
    r"The (?:total |remaining |number )|Number of)\b)"
)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def sentence_steps(cot):
    lines = []
    for paragraph in re.split(r"\n+", cot.replace("\r", "\n")):
        paragraph = BULLET_RE.sub("", paragraph).strip()
        if paragraph:
            lines.extend(SENTENCE_BOUNDARY_RE.split(paragraph))
    steps = []
    for line in lines:
        pieces = DISCOURSE_BOUNDARY_RE.split(line) if line.count("=") > 1 else [line]
        steps.extend(step for piece in pieces if (step := clean(piece)))
    return steps


def render_cot(cot):
    steps = [clean(step) for step in sentence_steps(str(cot or "").strip()) if clean(step)]
    return " ".join(f"{step.rstrip('.!?')}." for step in steps)


def drop_last_step(cot):
    return " ".join(". ".join(cot.split(". ")[:-1]).split())


def train_rows():
    name, revision = GSM8K_AUG_NL
    for index, row in enumerate(load_dataset(name, revision=revision, split="train")):
        yield {
            "example_id": f"gsm8k_aug_nl:train:{index}",
            "question": row["question"],
            "cot": drop_last_step(render_cot(row["cot"])),
            "answer": str(row["answer"]),
        }


def validation_rows():
    with urllib.request.urlopen(VALIDATION_URL) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != VALIDATION_SHA256:
        raise ValueError("unexpected checksum for the GSM8K validation file")
    for index, line in enumerate(payload.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        question, rest = line.split("||", 1)
        cot, answer = rest.rsplit("####", 1)
        yield {
            "example_id": f"gsm8k_aug:validation:{index}",
            "question": question,
            "cot": render_cot(cot),
            "answer": answer.strip(),
        }


def test_rows():
    name, revision = GSM8K
    for index, row in enumerate(load_dataset(name, "main", revision=revision, split="test")):
        yield {
            "example_id": f"gsm8k:test:{index}",
            "question": row["question"],
            "cot": "",
            "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
        }


def write(path, rows):
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"{path}: {count} rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/gsm8k_nl"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "train.jsonl", train_rows())
    write(args.output_dir / "val.jsonl", validation_rows())
    write(args.output_dir / "test.jsonl", test_rows())


if __name__ == "__main__":
    main()
