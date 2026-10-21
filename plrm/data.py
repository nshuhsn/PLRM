import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from plrm.model import IGNORE_INDEX


ANSWER_TEMPLATE = "The answer is: {}"
HIDDEN_BRANCH_TEMPLATE = "Answer: {}"


def normalize_question(text):
    return " ".join(str(text).split())


def format_question(text, task):
    # HiddenBranch inputs keep their line structure (Facts / Rules / Question blocks).
    if task == "hidden_branch":
        return str(text)
    return normalize_question(text)


def format_answer(answer, task):
    if task == "hidden_branch":
        return HIDDEN_BRANCH_TEMPLATE.format(json.dumps([str(answer).strip()]))
    return ANSWER_TEMPLATE.format(str(answer).strip())


def gold_answers(row):
    return row["answers"] if "answers" in row else [row["answer"]]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_example(question_ids, cot_ids, answer_ids, model, eos_token_id):
    explanation = cot_ids + [eos_token_id]
    return {
        "encoder_input_ids": torch.tensor(question_ids + [model.bot_id]),
        "posterior_input_ids": torch.tensor(question_ids + cot_ids),
        "posterior_step_idx": torch.tensor(
            [0] * len(question_ids) + list(range(1, len(cot_ids) + 1))
        ),
        "decoder_input_ids": torch.tensor([model.eot_id] + answer_ids),
        "labels": torch.tensor([IGNORE_INDEX] + answer_ids),
        "explain_input_ids": torch.tensor(explanation),
        "explain_labels": torch.tensor(
            explanation if cot_ids else [IGNORE_INDEX] * len(explanation)
        ),
    }


class ReasoningDataset(Dataset):
    def __init__(self, path, tokenizer, model, max_tokens, task):
        self.rows = []
        self.n_source = 0
        self.n_over_length = 0
        eos = tokenizer.eos_token_id
        for example in read_jsonl(path):
            self.n_source += 1
            question = format_question(example["question"], task)
            cot = str(example.get("cot", "")).strip()
            answer = format_answer(example["answer"], task)
            qids = tokenizer(question, add_special_tokens=False)["input_ids"]
            cids = tokenizer(cot, add_special_tokens=False)["input_ids"] if cot else []
            aids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [eos]
            if len(qids) + len(cids) + len(aids) > max_tokens:
                self.n_over_length += 1
                continue
            self.rows.append(build_example(qids, cids, aids, model, eos))
        print(
            f"loaded {len(self.rows)} examples from {path} "
            f"({self.n_over_length} over {max_tokens} tokens skipped)",
            flush=True,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def left_pad(sequences, value):
    reversed_rows = [row.flip(0) for row in sequences]
    return torch.nn.utils.rnn.pad_sequence(
        reversed_rows, batch_first=True, padding_value=value
    ).flip(1)


def right_pad(sequences, value):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=value)


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, rows):
        encoder = left_pad([row["encoder_input_ids"] for row in rows], self.pad_token_id)
        posterior = left_pad([row["posterior_input_ids"] for row in rows], self.pad_token_id)
        explanation = right_pad([row["explain_input_ids"] for row in rows], self.pad_token_id)
        return {
            "encoder_input_ids": encoder,
            "encoder_attention_mask": encoder.ne(self.pad_token_id),
            "posterior_input_ids": posterior,
            "posterior_attention_mask": posterior.ne(self.pad_token_id),
            "posterior_step_idx": left_pad([row["posterior_step_idx"] for row in rows], 0),
            "decoder_input_ids": right_pad(
                [row["decoder_input_ids"] for row in rows], self.pad_token_id
            ),
            "labels": right_pad([row["labels"] for row in rows], IGNORE_INDEX),
            "explain_input_ids": explanation,
            "explain_attention_mask": explanation.ne(self.pad_token_id),
            "explain_labels": right_pad([row["explain_labels"] for row in rows], IGNORE_INDEX),
        }
