import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"
    r"(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?"
)
CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", re.IGNORECASE)
HIDDEN_MARKER_RE = re.compile(r"Answer\s*:\s*", re.IGNORECASE)


def parse_math_answer(text):
    text = str(text).replace("$", "")
    for marker in ("####", "The answer is:", "Answer:"):
        if marker.lower() in text.lower():
            text = re.split(re.escape(marker), text, flags=re.IGNORECASE)[-1]
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    token = matches[-1].replace(",", "").replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def parse_choice_answer(text):
    text = str(text)
    marker_matches = re.findall(
        r"(?:the\s+answer\s+is|answer)\s*:\s*([A-E])\b", text, re.IGNORECASE
    )
    if marker_matches:
        return marker_matches[-1].upper()
    matches = CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else None


def normalize_fact(text):
    value = " ".join(str(text).strip().split())
    if value and not value.endswith("."):
        value += "."
    return value.casefold()


def parse_hidden_fact(text):
    """Return the single fact in 'Answer: ["..."]', or None if it is not a one-element list."""
    matches = list(HIDDEN_MARKER_RE.finditer(str(text)))
    if not matches:
        return None
    suffix = str(text)[matches[-1].end():].lstrip()
    start = suffix.find("[")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(suffix[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        return None
    return normalize_fact(value[0])


def is_correct(prediction, answers, task):
    if task == "hidden_branch":
        parsed = parse_hidden_fact(prediction)
        return parsed is not None and parsed in {normalize_fact(answer) for answer in answers}
    if task == "commonsenseqa":
        parsed = parse_choice_answer(prediction)
        return parsed is not None and parsed in {str(answer).strip().upper() for answer in answers}
    parsed = parse_math_answer(prediction)
    return parsed is not None and parsed in {parse_math_answer(answer) for answer in answers}


def tv_score(predictions, answers):
    """1 - TV(P, Q): P is the empirical distribution over the gold answers plus an invalid bucket,
    Q is uniform over the gold answers with no invalid mass."""
    gold = sorted({normalize_fact(answer) for answer in answers})
    draws = len(predictions)
    counts = {answer: 0 for answer in gold}
    for prediction in predictions:
        parsed = parse_hidden_fact(prediction)
        if parsed in counts:
            counts[parsed] += 1
    invalid = draws - sum(counts.values())
    distance = invalid / draws + sum(abs(count / draws - 1 / len(gold)) for count in counts.values())
    return 1.0 - 0.5 * distance
