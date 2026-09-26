import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"
    r"(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?"
)
CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", re.IGNORECASE)


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


def is_correct(prediction, answer, task):
    if task == "commonsenseqa":
        parsed = parse_choice_answer(prediction)
        return parsed is not None and parsed == answer
    parsed = parse_math_answer(prediction)
    return parsed is not None and parsed == parse_math_answer(answer)
