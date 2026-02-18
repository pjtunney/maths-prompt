import math
import re


def extract_number(text: str) -> float | None:
    """Extract the first number from model output.

    Strips whitespace, then returns the first match of an integer,
    decimal, negative, or simple fraction (e.g. 1/4).
    """
    text = text.strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if not matches:
        return None

    first = matches[0]
    if "/" in first:
        parts = first.split("/")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(first)
    except ValueError:
        return None


def _round_sig(x: float, sig: int) -> float:
    """Round x to `sig` significant figures."""
    if x == 0:
        return 0.0
    d = math.floor(math.log10(abs(x)))
    factor = 10 ** (sig - 1 - d)
    return round(x * factor) / factor


def check_answer(extracted: float | None, expected: float) -> bool:
    """Check if extracted matches expected.

    Integers (expected has no fractional part) must match exactly.
    Floats are compared to 3 significant figures.
    """
    if extracted is None:
        return False
    if expected == int(expected):
        return extracted == expected
    return _round_sig(extracted, 3) == _round_sig(expected, 3)
