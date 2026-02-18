import re

from maths_prompt.config import FLOAT_TOLERANCE


def extract_number(text: str) -> float | None:
    """Extract the last number from model output using regex.

    Handles integers, decimals, negatives, and simple fractions like 1/4.
    Returns the last match (most likely to be the final answer).
    """
    # Match numbers: optional negative, digits, optional decimal, optional fraction
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if not matches:
        return None

    last = matches[-1]
    if "/" in last:
        parts = last.split("/")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(last)
    except ValueError:
        return None


def check_answer(
    extracted: float | None,
    expected: float,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    """Check if extracted answer matches expected within tolerance."""
    if extracted is None:
        return False
    return abs(extracted - expected) <= tolerance
