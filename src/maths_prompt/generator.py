import random
from dataclasses import dataclass


@dataclass
class Problem:
    question: str
    answer: float
    category: str = "arithmetic"


def _random_expr(depth: int = 0, max_depth: int = 2) -> tuple[str, float]:
    """Recursively build a random expression tree. Returns (expr_string, value)."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.4):
        upper = 99 if depth > 0 else 999
        n = random.randint(1, upper)
        return str(n), float(n)

    left_str, left_val = _random_expr(depth + 1, max_depth)
    right_str, right_val = _random_expr(depth + 1, max_depth)
    op = random.choice(["+", "-", "*", "/"])

    # Ensure clean division
    if op == "/":
        if right_val == 0 or left_val % right_val != 0:
            op = random.choice(["+", "-", "*"])

    result = eval(f"{left_val} {op} {right_val}")

    # Bail if result is unreasonable
    if not (-1e9 < result < 1e9):
        return left_str, left_val

    use_brackets = depth > 0 and op in ("+", "-")
    if use_brackets:
        expr = f"({left_str} {op} {right_str})"
    else:
        expr = f"{left_str} {op} {right_str}"

    return expr, float(result)


def generate_problems(n: int = 80) -> list[Problem]:
    """Generate n random arithmetic training problems."""
    problems = []
    for _ in range(n):
        max_depth = random.choice([1, 1, 2, 2, 3])
        expr, answer = _random_expr(max_depth=max_depth)
        problems.append(Problem(question=expr, answer=answer))
    return problems
