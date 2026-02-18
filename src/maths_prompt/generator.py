import random
from dataclasses import dataclass

TEST_SEED = 42  # Fixed seed — test set is deterministic across runs


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


def _gen_exponent(rng: random.Random) -> Problem:
    base = rng.randint(2, 9)
    exp = rng.randint(2, 5)
    expr = f"{base} ** {exp}"
    return Problem(question=expr, answer=float(base**exp), category="exponents")


def _gen_modulo(rng: random.Random) -> Problem:
    a = rng.randint(10, 999)
    b = rng.randint(2, 30)
    expr = f"{a} % {b}"
    return Problem(question=expr, answer=float(a % b), category="modulo")


def _gen_long_chain(rng: random.Random) -> Problem:
    op = rng.choice(["+", "-", "*"])
    length = rng.randint(5, 10)
    if op == "*":
        terms = [rng.randint(1, 9) for _ in range(length)]
    else:
        terms = [rng.randint(1, 99) for _ in range(length)]
    expr = f" {op} ".join(str(t) for t in terms)
    answer = float(eval(expr))
    return Problem(question=expr, answer=answer, category="long_chain")


def _gen_deeply_nested(rng: random.Random) -> Problem:
    # ((a op b) op (c op d)) op e  — always produces clean answer
    for _ in range(20):  # retry until clean
        a, b, c, d, e = (rng.randint(1, 20) for _ in range(5))
        op1 = rng.choice(["+", "-", "*"])
        op2 = rng.choice(["+", "-", "*"])
        op3 = rng.choice(["+", "-"])
        expr = f"(({a} {op1} {b}) {op3} ({c} {op2} {d})) + {e}"
        try:
            answer = float(eval(expr))
            if -1e6 < answer < 1e6:
                return Problem(question=expr, answer=answer, category="deeply_nested")
        except Exception:
            continue
    # fallback
    return Problem(question="(2 + 3) * (4 - 1)", answer=15.0, category="deeply_nested")


def _gen_negative(rng: random.Random) -> Problem:
    a = rng.randint(1, 50)
    b = rng.randint(1, 50)
    c = rng.randint(1, 50)
    op = rng.choice(["+", "-", "*"])
    expr = f"(-{a}) {op} {b} + {c}"
    answer = float(eval(expr))
    return Problem(question=expr, answer=answer, category="negatives")


def _gen_decimal(rng: random.Random) -> Problem:
    # Use halves/quarters to keep answers exact
    a = rng.randint(1, 20) * 0.5
    b = rng.randint(1, 20) * 0.5
    op = rng.choice(["+", "-", "*"])
    expr = f"{a} {op} {b}"
    answer = float(eval(expr))
    return Problem(question=expr, answer=answer, category="decimals")


def _gen_large_number(rng: random.Random) -> Problem:
    a = rng.randint(1000, 9999)
    b = rng.randint(1000, 9999)
    op = rng.choice(["+", "-"])
    expr = f"{a} {op} {b}"
    answer = float(eval(expr))
    return Problem(question=expr, answer=answer, category="large_numbers")


_TEST_GENERATORS = [
    _gen_exponent,
    _gen_modulo,
    _gen_long_chain,
    _gen_deeply_nested,
    _gen_negative,
    _gen_decimal,
    _gen_large_number,
]


def generate_test_problems(n: int = 1000) -> list[Problem]:
    """Generate n held-out test problems with a fixed seed (deterministic).

    Covers 7 structurally different categories that do not appear in training:
    exponents, modulo, long_chain, deeply_nested, negatives, decimals, large_numbers.
    """
    rng = random.Random(TEST_SEED)
    problems = []
    gen_cycle = [g for g in _TEST_GENERATORS]  # equal weight per category
    for i in range(n):
        gen = gen_cycle[i % len(gen_cycle)]
        problems.append(gen(rng))
    return problems
