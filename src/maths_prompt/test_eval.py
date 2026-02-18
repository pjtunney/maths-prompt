"""Evaluate the best prompt against the held-out test set."""

import json
from datetime import datetime, timezone

from maths_prompt.config import TEST_LOG_PATH, TEST_PROBLEMS_PATH
from maths_prompt.model import query_model
from maths_prompt.scorer import check_answer, extract_number


def load_test_problems() -> list[dict]:
    """Load test problems from data/test_problems.json."""
    with open(TEST_PROBLEMS_PATH) as f:
        return json.load(f)


def run_test_eval(prompt: str) -> float:
    """Evaluate prompt against the held-out test set.

    Logs full details to logs/test_results.jsonl.
    Returns accuracy as a float.
    """
    problems = load_test_problems()
    correct = 0
    details = []

    for p in problems:
        response = query_model(prompt, p["question"])
        extracted = extract_number(response)
        is_correct = check_answer(extracted, p["answer"])
        if is_correct:
            correct += 1
        details.append(
            {
                "category": p.get("category", "unknown"),
                "question": p["question"],
                "answer": p["answer"],
                "model_response": response,
                "extracted": extracted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(problems) if problems else 0.0

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "num_problems": len(problems),
        "num_correct": correct,
        "accuracy": accuracy,
        "problems": details,
    }
    TEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return accuracy
