"""Evaluate the best prompt against the held-out test set."""

import json
from datetime import datetime, timezone

from maths_prompt.config import TEST_LOG_PATH, TEST_PROBLEM_COUNT, PromptPair
from maths_prompt.generator import generate_test_problems
from maths_prompt.model import query_model_batch
from maths_prompt.scorer import check_answer, extract_number


def run_test_eval(prompt_pair: PromptPair) -> float:
    """Evaluate a prefix/suffix pair against the held-out test set.

    Generates TEST_PROBLEM_COUNT problems deterministically (fixed seed).
    Logs full details to logs/test_results.jsonl.
    Returns accuracy as a float.
    """
    raw_problems = generate_test_problems(n=TEST_PROBLEM_COUNT)
    problems = [{"question": p.question, "answer": p.answer, "category": p.category} for p in raw_problems]
    correct = 0
    details = []

    responses = query_model_batch(
        prompt_pair.problem_prefix,
        [p["question"] for p in problems],
        prompt_pair.answer_prefix,
    )
    for p, response in zip(problems, responses):
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
        "problem_prefix": prompt_pair.problem_prefix,
        "answer_prefix": prompt_pair.answer_prefix,
        "num_problems": len(problems),
        "num_correct": correct,
        "accuracy": accuracy,
        "problems": details,
    }
    TEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return accuracy
