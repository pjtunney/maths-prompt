"""Evaluate a prompt against randomly-generated math problems."""

import json
from datetime import datetime, timezone

from maths_prompt.config import EVAL_LOG_PATH, TRAIN_PROBLEM_COUNT
from maths_prompt.generator import generate_problems
from maths_prompt.model import query_model_batch
from maths_prompt.scorer import check_answer, extract_number

def _load_last_iteration() -> int:
    if not EVAL_LOG_PATH.exists():
        return 0
    last = 0
    with open(EVAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("iteration", 0) > last:
                    last = entry["iteration"]
            except json.JSONDecodeError:
                pass
    return last

_iteration = _load_last_iteration()


def evaluate_prompt(prompt: str, session: int) -> str:
    """Test a system prompt against freshly-randomised math problems.

    Returns only the accuracy score string.
    """
    global _iteration
    _iteration += 1

    problems = generate_problems(n=TRAIN_PROBLEM_COUNT)
    correct = 0
    details = []

    responses = query_model_batch(prompt, [p.question for p in problems])
    for p, response in zip(problems, responses):
        extracted = extract_number(response)
        is_correct = check_answer(extracted, p.answer)
        if is_correct:
            correct += 1
        details.append(
            {
                "category": p.category,
                "question": p.question,
                "answer": p.answer,
                "model_response": response,
                "extracted": extracted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(problems)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": _iteration,
        "session": session,
        "prompt": prompt,
        "num_problems": len(problems),
        "num_correct": correct,
        "accuracy": accuracy,
        "problems": details,
    }
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return f"Accuracy: {accuracy:.1%} ({correct}/{len(problems)} correct)"
