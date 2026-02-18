"""MCP server exposing evaluate_prompt tool to sandboxed Claude."""

import json
import time
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from maths_prompt.config import EVAL_LOG_PATH, TRAIN_PROBLEM_COUNT
from maths_prompt.generator import generate_problems
from maths_prompt.model import query_model_batch
from maths_prompt.scorer import check_answer, extract_number

mcp = FastMCP("maths-eval")

_iteration = 0
_session = 1


@mcp.tool()
def evaluate_prompt(prompt: str) -> str:
    """Test a system prompt against 400 randomly-generated math problems.

    Returns only the accuracy score. Problems are freshly randomised each call.
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

    # Log everything
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": _iteration,
        "session": _session,
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
