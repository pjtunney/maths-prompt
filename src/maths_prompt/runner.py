"""Prompt optimizer using direct Anthropic API calls."""

import json
import os
from dataclasses import dataclass, field

import anthropic

from maths_prompt.config import (
    API_MODEL,
    EVAL_LOG_PATH,
    MAX_TOKENS_PER_TURN,
    MAX_TOOL_CALLS,
    OPTIMIZER_SYSTEM_PROMPT,
)
from maths_prompt.evaluator import evaluate_prompt

EVALUATE_PROMPT_TOOL = {
    "name": "evaluate_prompt",
    "description": (
        "Test a system prompt against 400 randomly-generated math problems. "
        "Returns only the accuracy score. Problems are freshly randomised each call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The system prompt to test.",
            }
        },
        "required": ["prompt"],
    },
}


@dataclass
class OptimizerResult:
    success: bool
    summary: str | None
    tool_calls_made: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    fatal_error: str | None = None  # Set for billing/auth errors — stops retry loop


def load_best_from_logs() -> tuple[str | None, float]:
    """Read evaluation logs and return (best_prompt, best_accuracy)."""
    if not EVAL_LOG_PATH.exists():
        return None, 0.0

    best_prompt = None
    best_score = 0.0

    with open(EVAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["accuracy"] > best_score:
                best_score = entry["accuracy"]
                best_prompt = entry["prompt"]

    return best_prompt, best_score


def build_task(best_prompt: str | None, best_score: float, previous_summary: str | None = None) -> str:
    """Build the task prompt for the optimizer."""
    if best_prompt:
        best_context = (
            f"\n\nPrevious best result: {best_score:.1%} accuracy with this prompt:\n"
            f"---\n{best_prompt}\n---\n"
            f"Try to beat this score. You can start from this prompt or try something completely different."
        )
    else:
        best_context = ""

    if previous_summary:
        summary_context = (
            f"\n\nSummary from the previous session (what was tried and learned):\n"
            f"---\n{previous_summary}\n---\n"
        )
    else:
        summary_context = ""

    return (
        "Optimise the system prompt for the math model. "
        "Use evaluate_prompt() to test prompts and iterate. "
        "Try at least 8-10 different prompt variations. "
        "Focus on maximising accuracy."
        f"{best_context}"
        f"{summary_context}"
    )


def _is_fatal_api_error(e: anthropic.APIStatusError) -> str | None:
    """Return a human-readable message if this is a fatal (non-retriable) API error."""
    msg = str(e).lower()
    if e.status_code == 401 or isinstance(e, anthropic.AuthenticationError):
        return f"API authentication failed — check MATHS_PROMPT_API_KEY. Detail: {e}"
    if e.status_code == 402 or "credit" in msg or "billing" in msg or "balance" in msg:
        return f"API credits exhausted — top up your Anthropic balance to continue. Detail: {e}"
    if e.status_code == 403 and ("credit" in msg or "billing" in msg):
        return f"API billing error — check your Anthropic account. Detail: {e}"
    return None


def run_optimizer(
    best_prompt: str | None,
    best_score: float,
    previous_summary: str | None = None,
    session: int = 1,
) -> OptimizerResult:
    """Run the optimizer using direct Anthropic API calls with prompt caching."""
    client = anthropic.Anthropic(api_key=os.environ["MATHS_PROMPT_API_KEY"])

    task = build_task(best_prompt, best_score, previous_summary)

    system_prompt = OPTIMIZER_SYSTEM_PROMPT
    if best_prompt:
        system_prompt += (
            f"\n\nPrevious best result: {best_score:.1%} accuracy with this prompt:\n"
            f"---\n{best_prompt}\n---\n"
        )

    system_with_cache = [{"type": "text", "text": system_prompt}]

    messages: list = [
        {
            "role": "user",
            "content": [{"type": "text", "text": task}],
        }
    ]

    tool_call_count = 0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0

    def _accumulate_usage(usage) -> None:
        nonlocal input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
        if usage is None:
            return
        input_tokens += getattr(usage, "input_tokens", 0) or 0
        output_tokens += getattr(usage, "output_tokens", 0) or 0
        cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    try:
        while True:
            try:
                response = client.messages.create(
                    model=API_MODEL,
                    max_tokens=MAX_TOKENS_PER_TURN,
                    system=system_with_cache,
                    tools=[EVALUATE_PROMPT_TOOL],
                    messages=messages,
                )
            except anthropic.APIStatusError as e:
                fatal_msg = _is_fatal_api_error(e)
                if fatal_msg:
                    print(f"\n[FATAL] {fatal_msg}", flush=True)
                    return OptimizerResult(
                        success=False,
                        summary=None,
                        tool_calls_made=tool_call_count,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                        fatal_error=fatal_msg,
                    )
                if isinstance(e, anthropic.RateLimitError):
                    print(f"  Rate limited (will retry in 60s): {e}", flush=True)
                    import time
                    time.sleep(60)
                    continue
                raise  # Other API errors propagate to _run_loop

            _accumulate_usage(response.usage)

            # Append assistant response
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call_count += 1
                        print(f"  evaluate_prompt call #{tool_call_count}", flush=True)
                        result_text = evaluate_prompt(block.input["prompt"], session=session)
                        print(f"  -> {result_text}", flush=True)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            }
                        )

                # Cache only the most recent message. First strip any existing
                # cache_control from all prior messages, then add to the new one.
                for msg in messages:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                item.pop("cache_control", None)
                if tool_results:
                    tool_results[-1]["cache_control"] = {"type": "ephemeral"}
                messages.append({"role": "user", "content": tool_results})

                if tool_call_count >= MAX_TOOL_CALLS:
                    print(f"  Reached max tool calls ({MAX_TOOL_CALLS})", flush=True)
                    break
            else:
                break  # end_turn or other stop reason

    except anthropic.AuthenticationError as e:
        fatal_msg = f"API authentication failed — check MATHS_PROMPT_API_KEY. Detail: {e}"
        print(f"\n[FATAL] {fatal_msg}", flush=True)
        return OptimizerResult(
            success=False,
            summary=None,
            tool_calls_made=tool_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            fatal_error=fatal_msg,
        )

    # Print token summary
    print(
        f"  Tokens: {input_tokens:,} in / {output_tokens:,} out"
        f" | Cache: {cache_creation_tokens:,} created / {cache_read_tokens:,} read",
        flush=True,
    )

    # Ask for a session summary
    summary = None
    try:
        messages.append(
            {
                "role": "user",
                "content": "Summarise what you tried and learned in this session. Be concise — this will be passed to the next session as context.",
            }
        )
        summary_response = client.messages.create(
            model=API_MODEL,
            max_tokens=MAX_TOKENS_PER_TURN,
            system=system_with_cache,
            tools=[EVALUATE_PROMPT_TOOL],
            messages=messages,
        )
        _accumulate_usage(summary_response.usage)
        text_blocks = [b.text for b in summary_response.content if b.type == "text"]
        summary = "\n".join(text_blocks) if text_blocks else None
    except Exception as e:
        print(f"  Warning: failed to get summary: {e}", flush=True)

    return OptimizerResult(
        success=True,
        summary=summary,
        tool_calls_made=tool_call_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )
