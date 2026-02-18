"""Spawn sandboxed Claude Code as the prompt optimizer."""

import json
import os
import subprocess

from maths_prompt.config import (
    CLAUDE_MODEL,
    EVAL_LOG_PATH,
    MAX_TURNS,
    MCP_CONFIG_PATH,
    OPTIMIZER_SYSTEM_PROMPT,
    SANDBOX_SETTINGS_PATH,
    SESSION_TIMEOUT_SECONDS,
)


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
    """Build the task prompt for the Claude Code instance."""
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


def run_optimizer(best_prompt: str | None, best_score: float, previous_summary: str | None = None) -> subprocess.CompletedProcess:
    """Launch a sandboxed Claude Code instance to optimize prompts."""
    task = build_task(best_prompt, best_score, previous_summary)

    # Strip CLAUDECODE env var to prevent nested detection
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    system_prompt = OPTIMIZER_SYSTEM_PROMPT
    if best_prompt:
        system_prompt += (
            f"\n\nPrevious best result: {best_score:.1%} accuracy with this prompt:\n"
            f"---\n{best_prompt}\n---\n"
        )

    return subprocess.run(
        [
            "claude",
            "--print",
            "--tools", "",
            "--max-turns", str(MAX_TURNS),
            "--mcp-config", str(MCP_CONFIG_PATH),
            "--strict-mcp-config",
            "--settings", str(SANDBOX_SETTINGS_PATH),
            "--dangerously-skip-permissions",
            "--model", CLAUDE_MODEL,
            "--system-prompt", system_prompt,
            "-p", task,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=SESSION_TIMEOUT_SECONDS,
    )
