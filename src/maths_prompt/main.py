"""CLI entrypoint: run the prompt optimization loop."""

import json
import os
import time
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from maths_prompt.config import (
    API_MODEL,
    EVAL_LOG_PATH,
    MAX_RETRIES,
    MAX_SESSIONS,
    MLX_MODEL_PATH,
    RETRY_DELAY_SECONDS,
    SESSION_LOG_PATH,
    TEST_LOG_PATH,
    TRAIN_PROBLEM_COUNT,
)

app = typer.Typer(add_completion=False)
console = Console()


def _verify_mlx():
    """Check that the MLX model directory exists."""
    if not MLX_MODEL_PATH.exists():
        console.print(f"[red]MLX model not found at {MLX_MODEL_PATH}[/]")
        console.print("[yellow]Run: uv run python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B --mlx-path models/Qwen2.5-0.5B-4bit -q[/]")
        raise SystemExit(1)


def _print_config():
    """Print current configuration."""
    table = Table(title="maths-prompt configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("MLX model", str(MLX_MODEL_PATH))
    table.add_row("API model", API_MODEL)
    table.add_row("Training problems per eval", str(TRAIN_PROBLEM_COUNT))
    table.add_row("Max sessions", str(MAX_SESSIONS))
    table.add_row("Max retries on failure", str(MAX_RETRIES))
    table.add_row("Retry delay", f"{RETRY_DELAY_SECONDS}s")
    console.print(table)


def _log_session(session: int, result, test_acc: float | None = None) -> None:
    """Append a session summary line to sessions.jsonl."""
    # Rough cost estimate for claude-sonnet-4-6 ($/1M tokens):
    #   Input: $3, Output: $15, Cache creation: $3.75, Cache read: $0.30
    cost = (
        result.input_tokens * 3.0
        + result.output_tokens * 15.0
        + result.cache_creation_tokens * 3.75
        + result.cache_read_tokens * 0.30
    ) / 1_000_000

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "success": result.success,
        "tool_calls_made": result.tool_calls_made,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "estimated_cost_usd": round(cost, 4),
        "test_accuracy": test_acc,
    }
    SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.command()
def run():
    """Run the prompt optimization loop (Ctrl+C to stop)."""
    _verify_mlx()

    if "MATHS_PROMPT_API_KEY" not in os.environ:
        console.print("[red]MATHS_PROMPT_API_KEY environment variable not set[/]")
        console.print("[yellow]Export your Anthropic API key: export MATHS_PROMPT_API_KEY=sk-ant-...[/]")
        raise SystemExit(1)

    _print_config()

    from maths_prompt.runner import load_best_from_logs, run_optimizer
    from maths_prompt.test_eval import run_test_eval

    consecutive_failures = 0
    previous_summary: str | None = None

    # Resume session numbering from where previous runs left off
    start_session = 1
    if EVAL_LOG_PATH.exists():
        with open(EVAL_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    s = entry.get("session", 0)
                    if s >= start_session:
                        start_session = s + 1
                except json.JSONDecodeError:
                    pass

    try:
        for session in range(start_session, start_session + MAX_SESSIONS):
            best_prompt, best_score = load_best_from_logs()
            console.print(f"\n--- Session {session} | Best so far: {best_score:.1%} ---")

            try:
                result = run_optimizer(best_prompt, best_score, previous_summary, session=session)
                consecutive_failures = 0
            except Exception as e:
                console.print(f"[red]Session failed: {e}[/]")
                consecutive_failures += 1
                if consecutive_failures >= MAX_RETRIES:
                    console.print(f"[red]{MAX_RETRIES} consecutive failures — stopping.[/]")
                    break
                best_prompt, best_score = load_best_from_logs()
                if best_prompt:
                    console.print("Running test eval on best prompt so far...")
                    test_acc = run_test_eval(best_prompt)
                    console.print(f"Test accuracy: {test_acc:.1%}")
                console.print(f"Retrying in {RETRY_DELAY_SECONDS}s ({consecutive_failures}/{MAX_RETRIES} failures)...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            if result.fatal_error:
                console.print("[red]Stopping optimization loop due to fatal error.[/]")
                _log_session(session, result)
                break

            test_acc = None
            best_prompt, best_score = load_best_from_logs()
            if best_prompt:
                console.print("Running test set evaluation...")
                test_acc = run_test_eval(best_prompt)
                console.print(f"Training best: {best_score:.1%} | Test accuracy: {test_acc:.1%}")

            _log_session(session, result, test_acc)

            if result.success:
                console.print(f"Session completed ({result.tool_calls_made} tool calls)")
                if result.summary:
                    console.print(result.summary[:2000])
                previous_summary = result.summary
            else:
                console.print("Session did not complete successfully")
                consecutive_failures += 1
                if consecutive_failures >= MAX_RETRIES:
                    console.print(f"[red]{MAX_RETRIES} consecutive failures — stopping.[/]")
                    break
                console.print(f"Retrying in {RETRY_DELAY_SECONDS}s ({consecutive_failures}/{MAX_RETRIES} failures)...")
                time.sleep(RETRY_DELAY_SECONDS)

        console.print("\n[green]Done![/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        best_prompt, best_score = load_best_from_logs()
        if best_prompt:
            console.print(f"Best training accuracy so far: {best_score:.1%}")
            console.print("Running test eval on best prompt...")
            test_acc = run_test_eval(best_prompt)
            console.print(f"Test accuracy: {test_acc:.1%}")
        else:
            console.print("No evaluations logged yet.")
        console.print("Will resume from best prompt on next run.")


@app.command()
def status():
    """Show latest accuracy from logs."""
    if EVAL_LOG_PATH.exists():
        best_score = 0.0
        count = 0
        with open(EVAL_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                count += 1
                if entry["accuracy"] > best_score:
                    best_score = entry["accuracy"]
        console.print(f"Evaluations logged: {count}")
        console.print(f"Best training accuracy: {best_score:.1%}")
    else:
        console.print("[dim]No evaluation logs yet.[/]")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt")):
    """Wipe all logs and start fresh."""
    if EVAL_LOG_PATH.exists():
        best_score = 0.0
        count = 0
        with open(EVAL_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                count += 1
                if entry["accuracy"] > best_score:
                    best_score = entry["accuracy"]
        console.print(f"Evaluations logged: {count}")
        console.print(f"Best training accuracy: {best_score:.1%}")
    else:
        console.print("[dim]No evaluation logs found.[/]")

    if not yes:
        typer.confirm("Delete all log files and start fresh?", abort=True)

    for path in (EVAL_LOG_PATH, SESSION_LOG_PATH, TEST_LOG_PATH):
        if path.exists():
            path.unlink()

    console.print("[green]Logs deleted. Starting fresh on next run.[/]")
