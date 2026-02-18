"""CLI entrypoint: start/stop/status/logs for the prompt optimization daemon."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from maths_prompt.config import (
    API_MODEL,
    EVAL_LOG_PATH,
    MAX_RETRIES,
    MLX_MODEL_PATH,
    OPTIMIZER_LOG_PATH,
    PID_FILE_PATH,
    RETRY_DELAY_SECONDS,
    SESSION_LOG_PATH,
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
    table.add_row("Retry delay", f"{RETRY_DELAY_SECONDS}s")
    table.add_row("Max retries", str(MAX_RETRIES))
    console.print(table)


def _is_running() -> int | None:
    """Return PID if optimizer is running, else None."""
    if not PID_FILE_PATH.exists():
        return None
    try:
        pid = int(PID_FILE_PATH.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE_PATH.unlink(missing_ok=True)
        return None


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


def _run_loop():
    """The actual optimization loop (runs in the background process)."""
    from maths_prompt.runner import load_best_from_logs, run_optimizer
    from maths_prompt.test_eval import run_test_eval

    session = 0
    previous_summary: str | None = None

    for attempt in range(MAX_RETRIES):
        session += 1
        best_prompt, best_score = load_best_from_logs()
        print(f"\n--- Session {session} | Best so far: {best_score:.1%} ---", flush=True)

        try:
            result = run_optimizer(best_prompt, best_score, previous_summary, session=session)
        except Exception as e:
            print(f"Session failed: {e}", flush=True)
            best_prompt, best_score = load_best_from_logs()
            if best_prompt:
                print("Running test eval on best prompt so far...", flush=True)
                test_acc = run_test_eval(best_prompt)
                print(f"Test accuracy: {test_acc:.1%}", flush=True)
            print(f"Retrying in {RETRY_DELAY_SECONDS}s...", flush=True)
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        if result.fatal_error:
            # Billing/auth errors — no point retrying
            print(f"Stopping optimization loop due to fatal error.", flush=True)
            _log_session(session, result)
            break

        test_acc = None
        best_prompt, best_score = load_best_from_logs()
        if best_prompt:
            print("Running test set evaluation...", flush=True)
            test_acc = run_test_eval(best_prompt)
            print(f"Training best: {best_score:.1%} | Test accuracy: {test_acc:.1%}", flush=True)

        _log_session(session, result, test_acc)

        if result.success:
            print(f"Session completed ({result.tool_calls_made} tool calls)", flush=True)
            if result.summary:
                print(result.summary[:2000], flush=True)
            previous_summary = result.summary
        else:
            print("Session did not complete successfully", flush=True)
            print(f"Retrying in {RETRY_DELAY_SECONDS}s...", flush=True)
            time.sleep(RETRY_DELAY_SECONDS)

    print("\nDone!", flush=True)


@app.command()
def start():
    """Start the optimization loop as a background daemon."""
    _verify_mlx()

    if "MATHS_PROMPT_API_KEY" not in os.environ:
        console.print("[red]MATHS_PROMPT_API_KEY environment variable not set[/]")
        console.print("[yellow]Export your Anthropic API key: export MATHS_PROMPT_API_KEY=sk-ant-...[/]")
        raise SystemExit(1)

    existing = _is_running()
    if existing:
        console.print(f"[yellow]Optimizer already running (PID {existing})[/]")
        raise SystemExit(1)

    _print_config()

    OPTIMIZER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(OPTIMIZER_LOG_PATH, "a")

    proc = subprocess.Popen(
        [sys.executable, "-c", "from maths_prompt.main import _run_loop; _run_loop()"],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    PID_FILE_PATH.write_text(str(proc.pid))
    console.print(f"[green]Started optimizer (PID {proc.pid})[/]")
    console.print(f"[dim]Log: {OPTIMIZER_LOG_PATH}[/]")
    console.print("[dim]Use 'maths-prompt logs' to watch output, 'maths-prompt stop' to kill.[/]")


@app.command()
def stop():
    """Stop the running optimization daemon."""
    pid = _is_running()
    if not pid:
        console.print("[yellow]Optimizer is not running.[/]")
        return

    console.print(f"Stopping optimizer (PID {pid})...")

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # Wait up to 5 seconds for graceful exit
    for _ in range(10):
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except ProcessLookupError:
            break
    else:
        # Still alive — SIGKILL
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    PID_FILE_PATH.unlink(missing_ok=True)
    console.print("[green]Stopped.[/]")


@app.command()
def status():
    """Show whether the optimizer is running and latest accuracy."""
    pid = _is_running()
    if pid:
        console.print(f"[green]Running[/] (PID {pid})")
    else:
        console.print("[yellow]Not running[/]")

    # Show best accuracy from logs
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

    # Tail of log file
    if OPTIMIZER_LOG_PATH.exists():
        console.print(f"\n[bold]Last 10 lines of {OPTIMIZER_LOG_PATH}:[/]")
        lines = OPTIMIZER_LOG_PATH.read_text().splitlines()
        for line in lines[-10:]:
            console.print(f"  {line}")


@app.command()
def logs():
    """Tail the optimizer log in real-time (Ctrl+C to stop watching)."""
    if not OPTIMIZER_LOG_PATH.exists():
        console.print("[yellow]No log file yet. Start the optimizer first.[/]")
        raise SystemExit(1)

    console.print(f"[dim]Tailing {OPTIMIZER_LOG_PATH} (Ctrl+C to stop)...[/]\n")
    try:
        subprocess.run(["tail", "-f", str(OPTIMIZER_LOG_PATH)])
    except KeyboardInterrupt:
        pass
