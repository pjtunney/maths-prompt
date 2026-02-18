"""CLI entrypoint: run the prompt optimization loop."""

import subprocess
import time

from rich.console import Console
from rich.table import Table

from maths_prompt.config import (
    CLAUDE_MODEL,
    MAX_RETRIES,
    MLX_MODEL_PATH,
    RETRY_DELAY_SECONDS,
    TRAIN_PROBLEM_COUNT,
)
from maths_prompt.runner import load_best_from_logs, run_optimizer
from maths_prompt.test_eval import run_test_eval

console = Console()


def verify_mlx():
    """Check that the MLX model directory exists."""
    if not MLX_MODEL_PATH.exists():
        console.print(f"[red]MLX model not found at {MLX_MODEL_PATH}[/]")
        console.print("[yellow]Run: uv run python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B --mlx-path models/Qwen2.5-0.5B-4bit -q[/]")
        raise SystemExit(1)
    console.print(f"[green]MLX model found at {MLX_MODEL_PATH}[/]")


def print_config():
    """Print current configuration."""
    table = Table(title="maths-prompt configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("MLX model", str(MLX_MODEL_PATH))
    table.add_row("Claude model", CLAUDE_MODEL)
    table.add_row("Training problems per eval", str(TRAIN_PROBLEM_COUNT))
    table.add_row("Retry delay", f"{RETRY_DELAY_SECONDS}s")
    table.add_row("Max retries", str(MAX_RETRIES))
    console.print(table)


def main():
    verify_mlx()
    print_config()

    session = 0
    previous_summary: str | None = None
    for attempt in range(MAX_RETRIES):
        session += 1
        best_prompt, best_score = load_best_from_logs()
        console.print(f"\n[bold]Session {session}[/] | Best so far: {best_score:.1%}")

        try:
            result = run_optimizer(best_prompt, best_score, previous_summary)
        except subprocess.TimeoutExpired:
            console.print("[yellow]Session timed out[/]")
            best_prompt, best_score = load_best_from_logs()
            if best_prompt:
                console.print("[cyan]Running test eval on best prompt so far...[/]")
                test_acc = run_test_eval(best_prompt)
                console.print(f"[bold]Test accuracy: {test_acc:.1%}[/]")
            console.print(f"[yellow]Retrying in {RETRY_DELAY_SECONDS}s...[/]")
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        if result.returncode == 0:
            console.print("[green]Session completed successfully[/]")
            if result.stdout:
                console.print(result.stdout[:2000])
            previous_summary = result.stdout.strip() or None
        else:
            console.print(f"[yellow]Claude exited with code {result.returncode}[/]")
            if result.stderr:
                console.print(f"[dim]{result.stderr[:500]}[/]")
            console.print(f"[yellow]Retrying in {RETRY_DELAY_SECONDS}s...[/]")
            time.sleep(RETRY_DELAY_SECONDS)

        # Run test eval on best prompt
        best_prompt, best_score = load_best_from_logs()
        if best_prompt:
            console.print("[cyan]Running test set evaluation...[/]")
            test_acc = run_test_eval(best_prompt)
            console.print(f"[bold]Training best: {best_score:.1%} | Test accuracy: {test_acc:.1%}[/]")

    console.print("\n[bold green]Done![/]")
