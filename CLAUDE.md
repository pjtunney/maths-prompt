# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies
uv run maths-prompt start        # Start the optimization daemon
uv run maths-prompt stop         # Stop the daemon
uv run maths-prompt status       # Check if running + best accuracy
uv run maths-prompt logs         # Tail the optimizer log (Ctrl+C to stop)
uv run pytest                    # Run tests

# Smoke tests
uv run python -c "from maths_prompt.generator import generate_problems; [print(p) for p in generate_problems(5)]"
uv run python -c "from maths_prompt.scorer import extract_number; print(extract_number('The answer is 42.'))"
uv run python -c "from maths_prompt.model import query_model; print(query_model('Solve this.', '2 + 2'))"
uv run python -c "from maths_prompt.model import query_model_batch; print(query_model_batch('Solve this.', ['2 + 2', '3 * 4']))"

# Inspect logs
python -c "import json; [print(f\"{json.loads(l)['iteration']}: {json.loads(l)['accuracy']:.1%}\") for l in open('logs/evaluations.jsonl')]"

# Open the Streamlit dashboard
uv run streamlit run src/maths_prompt/dashboard.py
```

Requires the MLX model to be converted once before first use:
```bash
uv run python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B --mlx-path models/Qwen2.5-0.5B-4bit -q
```

Requires a valid Anthropic API key:
```bash
export MATHS_PROMPT_API_KEY=sk-ant-...
```

## Architecture

This project has two distinct runtime roles:

**1. The runner (`main.py` → `runner.py`)**
Runs as a background daemon (via `maths-prompt start`). Calls the Anthropic API directly using the `anthropic` Python SDK, providing `evaluate_prompt` as a tool. Retries on failure (5-minute backoff, up to 50 retries). After each session completes or fails, it independently evaluates the best-known prompt against the held-out test set (`test_eval.py`). Passes the best prompt, score, and previous session summary forward as context to the next session (session compaction).

**2. The Anthropic API conversation**
The runner sends messages to the Anthropic API with a single tool definition (`evaluate_prompt`). Claude calls this tool in a loop, receiving only the accuracy percentage back. The conversation continues until Claude stops calling tools or the max tool call limit (25) is reached. After the main loop, a final API call asks Claude to summarise what was tried and learned.

**3. The evaluator (`evaluator.py`)**
Called locally by the runner when Claude invokes the `evaluate_prompt` tool. Generates 400 freshly randomised training problems, runs them against mlx-lm (batch inference), scores purely in Python (no LLM calls for scoring), logs everything to `logs/evaluations.jsonl`, and returns only `"Accuracy: X% (n/400 correct)"`.

## Key design invariants

- **Scoring is zero-LLM**: `scorer.py` uses regex + float comparison only. Never add LLM calls to the scoring path.
- **Training problems are always freshly randomised**: `generator.py` is called fresh each `evaluate_prompt` invocation. Do not cache or reuse problems within a session.
- **Test set uses a fixed seed**: `generator.py`'s `generate_test_problems()` produces 1000 problems deterministically (seed 42) across 7 categories (exponents, modulo, long chains, deeply nested, negatives, decimals, large numbers). These problem types do not appear in training. Claude never evaluates on them — only the outer runner does via `test_eval.py`.
- **Information asymmetry is intentional**: The optimizer Claude only ever sees the accuracy percentage. Full details (model responses, per-problem results) are logged but never surfaced to Claude.

## Configuration

All tuneable values are in `config.py`: model names, problem counts, retry settings, log paths, and the optimizer system prompt. Change things there, not inline.

## Log format

`logs/evaluations.jsonl` — one JSON line per `evaluate_prompt` call, with full prompt, all 400 problems, model responses, extracted answers, and per-problem correctness.

`logs/test_results.jsonl` — same schema, written by `test_eval.py` after each session.

`logs/optimizer.log` — stdout/stderr from the background daemon.

All log files are gitignored (only `logs/.gitkeep` is tracked).
