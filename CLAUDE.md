# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies
uv run maths-prompt              # Run the full optimization loop
uv run pytest                    # Run tests
uv run jupyter notebook notebooks/analysis.ipynb  # Open progress dashboard

# Smoke tests
uv run python -c "from maths_prompt.generator import generate_problems; [print(p) for p in generate_problems(5)]"
uv run python -c "from maths_prompt.scorer import extract_number; print(extract_number('The answer is 42.'))"
uv run python -c "from maths_prompt.model import query_model; print(query_model('Solve this.', '2 + 2'))"

# Inspect logs
python -c "import json; [print(f\"{json.loads(l)['iteration']}: {json.loads(l)['accuracy']:.1%}\") for l in open('logs/evaluations.jsonl')]"
```

Requires Ollama running locally: `ollama pull qwen2.5:0.5b`

## Architecture

This project has two distinct runtime roles that must not be conflated:

**1. The runner (`main.py` → `runner.py`)**
Runs in the outer shell. Spawns a sandboxed Claude Code subprocess in a loop, retrying on failure (5-minute backoff, up to 50 retries). After each session completes or fails, it independently evaluates the best-known prompt against the held-out test set (`test_eval.py`). Passes the best prompt and score forward as context to the next session via the system prompt and task message.

**2. The sandboxed Claude Code instance**
Launched by `runner.py` with `--tools ""` (no built-in tools) and `--strict-mcp-config` so the only available tool is `evaluate_prompt` from our MCP server. It has no file system access, no bash, no web access. Its sole job is to call `evaluate_prompt` in a loop and converge on a good prompt. The `CLAUDECODE` env var is stripped before spawning to prevent nested-instance detection issues.

**3. The MCP server (`mcp_server.py`)**
Bridges the two roles. Receives prompt strings from the sandboxed Claude, runs them against 80 freshly randomised training problems via Ollama, scores purely in Python (no LLM calls for scoring), logs everything to `logs/evaluations.jsonl`, and returns only `"Accuracy: X% (n/80 correct)"` — Claude sees nothing else.

## Key design invariants

- **Scoring is zero-LLM**: `scorer.py` uses regex + float comparison only. Never add LLM calls to the scoring path.
- **Training problems are always freshly randomised**: `generator.py` is called fresh each `evaluate_prompt` invocation. Do not cache or reuse problems within a session.
- **Test set is fixed**: `data/test_problems.json` is committed and never regenerated. It contains structurally different problem types (exponents, modulo, negatives, decimals, large numbers, long chains) that do not appear in training. Claude never evaluates on it — only the outer runner does via `test_eval.py`.
- **Information asymmetry is intentional**: The sandboxed Claude only ever sees the accuracy percentage. Full details (model responses, per-problem results) are logged but never surfaced to Claude.

## Configuration

All tuneable values are in `config.py`: model names, problem counts, retry settings, log paths, and the optimizer system prompt. Change things there, not inline.

## Log format

`logs/evaluations.jsonl` — one JSON line per `evaluate_prompt` call, with full prompt, all 80 problems, model responses, extracted answers, and per-problem correctness.

`logs/test_results.jsonl` — same schema, written by `test_eval.py` after each session.

Both log files are gitignored (only `logs/.gitkeep` is tracked).
