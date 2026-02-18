# maths-prompt

Can iterative prompt engineering teach a raw base LLM to do maths?

A sandboxed Claude Code instance acts as the optimiser. Its only lever is rewriting the system prompt, and its only feedback is an accuracy score on freshly randomised problems each time.

## Design

- **No LLM scoring** — correctness is checked with pure Python arithmetic (regex + float compare)
- **Randomised training problems** (80 per eval) prevent prompt overfitting to specific questions
- **Separate test set** with structurally different problem types — Claude never sees these; they measure generalisation
- **Information asymmetry** — Claude only sees accuracy %. Everything else (prompts, model outputs, per-problem results) is logged locally
- **Free inference** — the model being prompted runs locally via Ollama

## Architecture

```
main.py (runner)
  └─ subprocess ──► sandboxed Claude Code
                        └─ MCP stdio ──► mcp_server.py
                                             ├─ generator.py  (random problems)
                                             ├─ model.py      (Ollama)
                                             └─ scorer.py     (pure Python)
```

Claude has exactly one tool: `evaluate_prompt(prompt) → "Accuracy: 35%"`. It calls this in a loop, iterating on the prompt.

## Model

`qwen2.5:0.5b` — a tiny base model with no RLHF and no instruction tuning. It cannot do maths by default.

## Problem types

**Training** (randomised each eval, Claude evaluates on these):
- Simple: `347 + 589`, `23 * 17`
- Medium: `12 + 34 * 5`, `(48 - 12) / 6`
- Hard: `(15 + 7) * (3 - 1)`, `((8 + 2) * 5) / 10`

**Test** (fixed, Claude never sees these — measures generalisation):
- Exponents: `2 ** 8`
- Modulo: `47 % 7`
- Long chains: `2 + 3 + 5 + 7 + 11 + 13`
- Deeply nested: `((2 + 3) * (4 - 1)) / (6 - 1)`
- Negatives: `(-5) * 3 + 12`
- Decimals: `2.5 * 4 + 1.5`
- Large numbers: `1234 + 5678`

## Setup

```bash
# Install dependencies
uv sync

# Pull the base model
ollama pull qwen2.5:0.5b

# Run the optimizer
uv run maths-prompt
```

Requires [Ollama](https://ollama.com) running locally and a valid `ANTHROPIC_API_KEY`.

## Logs

Every `evaluate_prompt` call is logged to `logs/evaluations.jsonl` with the full prompt, all 80 problems, model responses, and per-problem results. Test set evaluations go to `logs/test_results.jsonl`.

```bash
# Training accuracy over time
python -c "import json; [print(f\"{json.loads(l)['iteration']}: {json.loads(l)['accuracy']:.1%}\") for l in open('logs/evaluations.jsonl')]"

# Open the analysis notebook
uv run jupyter notebook notebooks/analysis.ipynb
```

## Project structure

```
maths-prompt/
├── src/maths_prompt/
│   ├── config.py          # Settings: model, paths, counts
│   ├── generator.py       # Random arithmetic expression generator
│   ├── model.py           # Ollama interface
│   ├── scorer.py          # Regex extraction + float comparison
│   ├── mcp_server.py      # FastMCP server (the only tool Claude can use)
│   ├── runner.py          # Sandboxed Claude Code launcher
│   ├── test_eval.py       # Held-out test set evaluator
│   └── main.py            # CLI entrypoint with retry loop
├── data/test_problems.json  # Fixed test set (39 problems, 7 categories)
├── logs/                    # JSONL logs (gitignored except .gitkeep)
├── notebooks/analysis.ipynb # Progress dashboard
├── mcp.json                 # MCP server config
└── sandbox_settings.json    # Claude Code sandbox settings
```
