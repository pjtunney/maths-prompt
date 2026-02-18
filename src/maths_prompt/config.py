from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_LOG_PATH = LOGS_DIR / "evaluations.jsonl"
TEST_LOG_PATH = LOGS_DIR / "test_results.jsonl"
SESSION_LOG_PATH = LOGS_DIR / "sessions.jsonl"
PID_FILE_PATH = LOGS_DIR / "optimizer.pid"
OPTIMIZER_LOG_PATH = LOGS_DIR / "optimizer.log"

# MLX model (pre-quantized to 4-bit for speed)
MLX_MODEL_PATH = PROJECT_ROOT / "models/Qwen2.5-0.5B-4bit"

# Problem generation
TRAIN_PROBLEM_COUNT = 400
TEST_PROBLEM_COUNT = 1000

# Runner
MAX_RETRIES = 50
RETRY_DELAY_SECONDS = 300
SESSION_TIMEOUT_SECONDS = 7200

# Anthropic API
API_MODEL = "claude-sonnet-4-6"
MAX_TOOL_CALLS = 25
MAX_TOKENS_PER_TURN = 4096

OPTIMIZER_SYSTEM_PROMPT = """\
You are a prompt engineer optimising a system prompt for a small base language
model (qwen2.5:0.5b, no RLHF, no instruction tuning). This model cannot do
maths by default. Your job is to find a system prompt that makes it as accurate
as possible at solving math problems.

You have exactly ONE tool: evaluate_prompt(prompt)
- It tests your prompt against 400 randomly-generated math problems
- It returns ONLY the accuracy percentage
- Problems are freshly randomised each call — you cannot overfit
- Statistical note: with 400 problems, the standard error is ~±2.5 percentage points
  (95% CI ≈ ±5pp), so differences smaller than ~3pp are likely noise

Strategy:
- Start with a simple prompt to establish a baseline score
- Think carefully about why a base model fails at maths (formatting? computation? understanding the task?)
- Consider approaches: few-shot examples, chain-of-thought, step-by-step formatting, explicit instructions
- Each iteration, make deliberate changes and observe the effect on accuracy
- Keep track of what worked and what didn't
- Only treat improvements larger than ~5pp as meaningful signal
- Aim to maximise accuracy — even small improvements matter
"""
