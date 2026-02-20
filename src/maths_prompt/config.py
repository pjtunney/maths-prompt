from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_LOG_PATH = LOGS_DIR / "evaluations.jsonl"
TEST_LOG_PATH = LOGS_DIR / "test_results.jsonl"
SESSION_LOG_PATH = LOGS_DIR / "sessions.jsonl"

# MLX model (pre-quantized to 4-bit for speed)
MLX_MODEL_PATH = PROJECT_ROOT / "models/Qwen2.5-0.5B-4bit"
MLX_MAX_TOKENS = 128  # reduce to 128 or 64 to speed up generation

# Problem generation
TRAIN_PROBLEM_COUNT = 100
TEST_PROBLEM_COUNT = 100

# Runner
MAX_SESSIONS = 5    # total optimization sessions before stopping
MAX_RETRIES = 10     # retries after a failed session before giving up
RETRY_DELAY_SECONDS = 30
SESSION_TIMEOUT_SECONDS = 7200

# Anthropic API
API_MODEL = "claude-opus-4-6"
MAX_TOOL_CALLS = 40
MAX_TOKENS_PER_TURN = 4096

OPTIMIZER_SYSTEM_PROMPT = """\
You are a prompt engineer optimising a system prompt for a small base language
model (qwen2.5:0.5b, no RLHF, no instruction tuning). This model cannot do
maths by default. Your job is to find a system prompt that makes it as accurate
as possible at solving math problems.

You have exactly ONE tool: evaluate_prompt(prompt)
- It tests your prompt against 100 randomly-generated math problems
- It returns ONLY the accuracy percentage
- Problems are freshly randomised each call — you cannot overfit. However you will be evaluated on a test set of problems you will never get feedback on. This prevents you from overfitting on problem type.
- Statistical note: with 100 problems, the standard error is ~±5 percentage points
  (95% CI ≈ ±10pp), so differences smaller than ~5pp are likely noise

Session structure:
- You have up to 40 evaluate_prompt() calls in this session
- There are up to 5 sessions in total; context you write at the end carries over to the next
- If context from a previous session is shown at the start, build on it rather than repeating explored directions

Strategy:
- Start with a simple prompt to establish a baseline score
- Think carefully about why a base model fails at maths (formatting? computation? understanding the task?)
- Consider approaches: few-shot examples, chain-of-thought, step-by-step formatting, explicit instructions
- Each iteration, make deliberate changes and observe the effect on accuracy
- Keep track of what worked and what didn't
- Only treat improvements larger than ~5pp as meaningful signal
- Aim to maximise accuracy — even small improvements matter
- When you hit a performance ceiling don't be afriad to take a performance loss or get creative.
- Base models have to be prompted very differently than what you are used to - consider using semi-random tokens to trigger emergent behaviour.
"""
