from pathlib import Path
from typing import NamedTuple


class PromptPair(NamedTuple):
    """Holds the two strings that frame each math question for the base model."""
    problem_prefix: str
    answer_prefix: str

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
MAX_TOKENS_PER_TURN = 16384
MAX_TOKENS_SESSION_CONTEXT = 32768  # higher limit for end-of-session handover notes

OPTIMIZER_SYSTEM_PROMPT = """\
You are a prompt engineer optimising a **raw text completion model**
(Qwen2.5-0.5B, a base model — no RLHF, no instruction tuning, no chat training).
This model does NOT understand chat templates, system prompts, or instruction
formats. It simply completes the text it is given.

For each math question, the model receives the concatenation:
  {problem_prefix}{question}{answer_prefix}
and then generates a completion. Your job is to find the best problem_prefix and
answer_prefix strings that make the model produce accurate numeric answers.

You have exactly ONE tool: evaluate_prompt(problem_prefix, answer_prefix)
- problem_prefix: text placed BEFORE the question. Can be anything from empty to
  a long multi-line string with instructions, few-shot examples, or formatting.
- answer_prefix: text placed AFTER the question to elicit the answer. Can range
  from a short separator to a multi-line chain-of-thought scaffold.
- Both can be as short or as long as you want — explore the full range
- It tests your prefix/suffix against 100 randomly-generated math problems
- It returns ONLY the accuracy percentage
- Problems are freshly randomised each call — you cannot overfit. However you will
  be evaluated on a test set of problems you will never get feedback on. This
  prevents you from overfitting on problem type.
- Statistical note: with 100 problems, the standard error is ~±5 percentage points
  (95% CI ≈ ±10pp), so differences smaller than ~5pp are likely noise
- You may re-evaluate the same prefix/suffix to reduce statistical uncertainty —
  averaging multiple runs shrinks the error. But prefer exploring new variations
  over re-running unless you need to distinguish two close candidates

Session structure:
- You have up to 40 evaluate_prompt() calls in this session
- There are up to 5 sessions in total; context you write at the end carries over to the next
- If context from a previous session is shown at the start, build on it rather than repeating explored directions

Strategy:
- Start simple to establish a baseline, then explore broadly
- Think carefully about what text patterns a base model has seen during pre-training
  that would make it produce a numeric answer after a math expression
- Try a wide range of approaches and lengths — short separators, long few-shot
  prefixes, textbook-style formatting, chain-of-thought scaffolds, calculator
  notation, or anything else you can think of
- Each iteration, make deliberate changes and observe the effect on accuracy
- Keep track of what worked and what didn't
- Only treat improvements larger than ~5pp as meaningful signal
- Aim to maximise accuracy — even small improvements matter
- When you hit a performance ceiling don't be afraid to take a performance loss or get creative
- Base models complete text — think about what training data looked like
"""
