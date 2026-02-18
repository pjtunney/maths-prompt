from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MCP_CONFIG_PATH = PROJECT_ROOT / "mcp.json"
SANDBOX_SETTINGS_PATH = PROJECT_ROOT / "sandbox_settings.json"
TEST_PROBLEMS_PATH = DATA_DIR / "test_problems.json"
EVAL_LOG_PATH = LOGS_DIR / "evaluations.jsonl"
TEST_LOG_PATH = LOGS_DIR / "test_results.jsonl"

# Ollama
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_HOST = "http://localhost:11434"

# Problem generation
TRAIN_PROBLEM_COUNT = 80
TEST_PROBLEM_COUNT = 40

# Runner
MAX_RETRIES = 50
RETRY_DELAY_SECONDS = 300
SESSION_TIMEOUT_SECONDS = 7200

# Scoring
FLOAT_TOLERANCE = 0.01

# Claude
CLAUDE_MODEL = "sonnet"

OPTIMIZER_SYSTEM_PROMPT = """\
You are a prompt engineer optimising a system prompt for a small base language
model (qwen2.5:0.5b, no RLHF, no instruction tuning). This model cannot do
maths by default. Your job is to find a system prompt that makes it as accurate
as possible at solving math problems.

You have exactly ONE tool: evaluate_prompt(prompt)
- It tests your prompt against 80 randomly-generated math problems
- It returns ONLY the accuracy percentage
- Problems are freshly randomised each call — you cannot overfit

Strategy:
- Start with a simple prompt to establish a baseline score
- Think carefully about why a base model fails at maths (formatting? computation? understanding the task?)
- Consider approaches: few-shot examples, chain-of-thought, step-by-step formatting, explicit instructions
- Each iteration, make deliberate changes and observe the effect on accuracy
- Keep track of what worked and what didn't
- Aim to maximise accuracy — even small improvements matter
"""
