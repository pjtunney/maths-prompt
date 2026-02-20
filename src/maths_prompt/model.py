import mlx_lm

from maths_prompt.config import MLX_MODEL_PATH, MLX_MAX_TOKENS

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = mlx_lm.load(str(MLX_MODEL_PATH))
    return _model, _tokenizer


def _format_prompt(problem_prefix: str, question: str, answer_prefix: str) -> str:
    return f"{problem_prefix}{question}{answer_prefix}"


def query_model(problem_prefix: str, question: str, answer_prefix: str) -> str:
    """Send a question to the MLX model with the given prefix/suffix."""
    model, tokenizer = _load()
    prompt = _format_prompt(problem_prefix, question, answer_prefix)
    return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=MLX_MAX_TOKENS, verbose=False)


def query_model_batch(problem_prefix: str, questions: list[str], answer_prefix: str) -> list[str]:
    """Send multiple questions to the MLX model in a single batch call."""
    model, tokenizer = _load()
    prompts = [
        tokenizer.encode(_format_prompt(problem_prefix, q, answer_prefix))
        for q in questions
    ]
    response = mlx_lm.batch_generate(model, tokenizer, prompts=prompts, max_tokens=MLX_MAX_TOKENS)
    return response.texts
