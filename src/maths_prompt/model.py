import mlx_lm

from maths_prompt.config import MLX_MODEL_PATH

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = mlx_lm.load(str(MLX_MODEL_PATH))
    return _model, _tokenizer


def _format_prompt(tokenizer, system_prompt: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question + " ="},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )



def query_model(system_prompt: str, question: str) -> str:
    """Send a question to the MLX model with the given system prompt."""
    model, tokenizer = _load()
    prompt = _format_prompt(tokenizer, system_prompt, question)
    return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)


def query_model_batch(system_prompt: str, questions: list[str]) -> list[str]:
    """Send multiple questions to the MLX model in a single batch call."""
    model, tokenizer = _load()
    prompts = [
        tokenizer.encode(_format_prompt(tokenizer, system_prompt, q))
        for q in questions
    ]
    response = mlx_lm.batch_generate(model, tokenizer, prompts=prompts, max_tokens=256)
    return response.texts
