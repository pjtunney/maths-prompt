import ollama as ollama_lib

from maths_prompt.config import OLLAMA_MODEL


def query_model(system_prompt: str, question: str, model: str | None = None) -> str:
    """Send a question to the Ollama base model with the given system prompt."""
    model = model or OLLAMA_MODEL
    response = ollama_lib.generate(
        model=model,
        system=system_prompt,
        prompt=question,
    )
    return response["response"]
