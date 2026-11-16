from functools import lru_cache

# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL


@lru_cache(maxsize=4)
def get_llm(model_name: str = GROQ_MODEL) -> ChatGroq:
    """Create a Groq chat model once per model name and reuse it."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=0)

