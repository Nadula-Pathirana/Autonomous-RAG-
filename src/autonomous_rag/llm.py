from functools import lru_cache

# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_FALLBACK_MODEL, GROQ_MODEL
from autonomous_rag.logger import get_logger
from autonomous_rag.utils import as_text, with_retry

logger = get_logger("llm")

# Errors where trying ANOTHER model can help: model missing, overloaded, rate-limited.
# Deliberately NOT included: 401/403 (bad key). A different model won't fix that.
FALLBACK_MARKERS = (
    "tool_use_failed", "model_not_found", "does not exist", "decommissioned",
    "404", "429", "rate limit", "rate_limit", "overloaded",
    "502", "503", "timeout", "timed out",
)


@lru_cache(maxsize=4)
def get_llm(model_name: str = GROQ_MODEL) -> ChatGroq:
    """Create a Groq chat model once per model name and reuse it."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=0)


def models_to_try() -> list[str]:
    """Primary first, then the fallback (skipping blanks and duplicates)."""
    return [m for m in dict.fromkeys([GROQ_MODEL, GROQ_FALLBACK_MODEL]) if m]


def should_fallback(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in FALLBACK_MARKERS)


def invoke_llm(messages: list, usage=None):
    """Call the LLM. If a model fails in a recoverable way, try the next one."""
    last_error = None
    for name in models_to_try():
        try:
            response = with_retry(get_llm(name).invoke, messages)
        except Exception as error:
            if not should_fallback(error):
                raise                      # bad API key etc.: don't hide it
            logger.warning("Model %s failed (%s). Trying the next model.", name, str(error)[:150])
            last_error = error
            continue
        if usage is not None:
            usage.add(response)
        return response
    # pyrefly: ignore[bad-raise]
    raise last_error


def stream_llm(messages: list, usage=None):
    """Yield the answer text piece by piece.

    We can only switch models BEFORE the first piece is shown. Once the user has
    seen text, switching would produce a confusing half-and-half answer.
    """
    last_error = None
    for name in models_to_try():
        started = False
        try:
            for chunk in get_llm(name).stream(messages):
                if usage is not None:
                    usage.add(chunk)
                text = as_text(chunk.content)
                if text:
                    started = True
                    yield text
            return
        except Exception as error:
            if started or not should_fallback(error):
                raise
            logger.warning("Model %s failed (%s). Trying the next model.", name, str(error)[:150])
            last_error = error
    # pyrefly: ignore[bad-raise]
    raise last_error