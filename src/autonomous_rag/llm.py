from functools import lru_cache

# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from config import (
    ANSWER_REASONING_EFFORT, GROQ_API_KEY, GROQ_FALLBACK_MODEL, GROQ_MODEL,
    ROUTER_MAX_TOKENS, ROUTER_MODEL, ROUTER_REASONING_EFFORT,
)
from autonomous_rag.logger import get_logger
from autonomous_rag.utils import ThinkFilter, as_text, with_retry

logger = get_logger("llm")

# Errors where trying ANOTHER model can help. Deliberately NOT included: 401/403 (bad key).
FALLBACK_MARKERS = (
    "tool_use_failed", "model_not_found", "does not exist", "decommissioned",
    "404", "429", "rate limit", "rate_limit", "overloaded",
    "502", "503", "timeout", "timed out",
)


@lru_cache(maxsize=8)
def get_llm(model_name: str = GROQ_MODEL, max_tokens=None, reasoning_effort=None) -> ChatGroq:
    """Create a Groq chat model once per (model, options) and reuse it."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    options = {}
    if max_tokens:
        options["max_tokens"] = max_tokens
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=0, **options)


def candidates(role: str = "answer") -> list[tuple]:
    """Models to try, in order, as (model, max_tokens, reasoning_effort) tuples."""
    if role == "router":
        chain = [(ROUTER_MODEL or GROQ_MODEL, ROUTER_MAX_TOKENS, ROUTER_REASONING_EFFORT)]
    else:
        chain = [(GROQ_MODEL, None, ANSWER_REASONING_EFFORT)]
    for name, effort in ((GROQ_MODEL, ANSWER_REASONING_EFFORT), (GROQ_FALLBACK_MODEL, None)):
        if name and name not in [c[0] for c in chain]:
            chain.append((name, None, effort))
    return chain


def should_fallback(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in FALLBACK_MARKERS)


def invoke_llm(messages: list, usage=None, role: str = "answer"):
    """Call the LLM. If a model fails in a recoverable way, try the next one."""
    last_error = None
    for candidate in candidates(role):
        try:
            response = with_retry(get_llm(*candidate).invoke, messages)
        except Exception as error:
            if not should_fallback(error):
                raise
            logger.warning("Model %s failed (%s). Trying the next one.", candidate[0], str(error)[:150])
            last_error = error
            continue
        if usage is not None:
            usage.add(response)
        return response
    # pyrefly: ignore[bad-raise]
    raise last_error


def stream_llm(messages: list, usage=None, role: str = "answer"):
    """Yield the answer text piece by piece (with <think> blocks removed).

    We can only switch models BEFORE the first piece is shown.
    """
    last_error = None
    for candidate in candidates(role):
        started = False
        think = ThinkFilter()
        try:
            for chunk in get_llm(*candidate).stream(messages):
                if usage is not None:
                    usage.add(chunk)
                text = think.feed(as_text(chunk.content))
                if text:
                    started = True
                    yield text
            tail = think.flush()
            if tail:
                yield tail
            return
        except Exception as error:
            if started or not should_fallback(error):
                raise
            logger.warning("Model %s failed (%s). Trying the next one.", candidate[0], str(error)[:150])
            last_error = error
    # pyrefly: ignore[bad-raise]
    raise last_error