class DocumentProcessingError(Exception):
    """A PDF could not be read or processed. The message is safe to show to the user."""


def is_transient(error: Exception) -> bool:
    """Is this a temporary problem that is worth retrying?"""
    text = str(error).lower()
    markers = (
        "429", "rate limit", "rate_limit", "quota", "resource_exhausted",
        "timeout", "timed out", "temporarily", "overloaded",
        "503", "502", "connection",
    )
    return any(marker in text for marker in markers)


def friendly_error(error: Exception) -> str:
    """Turn a technical exception into a message a normal user can act on."""
    text = str(error).lower()

    if "tool_use_failed" in text:
        return "The AI model produced an invalid tool call. Please try rephrasing your question."
    if any(m in text for m in ("too large", "context length", "maximum context", "413")):
        return (
            "The request was too large for the AI model. "
            "Try 'Clear conversation' or ask a shorter question."
        )
    if any(m in text for m in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")):
        return "The AI service is rate-limited right now (free-tier limits). Wait about a minute and try again."
    if any(m in text for m in ("401", "403", "api key", "api_key", "unauthorized", "permission")):
        return "An API key was rejected. Check GROQ_API_KEY, GOOGLE_API_KEY and TAVILY_API_KEY in your .env file."
    if any(m in text for m in ("timeout", "timed out", "connection", "network", "502", "503")):
        return "Couldn't reach the AI service. Check your internet connection and try again."
    return "Something unexpected went wrong. The details were saved to logs/app.log."