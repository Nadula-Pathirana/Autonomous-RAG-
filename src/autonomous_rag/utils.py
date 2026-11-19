import json
import re
import time

from config import RETRY_ATTEMPTS, RETRY_BASE_DELAY
from autonomous_rag.exceptions import is_transient
from autonomous_rag.logger import get_logger

logger = get_logger("retry")


def as_text(content) -> str:
    """Message content is usually a string, but can be a list of blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def strip_think(text: str) -> str:
    """Some models (like Qwen) print their reasoning inside <think>...</think>."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Find the first {...} object in a model reply and parse it."""
    match = re.search(r"\{.*\}", strip_think(text), flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the model output")
    return json.loads(match.group(0))


def format_history(history: list[dict], limit: int = 4, chars: int = 300) -> str:
    """Short text version of recent history, for the router and the verifier."""
    lines = [f"{m['role']}: {m['content'][:chars]}" for m in history[-limit:]]
    return "\n".join(lines) or "(none)"


class Usage:
    """Adds up token usage across all LLM calls in one turn (best effort)."""

    def __init__(self) -> None:
        self.total_tokens = 0

    def add(self, message) -> None:
        meta = getattr(message, "usage_metadata", None) or {}
        self.total_tokens += meta.get("total_tokens", 0)


def with_retry(func, *args, **kwargs):
    """Call func(*args, **kwargs), retrying temporary failures with exponential backoff.

    Waits RETRY_BASE_DELAY, then double that, and so on. Errors that are not
    temporary (a wrong API key, bad input) are raised immediately.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            if attempt == RETRY_ATTEMPTS or not is_transient(error):
                raise
            delay = RETRY_BASE_DELAY * 2 ** (attempt - 1)
            logger.warning(
                "Temporary failure (attempt %d/%d), retrying in %ds: %s",
                attempt, RETRY_ATTEMPTS, delay, error,
            )
            time.sleep(delay)