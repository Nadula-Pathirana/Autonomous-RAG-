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
    """Token usage for ONE node. Used to see where tokens go."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, message) -> None:
        meta = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += meta.get("input_tokens", 0)
        self.output_tokens += meta.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {"input": self.input_tokens, "output": self.output_tokens}


def _partial_length(buffer: str, tag: str) -> int:
    """How many characters at the end of buffer could be the START of tag?"""
    for size in range(min(len(tag) - 1, len(buffer)), 0, -1):
        if tag.startswith(buffer[-size:]):
            return size
    return 0


class ThinkFilter:
    """Hide <think>...</think> blocks in a stream of text pieces.

    A tag can arrive split across pieces ("<thi" + "nk>"), so we hold back
    anything that might be the start of a tag until we know.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.buffer = ""

    def feed(self, piece: str) -> str:
        self.buffer += piece
        out = ""
        while True:
            if self.inside:
                end = self.buffer.find(self.CLOSE)
                if end == -1:
                    self.buffer = self.buffer[-(len(self.CLOSE) - 1):]
                    return out
                self.buffer = self.buffer[end + len(self.CLOSE):]
                self.inside = False
            else:
                start = self.buffer.find(self.OPEN)
                if start == -1:
                    keep = _partial_length(self.buffer, self.OPEN)
                    cut = len(self.buffer) - keep
                    out += self.buffer[:cut]
                    self.buffer = self.buffer[cut:]
                    return out
                out += self.buffer[:start]
                self.buffer = self.buffer[start + len(self.OPEN):]
                self.inside = True

    def flush(self) -> str:
        """Call at the end of the stream to release anything still held back."""
        out = "" if self.inside else self.buffer
        self.buffer = ""
        return out


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