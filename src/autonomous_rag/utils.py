import time

from config import RETRY_ATTEMPTS, RETRY_BASE_DELAY
from autonomous_rag.exceptions import is_transient
from autonomous_rag.logger import get_logger

logger = get_logger("retry")


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