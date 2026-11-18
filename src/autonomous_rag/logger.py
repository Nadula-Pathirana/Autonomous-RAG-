import logging
import os
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to logs/app.log (set up once)."""
    root = logging.getLogger("autonomous_rag")
    if not root.handlers:
        os.makedirs(LOG_DIR, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(LOG_DIR, "app.log"),
            maxBytes=1_000_000,   # start a new file after ~1 MB
            backupCount=3,        # keep 3 old files
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(LOG_LEVEL)
        root.propagate = False
    return logging.getLogger(f"autonomous_rag.{name}")