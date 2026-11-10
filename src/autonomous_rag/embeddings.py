from functools import lru_cache

# pyrefly: ignore[missing-import]
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it (loading is slow)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)