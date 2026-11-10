from functools import lru_cache

# pyrefly: ignore[missing-import]
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from autonomous_rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def add_documents(chunks: list[Document]) -> None:
    """Embed chunks and store them in ChromaDB."""
    if not chunks:
        return
    get_vector_store().add_documents(chunks)


def search(query: str, k: int = TOP_K) -> list[Document]:
    """Return the k chunks most similar to the query."""
    return get_vector_store().similarity_search(query, k=k)