from functools import lru_cache

# pyrefly: ignore[missing-import]
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_BATCH_SIZE, TOP_K
from autonomous_rag.embeddings import get_embeddings
from autonomous_rag.logger import get_logger
from autonomous_rag.utils import with_retry

logger = get_logger("vector_store")
#This Python file is all about creating a function for creating a vector database locally
#Add documents to that vector database. 
#Creating a function to return top chunks 

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def add_documents(chunks: list[Document], progress_callback=None) -> None:
    """Embed chunks and store them in ChromaDB, one batch at a time.

    progress_callback(done, total) is called after each batch (used for a progress bar).
    """
    if not chunks:
        return

    store = get_vector_store()
    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        with_retry(store.add_documents, batch)   # retries rate limits and timeouts
        if progress_callback:
            progress_callback(min(start + EMBED_BATCH_SIZE, total), total)

    logger.info("Added %d chunks", total)


def search_with_scores(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Return (chunk, distance) pairs. LOWER distance = more similar."""
    return with_retry(get_vector_store().similarity_search_with_score, query, k=k)


def search(query: str, k: int = TOP_K) -> list[Document]:
    """Return the k chunks most similar to the query (without scores)."""
    return [doc for doc, _distance in search_with_scores(query, k=k)]

def list_documents() -> list[dict]:
    """Return [{'name': 'sample.pdf', 'chunks': 12}, ...] for everything stored."""
    data = get_vector_store().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in data["metadatas"] or []:
        name = (metadata or {}).get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "chunks": count} for name, count in sorted(counts.items())]


def document_exists(filename: str) -> bool:
    """Is a document with this file name already in the knowledge base?"""
    data = get_vector_store().get(where={"source": filename}, limit=1)
    return len(data["ids"]) > 0


def delete_document(filename: str) -> int:
    """Delete every chunk of one document. Returns how many chunks were removed."""
    store = get_vector_store()
    ids = store.get(where={"source": filename})["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def clear_knowledge_base() -> int:
    """Delete everything. Returns how many chunks were removed."""
    store = get_vector_store()
    ids = store.get()["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)