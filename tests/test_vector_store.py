import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from autonomous_rag import vector_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    vector_store.get_vector_store.cache_clear()
    yield vector_store
    vector_store.get_vector_store.cache_clear()


def make_docs(source, count):
    return [
        Document(
            page_content=f"{source} chunk {i}",
            metadata={"source": source, "page": 1, "chunk_id": i},
        )
        for i in range(count)
    ]


def test_add_list_exists_delete(store):
    store.add_documents(make_docs("a.pdf", 3) + make_docs("b.pdf", 2))
    assert store.list_documents() == [
        {"name": "a.pdf", "chunks": 3},
        {"name": "b.pdf", "chunks": 2},
    ]
    assert store.document_exists("a.pdf")
    assert store.delete_document("a.pdf") == 3
    assert not store.document_exists("a.pdf")
    assert store.document_exists("b.pdf")


def test_clear_knowledge_base(store):
    store.add_documents(make_docs("a.pdf", 2))
    assert store.clear_knowledge_base() == 2
    assert store.list_documents() == []


def test_batching_reports_progress(store, monkeypatch):
    monkeypatch.setattr(store, "EMBED_BATCH_SIZE", 2)
    calls = []
    store.add_documents(make_docs("a.pdf", 5), progress_callback=lambda d, t: calls.append((d, t)))
    assert calls == [(2, 5), (4, 5), (5, 5)]


def test_search_returns_scores(store):
    store.add_documents(make_docs("a.pdf", 3))
    results = store.search_with_scores("a.pdf chunk 1", k=2)
    assert len(results) == 2
    assert all(isinstance(distance, float) for _doc, distance in results)