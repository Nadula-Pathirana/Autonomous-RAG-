from langchain_core.tools import tool

from autonomous_rag.vector_store import search


@tool
def search_knowledge_base(query: str) -> str:
    """Search the user's UPLOADED DOCUMENTS (PDFs) for relevant passages.

    Use this tool when the question is about the content of the user's own
    files, company documents, reports, or anything that might be in the
    documents they uploaded. Input should be a focused search query, not a
    full conversation.
    """
    try:
        chunks = search(query)
    except Exception as e:
        return f"Knowledge base search failed: {e}"

    if not chunks:
        return "No relevant information found in the uploaded documents."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[KB-{i}: {source}, page {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)