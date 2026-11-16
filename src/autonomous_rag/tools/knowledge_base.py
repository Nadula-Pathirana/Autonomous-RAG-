from langchain_core.tools import tool

from autonomous_rag.sources import document_source
from autonomous_rag.vector_store import search


@tool(response_format="content_and_artifact")
def search_knowledge_base(query: str) -> tuple[str, list[dict]]:
    """Search the user's UPLOADED DOCUMENTS (PDFs) for relevant passages.

    Use this tool when the question is about the content of the user's own
    files, company documents, reports, or anything that might be in the
    documents they uploaded. The query must be a complete, standalone search
    phrase that names the topic (for example "AetherTech price match
    guarantee exclusions"). Never use pronouns such as "it" or "that".
    """
    try:
        chunks = search(query)
    except Exception as e:
        return f"Knowledge base search failed: {e}", []

    if not chunks:
        return "No relevant information found in the uploaded documents.", []

    parts = []
    sources = []
    for chunk in chunks:
        filename = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        # The heading is exactly the citation format the agent should copy
        parts.append(f"[{filename}, p.{page}]\n{chunk.page_content}")
        sources.append(document_source(filename, page, chunk.page_content))

    return "\n\n".join(parts), sources