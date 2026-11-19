import time

from langchain_core.tools import tool

from config import MAX_DISTANCE
from autonomous_rag.exceptions import friendly_error
from autonomous_rag.logger import get_logger
from autonomous_rag.sources import document_source
from autonomous_rag.vector_store import search_with_scores

logger = get_logger("tools.kb")


def run_kb_search(query: str) -> tuple[str, list[dict]]:
    """Search the knowledge base. Returns (text for the LLM, structured sources)."""
    started = time.perf_counter()
    try:
        results = search_with_scores(query)
    except Exception as error:
        logger.exception("kb search failed | query=%r", query)
        return f"Knowledge base search failed: {friendly_error(error)}", []

    if not results:
        logger.info("kb | query=%r | no results (is the knowledge base empty?)", query)
        return "The knowledge base returned nothing. It may be empty.", []

    kept = [
        (doc, distance)
        for doc, distance in results
        if MAX_DISTANCE is None or distance <= MAX_DISTANCE
    ]
    logger.info(
        "kb | query=%r | retrieved=%d kept=%d best_distance=%.3f | %.2fs",
        query, len(results), len(kept),
        min(distance for _doc, distance in results),
        time.perf_counter() - started,
    )

    if not kept:
        return (
            "No relevant information found in the uploaded documents "
            "(the closest passages were not similar enough to the question).",
            [],
        )

    parts, sources = [], []
    for doc, distance in kept:
        filename = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{filename}, p.{page}]\n{doc.page_content}")
        sources.append(document_source(filename, page, doc.page_content, distance))

    return "\n\n".join(parts), sources


@tool(response_format="content_and_artifact")
def search_knowledge_base(query: str) -> tuple[str, list[dict]]:
    """Search the user's UPLOADED DOCUMENTS (PDFs) for relevant passages."""
    return run_kb_search(query)