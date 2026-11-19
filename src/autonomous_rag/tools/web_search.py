import time

from langchain_core.tools import tool
# pyrefly: ignore[missing-import]
from tavily import TavilyClient

from config import MAX_WEB_RESULTS, TAVILY_API_KEY
from autonomous_rag.exceptions import friendly_error
from autonomous_rag.logger import get_logger
from autonomous_rag.sources import web_source
from autonomous_rag.utils import with_retry

logger = get_logger("tools.web")


def run_web_search(query: str) -> tuple[str, list[dict]]:
    """Search the web. Returns (text for the LLM, structured sources)."""
    if not TAVILY_API_KEY:
        return "Web search is unavailable: TAVILY_API_KEY is not set.", []

    started = time.perf_counter()
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = with_retry(client.search, query=query, max_results=MAX_WEB_RESULTS)
    except Exception as error:
        logger.exception("web search failed | query=%r", query)
        return f"Web search failed: {friendly_error(error)}", []

    results = response.get("results", [])
    logger.info("web | query=%r | results=%d | %.2fs", query, len(results), time.perf_counter() - started)
    if not results:
        return "No web results found.", []

    parts, sources = [], []
    for r in results:
        title = r.get("title") or "Untitled"
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"[{title}]({url})\n{content}")
        sources.append(web_source(title, url, content))

    return "\n\n".join(parts), sources


@tool(response_format="content_and_artifact")
def search_web(query: str) -> tuple[str, list[dict]]:
    """Search the INTERNET for current or external information."""
    return run_web_search(query)