from langchain_core.tools import tool
# pyrefly: ignore[missing-import]
from tavily import TavilyClient

from config import MAX_WEB_RESULTS, TAVILY_API_KEY


@tool
def search_web(query: str) -> str:
    """Search the INTERNET for current or external information.

    Use this tool for recent news, up-to-date facts, prices, public
    information about companies or people, or anything that is unlikely to
    be in the user's uploaded documents. Input should be a concise web search
    query.
    """
    if not TAVILY_API_KEY:
        return "Web search is unavailable: TAVILY_API_KEY is not set."

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=MAX_WEB_RESULTS)
    except Exception as e:
        return f"Web search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No web results found."

    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[WEB-{i}: {r.get('title', 'Untitled')}]\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"{r.get('content', '')}"
        )
    return "\n\n".join(parts)