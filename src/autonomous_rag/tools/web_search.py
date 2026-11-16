from langchain_core.tools import tool
# pyrefly: ignore[missing-import]
from tavily import TavilyClient

from autonomous_rag.sources import web_source
from config import MAX_WEB_RESULTS, TAVILY_API_KEY


@tool(response_format="content_and_artifact")
def search_web(query: str) -> tuple[str, list[dict]]:
    """Search the INTERNET for current or external information.

    Use this tool for recent news, up-to-date facts, prices, public
    information about companies or people, or anything that is unlikely to
    be in the user's uploaded documents. The query must be a complete,
    standalone search phrase that names the topic. Never use pronouns such as
    "it" or "that".
    """
    if not TAVILY_API_KEY:
        return "Web search is unavailable: TAVILY_API_KEY is not set.", []

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=MAX_WEB_RESULTS)
    except Exception as e:
        return f"Web search failed: {e}", []

    results = response.get("results", [])
    if not results:
        return "No web results found.", []

    parts = []
    sources = []
    for r in results:
        title = r.get("title") or "Untitled"
        url = r.get("url", "")
        content = r.get("content", "")
        # The heading is a ready-made markdown link the agent can copy
        parts.append(f"[{title}]({url})\n{content}")
        sources.append(web_source(title, url, content))

    return "\n\n".join(parts), sources