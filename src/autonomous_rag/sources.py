# pyrefly: ignore[missing-import]
from urllib.parse import urlparse
import re


def document_source(filename: str, page, snippet: str, score: float | None = None) -> dict:
    """A passage retrieved from one of the user's uploaded documents."""
    return {
        "type": "document",
        "title": filename,
        "location": f"page {page}",
        "snippet": snippet[:300],
        "score": None if score is None else round(float(score), 3),
    }


def web_source(title: str, url: str, snippet: str) -> dict:
    """A result retrieved from the web."""
    return {
        "type": "web",
        "title": title,
        "location": url,
        "snippet": snippet[:300],
    }


def dedupe_sources(sources: list[dict]) -> list[dict]:
    """Remove repeats (the agent may search the same page twice)."""
    seen = set()
    unique = []
    for source in sources:
        key = (source["type"], source["title"], source["location"])
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def mark_cited(sources: list[dict], answer: str) -> list[dict]:
    """Add a 'cited' flag: did the answer actually cite this source?

    Web results: the site's domain appears in the answer.
    Documents: the file name AND the page appear together, e.g. [sample.pdf, p.2].
    """
    lowered = answer.lower()
    for source in sources:
        if source["type"] == "web":
            domain = urlparse(source["location"]).netloc.lower()
            source["cited"] = bool(domain) and domain in lowered
        else:
            title = re.escape(source["title"].lower())
            page = re.escape(source["location"].removeprefix("page ").strip())
            pattern = rf"{title}[^\]\n]{{0,25}}?\bp(?:p|age)?\.?\s*{page}\b"
            source["cited"] = re.search(pattern, lowered) is not None
    return sources
