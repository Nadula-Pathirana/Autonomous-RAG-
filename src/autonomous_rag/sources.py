# pyrefly: ignore[missing-import]
from urllib.parse import urlparse


def document_source(filename: str, page, snippet: str) -> dict:
    """A passage retrieved from one of the user's uploaded documents."""
    return {
        "type": "document",
        "title": filename,
        "location": f"page {page}",
        "snippet": snippet[:300],
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
    """Add a 'cited' flag: did the answer actually mention this source?

    Documents count as cited if the file name appears in the answer.
    Web results count as cited if the site's domain appears in the answer.
    """
    lowered = answer.lower()
    for source in sources:
        if source["type"] == "web":
            domain = urlparse(source["location"]).netloc.lower()
            source["cited"] = bool(domain) and domain in lowered
        else:
            source["cited"] = source["title"].lower() in lowered
    return sources
