import re

CITATION_OR_LINK = re.compile(r"\[[^\]]*\](?:\([^)]*\))?")      # [file.pdf, p.2]  and  [Title](url)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")                  # "1. " at the start of a line
DOC_CITATION = re.compile(r"\[([^\]\n]+?\.pdf),\s*p\.?\s*(\d+)\]", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    text = CITATION_OR_LINK.sub(" ", text)
    text = LIST_MARKER.sub("", text)
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def grounding_warnings(answer: str, evidence: str, extra_context: str = "") -> list[str]:
    """Cheap checks that the answer sticks to the evidence. Returns a list of warnings."""
    warnings = []

    allowed = _numbers(evidence + " " + extra_context)
    unknown = sorted(n for n in _numbers(answer) if n not in allowed)
    if unknown:
        warnings.append("Numbers not found in the sources: " + ", ".join(unknown[:5]))

    lowered_evidence = evidence.lower()
    for name, page in DOC_CITATION.findall(answer):
        if f"[{name}, p.{page}]".lower() not in lowered_evidence:
            warnings.append(f"Cites {name} p.{page}, which was not retrieved this turn")

    return warnings