import time
from dataclasses import dataclass, field
from datetime import date

from config import ENABLE_VERIFICATION, MAX_EVIDENCE_CHARS
from autonomous_rag.llm import invoke_llm, stream_llm
from autonomous_rag.logger import get_logger
from autonomous_rag.prompts import REWRITE_PROMPT, SYSTEM_PROMPT, VERIFY_PROMPT
from autonomous_rag.router import RouteDecision, route_question
from autonomous_rag.sources import dedupe_sources, mark_cited
from autonomous_rag.tools.knowledge_base import run_kb_search
from autonomous_rag.tools.web_search import run_web_search
from autonomous_rag.utils import Usage, as_text, extract_json, format_history, strip_think

logger = get_logger("agent")

TOOL_KB = "search_knowledge_base"   # names the UI already knows
TOOL_WEB = "search_web"


@dataclass
class AgentResult:
    """Everything the UI and the eval need from one run."""

    answer: str
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    route: str = ""
    tokens: int = 0


# ---------- retrieval (done by code, not by the LLM) ----------

def _plan(decision: RouteDecision) -> list[tuple]:
    """Which searches to run: [(tool_name, query, function), ...]."""
    plan = []
    if decision.route in ("KB_ONLY", "BOTH"):
        plan.append((TOOL_KB, decision.kb_query, run_kb_search))
    if decision.route in ("WEB_ONLY", "BOTH"):
        plan.append((TOOL_WEB, decision.web_query, run_web_search))
    return plan


def _execute(tool_name, query, func, steps, sources, blocks) -> None:
    steps.append({"type": "call", "tool": tool_name, "input": {"query": query}})
    text, found = func(query)
    steps.append({"type": "result", "tool": tool_name, "output": text})
    sources.extend(found)
    blocks.append((tool_name, text))


def _evidence_text(blocks: list[tuple]) -> str:
    parts = []
    for tool_name, text in blocks:
        title = (
            "DOCUMENT RESULTS (the user's uploaded files)"
            if tool_name == TOOL_KB
            else "WEB RESULTS (the internet)"
        )
        parts.append(f"=== {title} ===\n{text[:MAX_EVIDENCE_CHARS]}")
    return "\n\n".join(parts)


# ---------- answering ----------

def _build_messages(question: str, history: list[dict], evidence: str) -> list[dict]:
    system = SYSTEM_PROMPT.format(today=date.today().isoformat())
    user = f"{evidence}\n\n=== QUESTION ===\n{question}" if evidence else question
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


def _verify(answer: str, evidence: str, history: list[dict], usage: Usage) -> list[str]:
    """Return the claims in the answer that the evidence does NOT support."""
    prompt = VERIFY_PROMPT.format(
        evidence=evidence, history=format_history(history, limit=6, chars=500), answer=answer
    )
    try:
        response = invoke_llm([{"role": "user", "content": prompt}], usage)
        claims = extract_json(as_text(response.content)).get("unsupported_claims") or []
    except Exception:
        logger.warning("verification failed; keeping the draft answer", exc_info=True)
        return []
    return [str(claim) for claim in claims if str(claim).strip()]


def _finalize(draft: str, evidence: str, history: list[dict], usage: Usage) -> str:
    """Check the draft against the evidence and fix it if needed."""
    if not (ENABLE_VERIFICATION and evidence):
        return draft
    claims = _verify(draft, evidence, history, usage)
    if not claims:
        return draft

    logger.warning("unsupported claims removed: %s", claims)
    prompt = REWRITE_PROMPT.format(
        evidence=evidence, answer=draft, claims="\n".join(f"- {c}" for c in claims)
    )
    try:
        response = invoke_llm([{"role": "user", "content": prompt}], usage)
        return strip_think(as_text(response.content)) or draft
    except Exception:
        logger.warning("rewrite failed; keeping the draft answer", exc_info=True)
        return draft


def _result(answer, steps, sources, decision, usage, started) -> AgentResult:
    sources = mark_cited(dedupe_sources(sources), answer)
    logger.info(
        "turn done | route=%s | tools=%s | sources=%d | tokens=%d | %.2fs",
        decision.route, [s["tool"] for s in steps if s["type"] == "call"],
        len(sources), usage.total_tokens, time.perf_counter() - started,
    )
    return AgentResult(
        answer=answer, steps=steps, sources=sources,
        route=decision.route, tokens=usage.total_tokens,
    )



def run_agent(question: str, history: list[dict] | None = None) -> AgentResult:
    """Run one turn and return the finished result (no streaming)."""
    history = history or []
    started, usage = time.perf_counter(), Usage()
    logger.info("turn start | history=%d | question=%r", len(history), question[:80])

    decision = route_question(question, history, usage)
    steps, sources, blocks = [], [], []
    for tool_name, query, func in _plan(decision):
        _execute(tool_name, query, func, steps, sources, blocks)

    evidence = _evidence_text(blocks)
    response = invoke_llm(_build_messages(question, history, evidence), usage)
    answer = _finalize(strip_think(as_text(response.content)), evidence, history, usage)
    return _result(answer, steps, sources, decision, usage, started)


def stream_agent(question: str, history: list[dict] | None = None):
    """Run one turn and yield events while it works.

    Events:
      {"event": "status", "text": "..."}   progress message
      {"event": "tool",   "name": "..."}   a search is starting
      {"event": "token",  "text": "..."}   a piece of the answer
      {"event": "done",   "result": AgentResult}   always last
    """
    history = history or []
    started, usage = time.perf_counter(), Usage()
    logger.info("turn start (stream) | history=%d | question=%r", len(history), question[:80])

    yield {"event": "status", "text": "Choosing where to look..."}
    decision = route_question(question, history, usage)

    steps, sources, blocks = [], [], []
    for tool_name, query, func in _plan(decision):
        yield {"event": "tool", "name": tool_name}
        _execute(tool_name, query, func, steps, sources, blocks)

    evidence = _evidence_text(blocks)
    messages = _build_messages(question, history, evidence)

    if ENABLE_VERIFICATION and evidence:
        # We must check the answer BEFORE showing it, so this path can't stream word by word.
        yield {"event": "status", "text": "Writing the answer..."}
        draft = strip_think(as_text(invoke_llm(messages, usage).content))
        yield {"event": "status", "text": "Checking the answer against the sources..."}
        answer = _finalize(draft, evidence, history, usage)
        yield {"event": "token", "text": answer}
    else:
        answer = ""
        for piece in stream_llm(messages, usage):
            answer += piece
            yield {"event": "token", "text": piece}

    yield {"event": "done", "result": _result(answer, steps, sources, decision, usage, started)}