import functools
import operator
import time
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from config import MAX_EVIDENCE_CHARS, VERIFY_MODE
from autonomous_rag.grounding import grounding_warnings
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


# ---------------------------------------------------------------- state

def merge_dicts(left: dict, right: dict) -> dict:
    return {**(left or {}), **(right or {})}


class RAGState(TypedDict, total=False):
    # inputs
    question: str
    history: list[dict]
    started: float
    # filled in by the nodes
    decision: RouteDecision
    steps: Annotated[list[dict], operator.add]      # parallel nodes both add steps
    sources: Annotated[list[dict], operator.add]
    blocks: Annotated[list[tuple], operator.add]    # (tool_name, text) evidence pieces
    evidence: str
    answer: str
    claims: list[str]
    timings: Annotated[dict, merge_dicts]           # seconds per node
    usage: Annotated[dict, merge_dicts]             # tokens per node


@dataclass
class AgentResult:
    """Everything the UI and the eval need from one run."""

    answer: str
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    route: str = ""
    tokens: int = 0
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def timed(name: str):
    """Record how long a node took (added to state['timings'])."""
    def decorator(fn):
        @functools.wraps(fn)
        def inner(state):
            started = time.perf_counter()
            update = fn(state)
            seconds = round(time.perf_counter() - started, 2)
            update["timings"] = {**update.get("timings", {}), name: seconds}
            return update
        return inner
    return decorator


def _evidence_text(blocks: list[tuple]) -> str:
    parts = []
    for tool_name, text in sorted(blocks, key=lambda b: b[0] != TOOL_KB):   # documents first
        title = (
            "DOCUMENT RESULTS (the user's uploaded files)"
            if tool_name == TOOL_KB
            else "WEB RESULTS (the internet)"
        )
        parts.append(f"=== {title} ===\n{text[:MAX_EVIDENCE_CHARS]}")
    return "\n\n".join(parts)


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
        logger.warning("verification failed; keeping the answer as it is", exc_info=True)
        return []
    return [str(claim) for claim in claims if str(claim).strip()]


# ---------------------------------------------------------------- nodes

@timed("route")
def route_node(state):
    get_stream_writer()({"event": "status", "text": "Thinking..."})
    usage = Usage()
    decision = route_question(state["question"], state["history"], usage)
    return {"decision": decision, "usage": {"route": usage.as_dict()}}


def _search(tool_name: str, query: str, func) -> dict:
    get_stream_writer()({"event": "tool", "name": tool_name})
    text, found = func(query)
    return {
        "steps": [
            {"type": "call", "tool": tool_name, "input": {"query": query}},
            {"type": "result", "tool": tool_name, "output": text},
        ],
        "sources": found,
        "blocks": [(tool_name, text)],
    }


@timed("kb_search")
def kb_search_node(state):
    return _search(TOOL_KB, state["decision"].kb_query, run_kb_search)


@timed("web_search")
def web_search_node(state):
    return _search(TOOL_WEB, state["decision"].web_query, run_web_search)


@timed("generate")
def generate_node(state):
    """Write the answer, sending each piece to the UI the moment it arrives."""
    write = get_stream_writer()
    usage = Usage()
    evidence = _evidence_text(state.get("blocks", []))
    messages = _build_messages(state["question"], state["history"], evidence)

    text, first_token_at = "", None
    for piece in stream_llm(messages, usage):
        if not text:
            piece = piece.lstrip()               # models often start with blank lines
            if not piece:
                continue
            first_token_at = round(time.perf_counter() - state["started"], 2)
        text += piece
        write({"event": "token", "text": piece})

    if not text:
        text = "I couldn't produce an answer. Please try again."
        write({"event": "token", "text": text})

    if usage.total_tokens == 0:                  # provider sent no usage while streaming:
        usage.input_tokens = sum(len(m["content"]) for m in messages) // 4   # estimate,
        usage.output_tokens = len(text) // 4                                 # ~4 chars/token

    return {
        "answer": text,
        "evidence": evidence,
        "usage": {"generate": usage.as_dict()},
        "timings": {"first_token_at": first_token_at or 0},
    }


@timed("verify")
def verify_node(state):
    get_stream_writer()({"event": "status", "text": "Fact-checking the answer..."})
    usage = Usage()
    claims = _verify(state["answer"], state["evidence"], state["history"], usage)
    return {"claims": claims, "usage": {"verify": usage.as_dict()}}


@timed("revise")
def revise_node(state):
    usage = Usage()
    logger.warning("unsupported claims removed: %s", state["claims"])
    prompt = REWRITE_PROMPT.format(
        evidence=state["evidence"],
        answer=state["answer"],
        claims="\n".join(f"- {c}" for c in state["claims"]),
    )
    answer = state["answer"]
    try:
        response = invoke_llm([{"role": "user", "content": prompt}], usage)
        answer = strip_think(as_text(response.content)) or answer
    except Exception:
        logger.warning("rewrite failed; keeping the original answer", exc_info=True)
    return {"answer": answer, "usage": {"revise": usage.as_dict()}}


# ---------------------------------------------------------------- edges (decisions)

def fan_out(state) -> list[str]:
    """After routing: which nodes to run. Two names = they run in PARALLEL."""
    route = state["decision"].route
    if route == "BOTH":
        return ["kb_search", "web_search"]
    if route == "KB_ONLY":
        return ["kb_search"]
    if route == "WEB_ONLY":
        return ["web_search"]
    return ["generate"]


def after_generate(state) -> str:
    if VERIFY_MODE == "after" and state.get("sources"):
        return "verify"
    return END


def after_verify(state) -> str:
    return "revise" if state.get("claims") else END


@lru_cache(maxsize=1)
def get_graph():
    """Build and compile the graph once."""
    builder = StateGraph(RAGState)
    builder.add_node("route", route_node)
    builder.add_node("kb_search", kb_search_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("revise", revise_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", fan_out, ["kb_search", "web_search", "generate"])
    builder.add_edge("kb_search", "generate")
    builder.add_edge("web_search", "generate")
    builder.add_conditional_edges("generate", after_generate, ["verify", END])
    builder.add_conditional_edges("verify", after_verify, ["revise", END])
    builder.add_edge("revise", END)
    return builder.compile()


# ---------------------------------------------------------------- public API (same names as before)

def _initial_state(question: str, history: list[dict] | None) -> dict:
    return {"question": question, "history": history or [], "started": time.perf_counter()}


def _to_result(state: dict) -> AgentResult:
    answer = state.get("answer", "")
    evidence = state.get("evidence", "")
    sources = mark_cited(dedupe_sources(state.get("sources", [])), answer)
    usage = state.get("usage", {})
    tokens = sum(u["input"] + u["output"] for u in usage.values())
    timings = {**state.get("timings", {}), "total": round(time.perf_counter() - state["started"], 2)}

    warnings = []
    if evidence:
        context = format_history(state["history"], limit=10, chars=2000) + " " + date.today().isoformat()
        warnings = grounding_warnings(answer, evidence, extra_context=context)

    logger.info(
        "turn done | route=%s | sources=%d | tokens=%d | warnings=%d | timings=%s",
        state["decision"].route, len(sources), tokens, len(warnings), timings,
    )
    return AgentResult(
        answer=answer, steps=state.get("steps", []), sources=sources,
        route=state["decision"].route, tokens=tokens, warnings=warnings,
        timings=timings, usage=usage,
    )


def run_agent(question: str, history: list[dict] | None = None) -> AgentResult:
    """Run one turn and return the finished result (no live streaming)."""
    logger.info("turn start | history=%d | question=%r", len(history or []), question[:80])
    state = get_graph().invoke(_initial_state(question, history))
    return _to_result(state)


def stream_agent(question: str, history: list[dict] | None = None):
    """Run one turn and yield events while it works.

    Events:
      {"event": "status", "text": "..."}   progress message
      {"event": "tool",   "name": "..."}   a search is starting
      {"event": "token",  "text": "..."}   a piece of the answer
      {"event": "done",   "result": AgentResult}   always last
    """
    logger.info("turn start (stream) | history=%d | question=%r", len(history or []), question[:80])
    final_state = None
    for mode, data in get_graph().stream(
        _initial_state(question, history), stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            yield data                    # our own events from get_stream_writer()
        else:
            final_state = data            # the full state after each step
    if not final_state or "decision" not in final_state:
        raise RuntimeError("The agent finished without producing a result.")
    yield {"event": "done", "result": _to_result(final_state)}