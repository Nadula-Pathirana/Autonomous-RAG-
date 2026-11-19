from typing import Literal

from pydantic import BaseModel

from config import PROBE_DISTANCE, ROUTER_HISTORY_MESSAGES, TAVILY_API_KEY
from autonomous_rag.llm import invoke_llm
from autonomous_rag.logger import get_logger
from autonomous_rag.utils import as_text, extract_json, format_history
from autonomous_rag.vector_store import list_documents, search_with_scores

logger = get_logger("router")


class RouteDecision(BaseModel):
    route: Literal["KB_ONLY", "WEB_ONLY", "BOTH", "NO_TOOL"]
    kb_query: str = ""
    web_query: str = ""
    about_conversation: bool = False
    reason: str = ""


ROUTER_PROMPT = """You are the routing step of a research assistant. Decide which sources
are needed to answer the user's LAST question. Do NOT answer the question.

Routes:
- KB_ONLY: the question is about the user's uploaded documents, or could plausibly be
  answered by them (policies, products, procedures, internal facts, error codes, or any
  vague question that might refer to the documents).
- WEB_ONLY: needs current events, recent facts, prices, or public information that cannot
  be in the user's documents.
- BOTH: needs the documents AND outside information (for example, comparing the user's
  documents with the outside world).
- NO_TOOL: ONLY greetings, questions about what you can do, or questions about the
  conversation itself (summarise it, rephrase a previous answer, "what did I ask?").

Rules:
- If unsure between NO_TOOL and KB_ONLY, choose KB_ONLY.
- kb_query and web_query must be complete, standalone search phrases that name the topic.
  Use the conversation to resolve words like "it", "that", "they". Never use pronouns.
  Leave a query empty if that source is not used.
- Set about_conversation to true ONLY if the question is about the chat itself or asks to
  rephrase a previous answer.

Documents currently uploaded: {documents}

Conversation so far:
{history}

Last question: {question}

Reply with ONE JSON object and nothing else:
{{"route": "KB_ONLY", "kb_query": "...", "web_query": "", "about_conversation": false, "reason": "..."}}"""


def parse_route(text: str) -> RouteDecision:
    """Turn the router's reply into a RouteDecision. Raises ValueError if it is invalid."""
    return RouteDecision.model_validate(extract_json(text))


def apply_rules(
    decision: RouteDecision,
    question: str,
    has_documents: bool,
    web_available: bool,
    probe_distance: float | None = None,
) -> RouteDecision:
    """Hard rules applied on top of the LLM's decision. Pure function, easy to test."""
    route = decision.route
    kb_query = decision.kb_query.strip() or question
    web_query = decision.web_query.strip() or question

    # Safety net: the router said "no tool", but the documents clearly match.
    if (
        route == "NO_TOOL"
        and not decision.about_conversation
        and probe_distance is not None
        and probe_distance <= PROBE_DISTANCE
    ):
        route = "KB_ONLY"

    # Don't plan to use a source that doesn't exist.
    if route == "BOTH" and not has_documents:
        route = "WEB_ONLY"
    if route == "BOTH" and not web_available:
        route = "KB_ONLY"

    return decision.model_copy(update={"route": route, "kb_query": kb_query, "web_query": web_query})


def route_question(question: str, history: list[dict], usage=None) -> RouteDecision:
    """Decide which sources to use for this question."""
    try:
        documents = list_documents()
    except Exception:
        logger.exception("could not list documents for the router")
        documents = []

    prompt = ROUTER_PROMPT.format(
        documents=", ".join(d["name"] for d in documents) or "(none)",
        history=format_history(history, limit=ROUTER_HISTORY_MESSAGES),
        question=question,
    )

    try:
        response = invoke_llm([{"role": "user", "content": prompt}], usage)
        decision = parse_route(as_text(response.content))
    except Exception as error:
        # Router failed: the documents are the safest default.
        logger.warning("router failed (%s). Defaulting to KB_ONLY.", str(error)[:150])
        decision = RouteDecision(route="KB_ONLY", reason="router failed")

    probe = None
    if documents and decision.route == "NO_TOOL" and not decision.about_conversation:
        try:
            nearest = search_with_scores(question, k=1)
            probe = nearest[0][1] if nearest else None
        except Exception:
            logger.warning("retrieval probe failed", exc_info=True)

    final = apply_rules(
        decision, question,
        has_documents=bool(documents),
        web_available=bool(TAVILY_API_KEY),
        probe_distance=probe,
    )
    logger.info(
        "route | llm=%s final=%s probe=%s | kb_query=%r web_query=%r | %s",
        decision.route, final.route, probe, final.kb_query, final.web_query, final.reason[:80],
    )
    return final