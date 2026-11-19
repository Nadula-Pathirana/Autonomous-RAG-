from typing import Literal
import re

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


GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b"
    r"[\s!.,?]*(what can you do\??\s*)?$",
    re.IGNORECASE,
)

ROUTER_PROMPT = """Route the user's LAST question. Do NOT answer it.
KB_ONLY: about the user's uploaded documents, or could plausibly be in them (policies,
  products, procedures, error codes, anything vague).
WEB_ONLY: needs current or public information that cannot be in the documents.
BOTH: needs the documents AND outside information.
NO_TOOL: ONLY greetings, "what can you do?", or questions about the chat itself
  (summarise, rephrase, "what did I ask?").
If unsure between NO_TOOL and KB_ONLY, choose KB_ONLY.
kb_query / web_query: complete standalone search phrases naming the topic. Resolve "it" or
  "that" from the conversation. Leave empty if unused.
about_conversation: true only for questions about the chat itself.

Documents: {documents}
Conversation:
{history}
Question: {question}

Reply with ONE JSON object only:
{{"route": "KB_ONLY", "kb_query": "...", "web_query": "", "about_conversation": false}}"""


def is_greeting(question: str) -> bool:
    return bool(GREETING.match(question))



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
    if is_greeting(question):                     # no LLM call needed
        logger.info("route | greeting fast path")
        return RouteDecision(route="NO_TOOL", about_conversation=True, reason="greeting")

    try:
        documents = list_documents()
    except Exception:
        logger.exception("could not list documents for the router")
        documents = []

    prompt = ROUTER_PROMPT.format(
        documents=", ".join(d["name"] for d in documents) or "(none)",
        history=format_history(history, limit=ROUTER_HISTORY_MESSAGES, chars=200),
        question=question,
    )

    try:
        response = invoke_llm([{"role": "user", "content": prompt}], usage, role="router")
        decision = parse_route(as_text(response.content))
    except Exception as error:
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
        "route | llm=%s final=%s probe=%s | kb_query=%r web_query=%r",
        decision.route, final.route, probe, final.kb_query, final.web_query,
    )
    return final