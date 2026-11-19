import time
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.errors import GraphRecursionError

from config import AGENT_RECURSION_LIMIT, GROQ_FALLBACK_MODEL, GROQ_MODEL
from autonomous_rag.llm import get_llm
from autonomous_rag.logger import get_logger
from autonomous_rag.prompts import SYSTEM_PROMPT
from autonomous_rag.sources import dedupe_sources, mark_cited
from autonomous_rag.tools import ALL_TOOLS

logger = get_logger("agent")

LOOP_MESSAGE = (
    "I got stuck in a loop while researching this. "
    "Try rephrasing or asking a simpler question."
)


@dataclass
class AgentResult:
    """Everything the UI needs from one agent run."""

    answer: str
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


@lru_cache(maxsize=4)
def get_agent(today: str, model_name: str = GROQ_MODEL):
    """Build the agent. Cached per date and model."""
    return create_agent(
        model=get_llm(model_name),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT.format(today=today),
    )


def _as_text(content) -> str:
    """Message content is usually a string, but can be a list of blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def _extract_steps(messages: list) -> list[dict]:
    """Pull out which tools the agent called, with what input and what came back."""
    steps = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                steps.append(
                    {"type": "call", "tool": call["name"], "input": call["args"]}
                )
        elif isinstance(message, ToolMessage):
            steps.append(
                {
                    "type": "result",
                    "tool": message.name,
                    "output": _as_text(message.content),
                }
            )
    return steps


def _extract_sources(messages: list, answer: str) -> list[dict]:
    """Collect the structured sources attached to this turn's tool results."""
    sources = []
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
            sources.extend(message.artifact)
    return mark_cited(dedupe_sources(sources), answer)


def _models_to_try() -> list[str]:
    """Primary model first, then the fallback (skipping duplicates)."""
    return list(dict.fromkeys([GROQ_MODEL, GROQ_FALLBACK_MODEL]))


def _build_result(new_messages: list, model_name: str, started: float) -> AgentResult:
    """Turn this turn's messages into an AgentResult, and log a summary."""
    answer = _as_text(new_messages[-1].content)
    steps = _extract_steps(new_messages)
    sources = _extract_sources(new_messages, answer)
    logger.info(
        "turn done | model=%s | tools=%s | sources=%d | answer_chars=%d | %.2fs",
        model_name,
        [s["tool"] for s in steps if s["type"] == "call"],
        len(sources),
        len(answer),
        time.perf_counter() - started,
    )
    return AgentResult(answer=answer, steps=steps, sources=sources)


def run_agent(question: str, history: list[dict] | None = None) -> AgentResult:
    """Run the agent and return the finished result (no streaming)."""
    input_messages = (history or []) + [{"role": "user", "content": question}]
    today = date.today().isoformat()
    started = time.perf_counter()
    logger.info("turn start | history=%d | question=%r", len(history or []), question[:80])
    last_error = None

    for model_name in _models_to_try():
        agent = get_agent(today, model_name)
        try:
            result = agent.invoke(
                {"messages": input_messages},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            logger.warning("recursion limit reached | question=%r", question[:80])
            return AgentResult(answer=LOOP_MESSAGE)
        except Exception as error:
            if "tool_use_failed" in str(error):
                logger.warning("tool_use_failed with %s, trying the fallback", model_name)
                last_error = error
                continue
            raise

        # Keep only the messages created during THIS turn
        new_messages = result["messages"][len(input_messages):]
        return _build_result(new_messages, model_name, started)

    # pyrefly: ignore[bad-raise]
    raise last_error


def stream_agent(question: str, history: list[dict] | None = None):
    """Run the agent and yield events while it works.

    Events (dictionaries):
      {"event": "token", "text": "..."}   a piece of the answer
      {"event": "tool",  "name": "..."}   the agent is calling a tool
      {"event": "reset"}                  discard text shown so far
      {"event": "done",  "result": AgentResult}   always the last event
    """
    input_messages = (history or []) + [{"role": "user", "content": question}]
    today = date.today().isoformat()
    started = time.perf_counter()
    logger.info("turn start (stream) | history=%d | question=%r", len(history or []), question[:80])
    last_error = None

    for model_name in _models_to_try():
        agent = get_agent(today, model_name)
        final_state = None
        announced = set()   # message ids we already announced a tool call for

        try:
            for mode, data in agent.stream(
                {"messages": input_messages},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
                stream_mode=["messages", "values"],
            ):
                if mode == "values":
                    final_state = data          # the full state after each step
                    continue

                chunk, _metadata = data         # mode == "messages"
                if not isinstance(chunk, AIMessageChunk):
                    continue                    # ignore tool outputs

                if chunk.tool_call_chunks:
                    # The model is calling a tool, so any text it wrote just
                    # before that was not the final answer: throw it away.
                    if chunk.id not in announced:
                        announced.add(chunk.id)
                        yield {"event": "reset"}
                    name = next(
                        (c.get("name") for c in chunk.tool_call_chunks if c.get("name")),
                        None,
                    )
                    if name:
                        yield {"event": "tool", "name": name}
                elif chunk.content:
                    yield {"event": "token", "text": _as_text(chunk.content)}

        except GraphRecursionError:
            logger.warning("recursion limit reached | question=%r", question[:80])
            yield {"event": "done", "result": AgentResult(answer=LOOP_MESSAGE)}
            return
        except Exception as error:
            if "tool_use_failed" in str(error):
                logger.warning("tool_use_failed with %s, trying the fallback", model_name)
                last_error = error
                yield {"event": "reset"}
                continue
            raise

        if not final_state:
            raise RuntimeError("The agent finished without producing a result.")

        new_messages = final_state["messages"][len(input_messages):]
        yield {"event": "done", "result": _build_result(new_messages, model_name, started)}
        return

    # pyrefly: ignore[bad-raise]
    raise last_error