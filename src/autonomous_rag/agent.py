from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from config import AGENT_RECURSION_LIMIT, GROQ_FALLBACK_MODEL, GROQ_MODEL
from autonomous_rag.llm import get_llm
from autonomous_rag.prompts import SYSTEM_PROMPT
from autonomous_rag.sources import dedupe_sources, mark_cited
from autonomous_rag.tools import ALL_TOOLS


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


def run_agent(question: str, history: list[dict] | None = None) -> AgentResult:
    """Run the agent on a question, with earlier messages as memory."""
    input_messages = (history or []) + [{"role": "user", "content": question}]
    today = date.today().isoformat()

    # Try the primary model first, then the fallback (skipping duplicates)
    models_to_try = list(dict.fromkeys([GROQ_MODEL, GROQ_FALLBACK_MODEL]))
    last_error = None

    for model_name in models_to_try:
        agent = get_agent(today, model_name)
        try:
            result = agent.invoke(
                {"messages": input_messages},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            return AgentResult(
                answer=(
                    "I got stuck in a loop while researching this. "
                    "Try rephrasing or asking a simpler question."
                )
            )
        except Exception as e:
            # Groq rejects malformed tool calls with a 'tool_use_failed' error
            if "tool_use_failed" in str(e):
                last_error = e
                continue  # try the next model
            raise

        # Keep only the messages created during THIS turn
        new_messages = result["messages"][len(input_messages):]
        answer = _as_text(new_messages[-1].content)
        return AgentResult(
            answer=answer,
            steps=_extract_steps(new_messages),
            sources=_extract_sources(new_messages, answer),
        )

    # pyrefly: ignore[bad-raise]
    raise last_error