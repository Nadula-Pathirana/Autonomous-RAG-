from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from config import AGENT_RECURSION_LIMIT
from autonomous_rag.llm import get_llm
from autonomous_rag.prompts import SYSTEM_PROMPT
from autonomous_rag.tools import ALL_TOOLS


@lru_cache(maxsize=2)
def get_agent(today: str):
    """Build the agent. Cached per date so the prompt always has today's date."""
    return create_agent(
        model=get_llm(),
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


def run_agent(question: str) -> tuple[str, list[dict]]:
    """Run the agent on a question. Returns (answer, steps)."""
    agent = get_agent(date.today().isoformat())

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
    except GraphRecursionError:
        return (
            "I got stuck in a loop while researching this. "
            "Try rephrasing or asking a simpler question.",
            [],
        )

    messages = result["messages"]
    answer = _as_text(messages[-1].content)
    return answer, _extract_steps(messages)