import pytest

from autonomous_rag import agent
from autonomous_rag.router import RouteDecision
from autonomous_rag.sources import document_source, web_source


@pytest.fixture(autouse=True)
def fresh_graph():
    agent.get_graph.cache_clear()
    yield
    agent.get_graph.cache_clear()


def fake_stream_factory(pieces, calls):
    def fake_stream(messages, usage=None, role="answer"):
        calls.append(messages)
        yield from pieces
    return fake_stream


def patch_searches(monkeypatch):
    monkeypatch.setattr(
        agent, "run_kb_search",
        lambda q: ("[a.pdf, p.1]\nreturns within 14 days", [document_source("a.pdf", 1, "x")]),
    )
    monkeypatch.setattr(
        agent, "run_web_search",
        lambda q: ("[T](https://x.com)\nweb text", [web_source("T", "https://x.com", "t")]),
    )


def test_both_route_runs_both_searches_and_answers_once(monkeypatch):
    calls = []
    patch_searches(monkeypatch)
    monkeypatch.setattr(
        agent, "route_question",
        lambda q, h, u=None: RouteDecision(route="BOTH", kb_query="kbq", web_query="webq"),
    )
    monkeypatch.setattr(agent, "stream_llm", fake_stream_factory(["Answer [a.pdf, p.1]"], calls))

    result = agent.run_agent("question")

    assert len(calls) == 1                                    # generate ran exactly once
    user_message = calls[0][-1]["content"]
    assert "DOCUMENT RESULTS" in user_message and "WEB RESULTS" in user_message
    assert user_message.index("DOCUMENT RESULTS") < user_message.index("WEB RESULTS")
    tools = {s["tool"] for s in result.steps if s["type"] == "call"}
    assert tools == {"search_knowledge_base", "search_web"}
    assert {"kb_search", "web_search", "generate", "total"} <= set(result.timings)
    assert result.sources[0]["cited"] is True


def test_no_tool_route_skips_searches(monkeypatch):
    calls = []
    patch_searches(monkeypatch)
    monkeypatch.setattr(
        agent, "route_question", lambda q, h, u=None: RouteDecision(route="NO_TOOL")
    )
    monkeypatch.setattr(agent, "stream_llm", fake_stream_factory(["Hello!"], calls))

    result = agent.run_agent("hi")

    assert result.steps == [] and result.answer == "Hello!"


def test_stream_agent_yields_tokens_then_done(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent, "route_question", lambda q, h, u=None: RouteDecision(route="NO_TOOL")
    )
    monkeypatch.setattr(agent, "stream_llm", fake_stream_factory(["Hel", "lo"], calls))

    events = list(agent.stream_agent("hi"))

    assert [e["text"] for e in events if e["event"] == "token"] == ["Hel", "lo"]
    assert events[-1]["event"] == "done"
    assert events[-1]["result"].answer == "Hello"