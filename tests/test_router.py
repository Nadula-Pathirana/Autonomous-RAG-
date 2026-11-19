import pytest

from autonomous_rag.router import RouteDecision, apply_rules, parse_route
from autonomous_rag.utils import extract_json


def decision(route="NO_TOOL", **kwargs):
    return RouteDecision(route=route, **kwargs)


def test_parse_route_ignores_think_block_and_extra_text():
    text = '<think>hmm {not json}</think>\nSure: {"route": "WEB_ONLY", "web_query": "Groq news"}'
    parsed = parse_route(text)
    assert parsed.route == "WEB_ONLY" and parsed.web_query == "Groq news"


def test_parse_route_rejects_unknown_route():
    with pytest.raises(ValueError):
        parse_route('{"route": "SEARCH_KNIGHT_BASE"}')


def test_extract_json_fails_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_no_tool_is_overridden_when_documents_clearly_match():
    final = apply_rules(decision("NO_TOOL"), "What happens if my secret is compromised?",
                        has_documents=True, web_available=True, probe_distance=0.4)
    assert final.route == "KB_ONLY"
    assert final.kb_query == "What happens if my secret is compromised?"


def test_no_tool_kept_for_questions_about_the_conversation():
    final = apply_rules(decision("NO_TOOL", about_conversation=True), "What did I ask earlier?",
                        has_documents=True, web_available=True, probe_distance=0.1)
    assert final.route == "NO_TOOL"


def test_no_tool_kept_when_documents_do_not_match():
    final = apply_rules(decision("NO_TOOL"), "Hi", True, True, probe_distance=0.95)
    assert final.route == "NO_TOOL"


def test_both_becomes_web_only_without_documents():
    assert apply_rules(decision("BOTH"), "q", False, True).route == "WEB_ONLY"


def test_both_becomes_kb_only_without_web():
    assert apply_rules(decision("BOTH"), "q", True, False).route == "KB_ONLY"


def test_empty_queries_fall_back_to_the_question():
    final = apply_rules(decision("BOTH"), "my question", True, True)
    assert final.kb_query == "my question" and final.web_query == "my question"