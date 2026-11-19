from autonomous_rag.grounding import grounding_warnings
from autonomous_rag.router import is_greeting
from autonomous_rag.utils import ThinkFilter


def feed_all(pieces):
    think = ThinkFilter()
    return "".join(think.feed(p) for p in pieces) + think.flush()


def test_think_filter_removes_a_block_split_across_pieces():
    assert feed_all(["<thi", "nk>secret</th", "ink>Hello", " world"]) == "Hello world"


def test_think_filter_passes_normal_text_through():
    assert feed_all(["Hello ", "<b>x</b>"]) == "Hello <b>x</b>"


def test_think_filter_keeps_a_lone_angle_bracket():
    assert feed_all(["a < b"]) == "a < b"


def test_flags_a_number_that_is_not_in_the_evidence():
    warnings = grounding_warnings("You have 45 days. [a.pdf, p.1]", "[a.pdf, p.1]\nReturns within 30 days")
    assert any("45" in w for w in warnings)


def test_no_warning_when_numbers_and_citations_are_supported():
    assert grounding_warnings("Returns take 30 days [a.pdf, p.1].", "[a.pdf, p.1]\nReturns within 30 days") == []


def test_flags_a_citation_that_was_not_retrieved():
    warnings = grounding_warnings("See [a.pdf, p.9].", "[a.pdf, p.1]\ntext")
    assert any("p.9" in w for w in warnings)


def test_greeting_fast_path():
    assert is_greeting("Hi, what can you do?")
    assert is_greeting("hello!")
    assert not is_greeting("Hello! Can you explain how you decide?")
    assert not is_greeting("What is the price match policy?")