from autonomous_rag.memory import build_history
from config import MAX_HISTORY_MESSAGES, MAX_MESSAGE_CHARS


def msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def test_skips_error_messages():
    history = build_history([
        msg("user", "q1", error=True),
        msg("assistant", "oops", error=True),
        msg("user", "q2"),
        msg("assistant", "a2"),
    ])
    assert [m["content"] for m in history] == ["q2", "a2"]


def test_keeps_only_recent_messages():
    messages = []
    for i in range(20):
        messages += [msg("user", f"q{i}"), msg("assistant", f"a{i}")]
    history = build_history(messages)
    assert len(history) <= MAX_HISTORY_MESSAGES
    assert history[-1]["content"] == "a19"


def test_shortens_long_messages():
    history = build_history([msg("user", "q"), msg("assistant", "x" * (MAX_MESSAGE_CHARS * 3))])
    assert len(history[-1]["content"]) < MAX_MESSAGE_CHARS + 50


def test_history_starts_with_a_user_message():
    history = build_history([msg("assistant", "hello"), msg("user", "q"), msg("assistant", "a")])
    assert history[0]["role"] == "user"


def test_empty_history():
    assert build_history([]) == []