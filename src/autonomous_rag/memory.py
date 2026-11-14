from config import MAX_HISTORY_CHARS, MAX_HISTORY_MESSAGES, MAX_MESSAGE_CHARS


def build_history(messages: list[dict]) -> list[dict]:
    """Turn Streamlit chat messages into the history that is sent to the agent.

    - keeps only user/assistant text (no tool outputs, those are bulky)
    - skips messages marked as errors
    - shortens very long messages
    - keeps only the most recent messages, within a character budget
    """
    cleaned = []
    for msg in messages:
        if msg.get("error") or msg.get("role") not in ("user", "assistant"):
            continue
        content = msg["content"].strip()
        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS] + " ...[shortened]"
        cleaned.append({"role": msg["role"], "content": content})

    # 1. Keep only the most recent N messages
    cleaned = cleaned[-MAX_HISTORY_MESSAGES:]

    # 2. Drop the oldest messages until we're under the character budget
    while cleaned and sum(len(m["content"]) for m in cleaned) > MAX_HISTORY_CHARS:
        cleaned.pop(0)

    # 3. History should start with a user message
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    return cleaned