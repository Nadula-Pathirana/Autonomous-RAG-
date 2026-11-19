"""Compare speed and tokens of a model under different reasoning settings.

    uv run python scripts/check_speed.py MODEL_A MODEL_B
"""
import sys
import time

# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from config import GROQ_API_KEY

PROMPT = (
    "Route this question: 'What is the price match policy?' "
    'Reply with ONE JSON object like {"route": "KB_ONLY"}.'
)


def attempt(model: str, **options) -> None:
    label = str(options) if options else "default settings"
    try:
        llm = ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=0, **options)
        started = time.perf_counter()
        reply = llm.invoke(PROMPT)
        seconds = time.perf_counter() - started
        usage = reply.usage_metadata or {}
        print(f"  {label:42} {seconds:5.1f}s  in={usage.get('input_tokens')}  out={usage.get('output_tokens')}")
    except Exception as error:
        print(f"  {label:42} FAIL: {str(error)[:90]}")


for model in sys.argv[1:]:
    print(model)
    attempt(model)
    for effort in ("none", "low", "medium", "high", "default"):
        attempt(model, reasoning_effort=effort)