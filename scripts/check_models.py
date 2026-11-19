"""List the Groq models your key can use, and test candidates.

    uv run python scripts/check_models.py                    # list available models
    uv run python scripts/check_models.py MODEL_A MODEL_B    # test these models
"""
import sys

# pyrefly: ignore[missing-import]
from groq import Groq
from langchain_core.tools import tool
# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from autonomous_rag.utils import as_text, extract_json, strip_think
from config import GROQ_API_KEY

SKIP = ("whisper", "tts", "guard", "orpheus", "embed")   # not chat models


@tool
def search_knowledge_base(query: str) -> str:
    """Search the user's uploaded documents."""
    return "ok"


def list_models() -> list[str]:
    client = Groq(api_key=GROQ_API_KEY)
    ids = sorted(m.id for m in client.models.list().data)
    return [i for i in ids if not any(s in i.lower() for s in SKIP)]


def check(label: str, fn) -> str:
    try:
        return "ok" if fn() else "FAIL (unexpected output)"
    except Exception as error:
        return f"FAIL: {str(error)[:90]}"


def test_model(name: str) -> None:
    llm = ChatGroq(model=name, api_key=GROQ_API_KEY, temperature=0)

    plain = check("plain", lambda: strip_think(as_text(
        llm.invoke("Reply with the single word: pong").content)).strip())

    json_ok = check("json", lambda: extract_json(as_text(llm.invoke(
        'Reply with ONLY this JSON object: {"route": "KB_ONLY"}').content))["route"] == "KB_ONLY")

    tools_ok = check("tools", lambda: any(
        "query" in call["args"]
        for call in llm.bind_tools([search_knowledge_base]).invoke(
            "Search my documents for the AetherTech price match policy").tool_calls))

    print(f"\n{name}")
    print(f"  plain reply : {plain}")
    print(f"  JSON output : {json_ok}   <- required by V6")
    print(f"  tool calling: {tools_ok}   <- NOT required by V6 (informational)")


if __name__ == "__main__":
    if not GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is missing in .env")
    if len(sys.argv) == 1:
        print("Models available to your key:")
        for model_id in list_models():
            print(f"  {model_id}")
        print("\nTest some with: uv run python scripts/check_models.py MODEL_A MODEL_B")
    else:
        for model_id in sys.argv[1:]:
            test_model(model_id)