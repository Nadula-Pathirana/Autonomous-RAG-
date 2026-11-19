import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from autonomous_rag.agent import run_agent
from autonomous_rag.llm import get_llm, invoke_llm
from autonomous_rag.utils import as_text, extract_json
from config import GROQ_MODEL, MAX_DISTANCE, TOP_K

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_FILE = Path("eval/results.md")
PAUSE_SECONDS = 3          # be gentle with free-tier rate limits
JUDGE_MODEL = ""           # optional: a DIFFERENT model than the one being tested

TOOL_NAMES = {"search_knowledge_base": "kb", "search_web": "web"}
EXPECTED_TOOLS = {"kb_only": {"kb"}, "web_only": {"web"}, "both": {"kb", "web"}, "none": set()}

JUDGE_PROMPT = """You are grading a retrieval-augmented assistant.

QUESTION: {question}

EVIDENCE (what the assistant retrieved):
{evidence}

ANSWER:
{answer}

faithful = every factual claim in the ANSWER is stated in the EVIDENCE
           (an honest "I couldn't find it" also counts as faithful).
relevant = the ANSWER actually addresses the QUESTION.

Reply with ONE JSON object and nothing else:
{{"faithful": true, "relevant": true, "unsupported_claims": []}}"""


def tools_used(result) -> set[str]:
    return {
        TOOL_NAMES.get(step["tool"], step["tool"])
        for step in result.steps if step["type"] == "call"
    }


def evidence_text(result) -> str:
    return "\n".join(step["output"] for step in result.steps if step["type"] == "result")


def has_facts(case: dict, text: str) -> bool:
    text = text.lower()
    all_ok = all(t.lower() in text for t in case.get("must_contain", []))
    any_of = case.get("must_contain_any", [])
    any_ok = (not any_of) or any(t.lower() in text for t in any_of)
    return all_ok and any_ok


def judge(case: dict, result) -> dict:
    """LLM judge. Returns None values when there is nothing to judge or the judge fails."""
    evidence = evidence_text(result)
    if not evidence:
        return {"faithful": None, "relevant": None}
    prompt = JUDGE_PROMPT.format(
        question=case["question"], evidence=evidence[:8000], answer=result.answer
    )
    try:
        if JUDGE_MODEL:
            response = get_llm(JUDGE_MODEL).invoke(prompt)
        else:
            response = invoke_llm([{"role": "user", "content": prompt}])
        verdict = extract_json(as_text(response.content))
        return {"faithful": bool(verdict.get("faithful")), "relevant": bool(verdict.get("relevant"))}
    except Exception:
        return {"faithful": None, "relevant": None}


def grade(case: dict, result) -> dict:
    """Return {metric: True/False/None}. None = not applicable or could not be measured."""
    cited = [s for s in result.sources if s.get("cited")]
    abstain = case.get("expect_abstain", False)

    citation_ok = True
    if case.get("expect_citation"):
        citation_ok = bool(cited)
    expected_sources = case.get("expected_sources")
    if expected_sources:
        citation_ok = citation_ok and all(
            s["title"] in expected_sources for s in cited if s["type"] == "document"
        )

    verdict = judge(case, result)
    return {
        "tool": tools_used(result) == EXPECTED_TOOLS[case["expect_tool"]],
        "retrieval": None if (case["expect_tool"] == "none" or abstain)
                     else has_facts(case, evidence_text(result)),
        "content": has_facts(case, result.answer),
        "citation": citation_ok,
        "abstain": (not cited) if abstain else None,
        "faithful": verdict["faithful"],
        "relevant": verdict["relevant"],
    }


def mark(value) -> str:
    return "-" if value is None else ("ok" if value else "FAIL")


def rate(rows: list[dict], key: str) -> str:
    values = [r["grades"][key] for r in rows if r["status"] != "ERROR" and r["grades"].get(key) is not None]
    return f"{sum(values)}/{len(values)}" if values else "n/a"


def main() -> None:
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    console = Console()
    rows = []

    for case in cases:
        console.print(f"[dim]Running:[/dim] {case['id']} ...")
        started = time.perf_counter()
        try:
            result = run_agent(case["question"])   # each question is independent (no history)
            grades = grade(case, result)
            passed = all(v for v in grades.values() if v is not None)
            rows.append({
                "id": case["id"], "status": "PASS" if passed else "FAIL", "grades": grades,
                "used": ", ".join(sorted(tools_used(result))) or "-", "route": result.route,
                "seconds": time.perf_counter() - started, "tokens": result.tokens, "note": "",
            })
        except Exception as error:
            # Infrastructure problem (model 404, rate limit...). NOT a RAG failure.
            rows.append({
                "id": case["id"], "status": "ERROR", "grades": {}, "used": "-", "route": "-",
                "seconds": time.perf_counter() - started, "tokens": 0, "note": str(error)[:80],
            })
        time.sleep(PAUSE_SECONDS)

    graded = [r for r in rows if r["status"] != "ERROR"]
    passed_count = sum(r["status"] == "PASS" for r in rows)
    errors = len(rows) - len(graded)
    avg_seconds = sum(r["seconds"] for r in graded) / max(len(graded), 1)
    avg_tokens = sum(r["tokens"] for r in graded) / max(len(graded), 1)
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"

    summary = [
        f"**Passed {passed_count}/{len(graded)} graded** ({errors} infrastructure errors not counted)",
        f"Tool selection {rate(rows, 'tool')} | Retrieval recall {rate(rows, 'retrieval')} | "
        f"Citation {rate(rows, 'citation')} | Abstention {rate(rows, 'abstain')} | "
        f"Faithfulness {rate(rows, 'faithful')} | Relevance {rate(rows, 'relevant')}",
        f"Avg latency {avg_seconds:.1f}s | Avg tokens {avg_tokens:.0f}",
    ]
    columns = ["Question", "Status", "Tools", "Tool", "Retrieval", "Content", "Citation",
               "Abstain", "Faithful", "Relevant", "Sec", "Tokens", "Note"]

    def row_cells(r):
        g = r["grades"]
        return [r["id"], r["status"], r["used"]] + [
            mark(g.get(k)) if g else "-"
            for k in ("tool", "retrieval", "content", "citation", "abstain", "faithful", "relevant")
        ] + [f"{r['seconds']:.1f}", str(r["tokens"]), r["note"]]

    # 1. Markdown report
    lines = [f"# Evaluation run {stamp}", "",
             f"Model: `{GROQ_MODEL}` | TOP_K: {TOP_K} | MAX_DISTANCE: {MAX_DISTANCE}", ""]
    lines += summary + ["", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row_cells(r)) + " |" for r in rows]
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. Terminal table
    table = Table(title=f"Evaluation Run: {stamp}", caption="\n".join(summary), show_lines=True)
    for name in columns:
        table.add_column(name, overflow="fold")
    for r in rows:
        table.add_row(*row_cells(r))
    console.print()
    console.print(table)


if __name__ == "__main__":
    main()