import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from autonomous_rag.agent import run_agent
from config import GROQ_MODEL, MAX_DISTANCE, TOP_K

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_FILE = Path("eval/results.md")
PAUSE_SECONDS = 3  # be gentle with free-tier rate limits

TOOL_NAMES = {"search_knowledge_base": "kb", "search_web": "web"}
EXPECTED_TOOLS = {
    "kb_only": {"kb"},
    "web_only": {"web"},
    "both": {"kb", "web"},
    "none": set(),
}


def tools_used(result) -> set[str]:
    return {
        TOOL_NAMES.get(step["tool"], step["tool"])
        for step in result.steps
        if step["type"] == "call"
    }


def grade(case: dict, result) -> dict:
    answer = result.answer.lower()
    used = tools_used(result)

    tool_ok = used == EXPECTED_TOOLS[case["expect_tool"]]

    contains_all = all(t.lower() in answer for t in case.get("must_contain", []))
    any_of = case.get("must_contain_any", [])
    contains_any = (not any_of) or any(t.lower() in answer for t in any_of)

    citation_ok = True
    if case.get("expect_citation"):
        citation_ok = any(source.get("cited") for source in result.sources)

    return {
        "tool": tool_ok,
        "content": contains_all and contains_any,
        "citation": citation_ok,
        "used": ", ".join(sorted(used)) or "-",
    }


def build_rich_table(
    rows: list[tuple[str, bool, dict, str]],
    passed_count: int,
    run_timestamp: str,
) -> Table:
    table = Table(
        title=f"Evaluation Run — {run_timestamp}",
        caption=(
            f"Model: [bold]{GROQ_MODEL}[/bold] | TOP_K: {TOP_K} | "
            f"MAX_DISTANCE: {MAX_DISTANCE} | "
            f"Passed: [bold green]{passed_count}/{len(rows)}[/bold green]"
        ),
        show_lines=True,
        header_style="bold cyan",
    )

    table.add_column("Question", style="bold", no_wrap=True)
    table.add_column("Pass", justify="center")
    table.add_column("Tools Used", style="magenta")
    table.add_column("Tool", justify="center")
    table.add_column("Content", justify="center")
    table.add_column("Citation", justify="center")
    table.add_column("Error", style="red", overflow="fold")

    status_badge = lambda ok: "[bold green]PASS[/bold green]" if ok else "[bold red]FAIL[/bold red]"

    for case_id, passed, g, error in rows:
        table.add_row(
            case_id,
            status_badge(passed),
            g["used"],
            status_badge(g["tool"]),
            status_badge(g["content"]),
            status_badge(g["citation"]),
            error or "-",
        )

    return table


def main() -> None:
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    console = Console()
    rows = []

    for case in cases:
        console.print(f"[dim]Running:[/dim] {case['id']} ...")
        try:
            result = run_agent(case["question"])  # each question is independent (no history)
            grades = grade(case, result)
            passed = grades["tool"] and grades["content"] and grades["citation"]
            rows.append((case["id"], passed, grades, ""))
        except Exception as error:
            rows.append((
                case["id"],
                False,
                {"used": "-", "tool": False, "content": False, "citation": False},
                str(error)[:80],
            ))
        time.sleep(PAUSE_SECONDS)

    passed_count = sum(1 for _id, passed, _g, _e in rows if passed)
    run_timestamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    mark = lambda ok: "Passed" if ok else "Failed"

    # 1. Save clean Markdown report to eval/results.md
    md_lines = [
        f"# Evaluation run {run_timestamp}",
        "",
        f"Model: `{GROQ_MODEL}` | TOP_K: {TOP_K} | MAX_DISTANCE: {MAX_DISTANCE}",
        f"**Passed {passed_count}/{len(rows)}**",
        "",
        "| Question | Pass | Tools used | Tool | Content | Citation | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for case_id, passed, g, error in rows:
        md_lines.append(
            f"| {case_id} | {mark(passed)} | {g['used']} | {mark(g['tool'])} | "
            f"{mark(g['content'])} | {mark(g['citation'])} | {error} |"
        )
    RESULTS_FILE.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 2. Print styled table to terminal
    console.print()
    console.print(build_rich_table(rows, passed_count, run_timestamp))


if __name__ == "__main__":
    main()