from datetime import date
from autonomous_rag.agent import get_agent
mermaid = get_agent(date.today().isoformat()).get_graph().draw_mermaid()
graph = get_agent(date.today().isoformat()).get_graph()
with open("workflow.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid)

# Optional PNG image (this uses the mermaid.ink website, so it needs internet)
with open("workflow.png", "wb") as f:
    f.write(graph.draw_mermaid_png())

print(mermaid)
