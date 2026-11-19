from autonomous_rag.agent import get_graph

graph = get_graph().get_graph()
mermaid = graph.draw_mermaid()

with open("workflow.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid)

# Optional PNG (uses the mermaid.ink website, so it needs internet)
with open("workflow.png", "wb") as f:
    f.write(graph.draw_mermaid_png())

print(mermaid)