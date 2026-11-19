from autonomous_rag.vector_store import search_with_scores

# Questions your documents CAN answer (edit these to match your PDFs)
RELEVANT = [
    "AetherTech price match guarantee",
    "How many days do I have to return an item?",
    "How is customer data encrypted?",
    "Which items are excluded from price matching?",
]

# Questions your documents can NOT answer
IRRELEVANT = [
    "What is the CEO's favourite colour?",
    "How do volcanoes erupt?",
    "Best pizza topping recipe",
    "Who won the 2018 football World Cup?",
]


def best_distance(question: str) -> float:
    return min(distance for _doc, distance in search_with_scores(question, k=3))


for label, questions in (("RELEVANT", RELEVANT), ("IRRELEVANT", IRRELEVANT)):
    print(f"\n{label} (best distance per question, lower = closer)")
    for question in sorted(questions, key=best_distance):
        print(f"  {best_distance(question):.3f}  {question}")