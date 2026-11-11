from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL
from autonomous_rag.vector_store import search
#This Python file all about given the system prompt 
#and creating a function for the LLM model 
#Turning a long chain document list into a proper string 
#Creating a function for receiving the answers when a user asks a question 

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
context provided below.

Rules:
- If the context does not contain the answer, say: "I couldn't find that in the uploaded documents."
- Do not make up information or use outside knowledge.
- Be clear and concise.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)


def format_context(chunks: list[Document]) -> str:
    """Turn retrieved chunks into one text block for the prompt."""
    return "\n\n".join(
        f"[Source: {c.metadata.get('source', 'unknown')}, page {c.metadata.get('page', '?')}]\n"
        f"{c.page_content}"
        for c in chunks
    )


def answer_question(question: str) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks, then ask the LLM to answer from them."""
    chunks = search(question)

    if not chunks:
        return "The knowledge base is empty. Please upload a PDF first.", []

    chain = prompt | get_llm()
    response = chain.invoke({"context": format_context(chunks), "question": question})
    return response.content, chunks