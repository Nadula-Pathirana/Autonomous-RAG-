import os

import streamlit as st

from config import MAX_HISTORY_MESSAGES, UPLOAD_DIR
from autonomous_rag.agent import run_agent
from autonomous_rag.document_processor import process_pdf
from autonomous_rag.memory import build_history
from autonomous_rag.vector_store import (
    add_documents,
    clear_knowledge_base,
    delete_document,
    document_exists,
    list_documents,
)

st.set_page_config(page_title="AutoRAG Assistant", page_icon="📚",layout="wide")
st.title("AutoRAG Assistant (V4)")

os.makedirs(UPLOAD_DIR, exist_ok=True)

TOOL_LABELS = {
    "search_knowledge_base": "📄 Your documents",
    "search_web": "🌐 Web",
}


def safe_markdown(text: str) -> str:
    """Streamlit treats $...$ as math. Escape dollar signs so prices display normally."""
    return text.replace("$", "\\$")


def render_sources(sources: list[dict]) -> None:
    """Show where the information came from. ✅ = the answer actually cites it."""
    if not sources:
        return

    documents = [s for s in sources if s["type"] == "document"]
    web = [s for s in sources if s["type"] == "web"]

    with st.expander(f"Sources ({len(sources)})"):
        st.caption("✅ = mentioned in the answer. Others were retrieved but not cited.")

        if documents:
            st.markdown("**📄 From your documents**")
            for s in documents:
                mark = "✅ " if s.get("cited") else ""
                st.markdown(safe_markdown(f"{mark}**{s['title']}**, {s['location']}"))
                st.caption(safe_markdown(s["snippet"][:200] + "..."))

        if web:
            st.markdown("**🌐 From the web**")
            for s in web:
                mark = "✅ " if s.get("cited") else ""
                st.markdown(safe_markdown(f"{mark}[{s['title']}]({s['location']})"))
                st.caption(safe_markdown(s["snippet"][:200] + "..."))


def render_steps(steps: list[dict]) -> None:
    """Show which tools the agent used and what they returned."""
    tools_used = []
    for step in steps:
        if step["type"] == "call" and step["tool"] not in tools_used:
            tools_used.append(step["tool"])

    if not tools_used:
        st.caption("Answered without searching.")
        return

    st.caption("Tools used: " + "  ·  ".join(TOOL_LABELS.get(t, t) for t in tools_used))
    with st.expander("Agent steps"):
        for step in steps:
            label = TOOL_LABELS.get(step["tool"], step["tool"])
            if step["type"] == "call":
                st.markdown(f"**➡️ Called** {label}")
                st.json(step["input"])
            else:
                st.markdown(f"**⬅️ Result from** {label}")
                st.text(step["output"][:1500])


# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history = the agent's memory

# --- Sidebar ---
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Process documents") and uploaded_files:
        for file in uploaded_files:
            # Duplicate protection: ask the database, not the session
            if document_exists(file.name):
                st.info(
                    f"{file.name} is already in the knowledge base. "
                    "Delete it first if you want to re-upload it."
                )
                continue

            # Save the upload to disk so pypdf can read it
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            try:
                with st.spinner(f"Processing {file.name}..."):
                    chunks = process_pdf(path)
                    add_documents(chunks)
                st.success(f"{file.name}: {len(chunks)} chunks added.")
            except Exception as e:
                st.error(f"Could not process {file.name}: {e}")

    # --- Knowledge base manager (placed AFTER the upload code, so it shows new files) ---
    st.divider()
    st.header("Knowledge base")
    documents = list_documents()

    if not documents:
        st.caption("No documents yet. Upload a PDF above.")

    for doc in documents:
        name_col, delete_col = st.columns([4, 1])
        name_col.text(f"📄 {doc['name']}\n{doc['chunks']} chunks")
        if delete_col.button("🗑", key=f"delete_{doc['name']}", help=f"Delete {doc['name']}"):
            delete_document(doc["name"])
            saved_file = os.path.join(UPLOAD_DIR, doc["name"])
            if os.path.exists(saved_file):
                os.remove(saved_file)
            st.rerun()

    if documents:
        confirm = st.checkbox("I want to delete ALL documents")
        if st.button("Clear knowledge base", disabled=not confirm):
            clear_knowledge_base()
            for leftover in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, leftover))
            st.rerun()

    st.divider()
    st.header("Conversation")
    st.caption(f"🧠 Remembers the last {MAX_HISTORY_MESSAGES} messages")
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Chat history display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(safe_markdown(msg["content"]))
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_steps(msg.get("steps", []))

# --- Chat input ---
if question := st.chat_input("Ask about your documents, or anything else"):
    # 1. Build the memory from the PREVIOUS messages (before adding this question)
    history = build_history(st.session_state.messages)

    # 2. Now show and store the new question
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(safe_markdown(question))

    # 3. Ask the agent, giving it the history
    with st.chat_message("assistant"):
        answer, steps, sources, failed = "", [], [], False
        try:
            with st.spinner("Thinking..."):
                result = run_agent(question, history)
            answer, steps, sources = result.answer, result.steps, result.sources
            st.markdown(safe_markdown(answer))
            render_sources(sources)
            # render_steps(steps)
        except Exception as e:
            answer = f"Something went wrong: {e}"
            failed = True
            user_msg["error"] = True   # don't remember a question that failed
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": steps,
            "sources": sources,
            "error": failed,
        }
    )