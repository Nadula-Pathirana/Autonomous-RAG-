import os
from pathlib import Path

import streamlit as st

from config import (
    ENABLE_STREAMING,
    MAX_HISTORY_MESSAGES,
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_MB,
    UPLOAD_DIR,
    check_environment,
)
from autonomous_rag.agent import run_agent, stream_agent
from autonomous_rag.document_processor import process_pdf
from autonomous_rag.exceptions import DocumentProcessingError, friendly_error
from autonomous_rag.logger import get_logger
from autonomous_rag.memory import build_history
from autonomous_rag.vector_store import (
    add_documents,
    clear_knowledge_base,
    delete_document,
    document_exists,
    list_documents,
)

logger = get_logger("app")

st.set_page_config(page_title="AutoRAG Assistant", page_icon="📚", layout="wide")
st.title("AutoRAG Assistant (V5)")

# --- Startup checks: fail early with a clear message ---
config_errors, config_warnings = check_environment()
for message in config_warnings:
    st.warning(message)
if config_errors:
    for message in config_errors:
        st.error(message)
    st.info("Add the missing keys to your .env file, then restart the app.")
    st.stop()

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
                if s.get("score") is not None:
                    st.caption(f"distance {s['score']} (lower = closer match)")
                st.caption(safe_markdown(s["snippet"][:200] + "..."))

        if web:
            st.markdown("**🌐 From the web**")
            for s in web:
                mark = "✅ " if s.get("cited") else ""
                st.markdown(safe_markdown(f"{mark}[{s['title']}]({s['location']})"))
                st.caption(safe_markdown(s["snippet"][:200] + "..."))


def render_steps(steps: list[dict], show_details: bool) -> None:
    """Show which tools the agent used (and optionally what they returned)."""
    tools_used = []
    for step in steps:
        if step["type"] == "call" and step["tool"] not in tools_used:
            tools_used.append(step["tool"])

    if not tools_used:
        st.caption("Answered without searching.")
        return

    st.caption("Tools used: " + "  ·  ".join(TOOL_LABELS.get(t, t) for t in tools_used))
    if show_details:
        with st.expander("Agent steps"):
            for step in steps:
                label = TOOL_LABELS.get(step["tool"], step["tool"])
                if step["type"] == "call":
                    st.markdown(f"**➡️ Called** {label}")
                    st.json(step["input"])
                else:
                    st.markdown(f"**⬅️ Result from** {label}")
                    st.text(step["output"][:1500])


def discard_upload(name: str, path: str) -> None:
    """Clean up after a failed upload: remove the saved file and any partial chunks."""
    try:
        delete_document(name)
    except Exception:
        logger.exception("Rollback failed for %s", name)
    if os.path.exists(path):
        os.remove(path)


def ingest_file(file) -> None:
    """Validate, process and store one uploaded PDF. Never leaves half a document behind."""
    name = Path(file.name).name   # strip any folder parts from the name

    if document_exists(name):
        st.info(f"{name} is already in the knowledge base. Delete it first to re-upload it.")
        return

    size_mb = file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"{name} is {size_mb:.1f} MB. The limit is {MAX_UPLOAD_MB} MB.")
        return

    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    progress = st.progress(0.0, text=f"Reading {name}...")
    try:
        chunks = process_pdf(path)
        add_documents(
            chunks,
            progress_callback=lambda done, total: progress.progress(
                done / total, text=f"Embedding {name}: {done}/{total} chunks"
            ),
        )
        progress.empty()
        st.success(f"{name}: {len(chunks)} chunks added.")
        logger.info("Uploaded %s | chunks=%d", name, len(chunks))
    except DocumentProcessingError as error:
        progress.empty()
        st.error(str(error))
        discard_upload(name, path)
    except Exception as error:
        progress.empty()
        logger.exception("Upload failed for %s", name)
        st.error(f"Could not add {name}. {friendly_error(error)}")
        discard_upload(name, path)


def answer_with_streaming(placeholder, question: str, history: list[dict]):
    """Show the answer as it is written. Returns the final AgentResult."""
    text = ""
    result = None
    placeholder.markdown("_Thinking..._")

    for event in stream_agent(question, history):
        kind = event["event"]
        if kind == "token":
            text += event["text"]
            placeholder.markdown(safe_markdown(text) + "▌")
        elif kind == "tool":
            label = TOOL_LABELS.get(event["name"], event["name"])
            placeholder.markdown(f"_Searching: {label}..._")
        elif kind == "status":                                   
            placeholder.markdown(f"_{event['text']}_")
        elif kind == "reset":
            text = ""
            placeholder.markdown("_Thinking..._")
        elif kind == "done":
            result = event["result"]
    return result


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
            ingest_file(file)

    # --- Knowledge base manager (placed AFTER the upload code, so it shows new files) ---
    st.divider()
    st.header("Knowledge base")
    try:
        documents = list_documents()
    except Exception as error:
        logger.exception("Could not list documents")
        st.error(f"Could not read the knowledge base. {friendly_error(error)}")
        documents = []

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
            logger.info("Deleted document %s", doc["name"])
            st.rerun()

    if documents:
        confirm = st.checkbox("I want to delete ALL documents")
        if st.button("Clear knowledge base", disabled=not confirm):
            clear_knowledge_base()
            for leftover in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, leftover))
            logger.info("Cleared the knowledge base")
            st.rerun()

    st.divider()
    st.header("Conversation")
    st.caption(f"🧠 Remembers the last {MAX_HISTORY_MESSAGES} messages")
    show_steps = st.checkbox("Show agent steps", value=False)
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Chat history display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(safe_markdown(msg["content"]))
        if msg["role"] == "assistant" and not msg.get("error"):
            render_sources(msg.get("sources", []))
            render_steps(msg.get("steps", []), show_steps)

# --- Chat input ---
question = st.chat_input(
    "Ask about your documents, or anything else", max_chars=MAX_QUESTION_CHARS
)
if question is not None and not question.strip():
    st.warning("Please type a question.")
    question = None

if question:
    question = question.strip()

    # 1. Build the memory from the PREVIOUS messages (before adding this question)
    history = build_history(st.session_state.messages)

    # 2. Now show and store the new question
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(safe_markdown(question))

    # 3. Ask the agent, giving it the history
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer, steps, sources, failed = "", [], [], False
        try:
            if ENABLE_STREAMING:
                result = answer_with_streaming(placeholder, question, history)
            else:
                with st.spinner("Thinking..."):
                    result = run_agent(question, history)
            answer, steps, sources = result.answer, result.steps, result.sources
            placeholder.markdown(safe_markdown(answer))
            render_sources(sources)
            render_steps(steps, show_steps)
        except Exception as error:
            logger.exception("Agent failed | question=%r", question[:80])
            answer = friendly_error(error)
            failed = True
            user_msg["error"] = True   # don't remember a question that failed
            placeholder.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": steps,
            "sources": sources,
            "error": failed,
        }
    )