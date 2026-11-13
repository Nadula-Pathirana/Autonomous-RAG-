import os

import streamlit as st

from config import UPLOAD_DIR
from autonomous_rag.agent import run_agent
from autonomous_rag.document_processor import process_pdf
from autonomous_rag.vector_store import add_documents

st.set_page_config(page_title="AutoRAG Assistant", page_icon="📚")
st.title("📚 AutoRAG Assistant (V2)")

os.makedirs(UPLOAD_DIR, exist_ok=True)

TOOL_LABELS = {
    "search_knowledge_base": "📄 Your documents",
    "search_web": "🌐 Web",
}


def render_steps(steps: list[dict]) -> None:
    """Show which tools the agent used and what they returned."""
    tools_used = []
    for step in steps:
        if step["type"] == "call" and step["tool"] not in tools_used:
            tools_used.append(step["tool"])

    if not tools_used:
        st.caption("Answered without searching.")
        return

    st.caption("Sources used: " + "  ·  ".join(TOOL_LABELS.get(t, t) for t in tools_used))
    with st.expander("Agent steps"):
        for step in steps:
            label = TOOL_LABELS.get(step["tool"], step["tool"])
            if step["type"] == "call":
                st.markdown(f"**➡️ Called** {label}")
                st.json(step["input"])
            else:
                st.markdown(f"**⬅️ Result from** {label}")
                st.text(step["output"][:1500])


# Session state setup 
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history for display
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []   # filenames already added this session

# Sidebar: document upload 
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Process documents") and uploaded_files:
        for file in uploaded_files:
            if file.name in st.session_state.processed_files:
                st.info(f"{file.name} was already processed.")
                continue

            # Save the upload to disk so pypdf can read it
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            try:
                with st.spinner(f"Processing {file.name}..."):
                    chunks = process_pdf(path)
                    add_documents(chunks)
                st.session_state.processed_files.append(file.name)
                st.success(f"{file.name}: {len(chunks)} chunks added.")
            except Exception as e:
                st.error(f"Could not process {file.name}: {e}")

    if st.session_state.processed_files:
        st.subheader("Added this session")
        for name in st.session_state.processed_files:
            st.write(f"• {name}")

# --- Chat history display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_steps(msg.get("steps", []))

# --- Chat input ---
if question := st.chat_input("Ask about your documents, or anything else"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        steps = []
        try:
            with st.spinner("Thinking..."):
                answer, steps = run_agent(question)
            st.markdown(answer)
            render_steps(steps)
        except Exception as e:
            answer = f"Something went wrong: {e}"
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )