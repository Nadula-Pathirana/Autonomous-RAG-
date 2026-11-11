import os

import streamlit as st

from config import UPLOAD_DIR
from autonomous_rag.document_processor import process_pdf
from autonomous_rag.llm import answer_question
from autonomous_rag.vector_store import add_documents

st.set_page_config(page_title="AutoRAG Assistant", page_icon="📚")
st.title("📚 AutoRAG Assistant (V1)")

os.makedirs(UPLOAD_DIR, exist_ok=True)

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

# --- Chat input ---
if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, chunks = answer_question(question)
            st.markdown(answer)

            # Show retrieved chunks, which is very useful while learning/debugging
            if chunks:
                with st.expander("Retrieved chunks"):
                    for c in chunks:
                        st.caption(
                            f"Source: {c.metadata.get('source')}, page {c.metadata.get('page')}"
                        )
                        st.text(c.page_content[:500])
        except Exception as e:
            answer = f"Something went wrong: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})