"""
RAG Bot — Streamlit UI
========================
Web interface for document Q&A using Retrieval-Augmented Generation.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from rag import (
    load_all_documents,
    chunk_all_documents,
    embed_chunks,
    find_similar_chunks,
    ask
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A Bot")
st.markdown("*Ask questions about your documents — powered by RAG*")
st.divider()


# --- LOAD DOCUMENTS (cached so it only runs once) ---
@st.cache_resource(show_spinner="Loading documents & generating embeddings...")
def initialize_rag():
    """Load, chunk, and embed all documents. Runs once and caches."""
    documents = load_all_documents("documents")
    if not documents:
        return None, None, None, None

    chunks, sources = chunk_all_documents(documents)
    embeddings = embed_chunks(chunks)
    doc_names = [name for name, _ in documents]

    return chunks, sources, embeddings, doc_names


# Initialize
chunks, sources, embeddings, doc_names = initialize_rag()


# --- CHECK IF DOCUMENTS LOADED ---
if chunks is None:
    st.error("❌ No documents found. Add .pdf or .txt files to the `documents/` folder.")
    st.stop()


# --- SIDEBAR: Document Info ---
with st.sidebar:
    st.header("📂 Loaded Documents")
    st.metric("Total Documents", len(doc_names))
    st.metric("Total Chunks", len(chunks))
    st.divider()

    for i, name in enumerate(doc_names, 1):
        st.markdown(f"{i}. `{name}`")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Your documents are chunked & embedded
    2. You ask a question
    3. Most relevant chunks are found (semantic search)
    4. LLM generates an answer using those chunks
    5. Sources are shown for verification
    """)


# --- CHAT INTERFACE ---
# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.caption(f"📄 Sources: {', '.join(message['sources'])}")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            relevant_chunks, relevant_sources = find_similar_chunks(
                question, chunks, sources, embeddings
            )
            answer = ask(question, relevant_chunks)
            unique_sources = list(dict.fromkeys(relevant_sources))

        st.write(answer)
        st.caption(f"📄 Sources: {', '.join(unique_sources)}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": unique_sources
    })
