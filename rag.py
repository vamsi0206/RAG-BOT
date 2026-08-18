"""
RAG Engine — Retrieval-Augmented Generation Pipeline
=====================================================
Core module for document ingestion, semantic search, and LLM-powered Q&A.
Supports PDF and text files with multi-document loading and source tracking.
"""

import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# --- CONFIGURATION ---
load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K_RESULTS = 3
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
SUPPORTED_EXTENSIONS = (".pdf", ".txt")

# --- INITIALIZE CLIENTS ---
client = Groq()
print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("[RAG] Model loaded ✅\n")


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document(file_path: str) -> str:
    """Load a single document (PDF or text) and return its content as a string."""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text


def load_all_documents(folder_path: str) -> list[tuple[str, str]]:
    """
    Load all supported documents from a folder.
    Returns: list of (filename, text_content) tuples.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Documents folder not found: {folder_path}")

    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(SUPPORTED_EXTENSIONS):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            text = load_document(file_path)
            if text.strip():
                documents.append((filename, text))
                print(f"  ✅ {filename} ({len(text.split())} words)")
            else:
                print(f"  ⚠️  {filename} (empty, skipped)")
        except Exception as e:
            print(f"  ❌ {filename} — {e}")

    return documents


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of approximately `chunk_size` words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def chunk_all_documents(
    documents: list[tuple[str, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> tuple[list[str], list[str]]:
    """
    Chunk all documents with source tracking.
    Returns: (all_chunks, all_sources) where sources[i] = filename for chunks[i].
    """
    all_chunks = []
    all_sources = []

    for filename, text in documents:
        chunks = chunk_text(text, chunk_size, overlap)
        all_chunks.extend(chunks)
        all_sources.extend([filename] * len(chunks))

    return all_chunks, all_sources


# ============================================================
# EMBEDDING & SEMANTIC SEARCH
# ============================================================

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Convert text chunks into embedding vectors."""
    if not chunks:
        return np.array([])
    return embedding_model.encode(chunks, show_progress_bar=False)


def find_similar_chunks(
    question: str,
    chunks: list[str],
    sources: list[str],
    embeddings: np.ndarray,
    top_k: int = TOP_K_RESULTS
) -> tuple[list[str], list[str]]:
    """
    Find the most semantically similar chunks to a question.
    Returns: (relevant_chunks, relevant_sources) sorted by similarity.
    """
    question_embedding = embedding_model.encode([question])[0]

    # Cosine similarity via dot product (embeddings are normalized by default)
    similarities = np.dot(embeddings, question_embedding)

    # Get top-k indices sorted by highest similarity
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_chunks = [chunks[i] for i in top_indices]
    relevant_sources = [sources[i] for i in top_indices]

    return relevant_chunks, relevant_sources


# ============================================================
# LLM GENERATION
# ============================================================

SYSTEM_PROMPT = """You are a precise document Q&A assistant. Answer questions ONLY using the provided context. 
If the answer cannot be found in the context, say: "I don't have enough information to answer that."
Be concise, accurate, and cite specific details from the context."""

def ask(question: str, relevant_chunks: list[str]) -> str:
    """Generate an answer using retrieved chunks as context."""
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Context:
{context}

---

Question: {question}

Answer based ONLY on the context above:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return response.choices[0].message.content


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    """Run the RAG pipeline as a terminal application."""
    print("=" * 55)
    print("   📄 DOCUMENT Q&A — Retrieval-Augmented Generation")
    print("=" * 55)

    folder_path = "documents"

    # Load
    print("\n[1/3] Loading documents...")
    documents = load_all_documents(folder_path)
    if not documents:
        print("No documents found. Add .pdf or .txt files to documents/ folder.")
        return

    total_words = sum(len(text.split()) for _, text in documents)
    print(f"\n  📊 {len(documents)} documents | {total_words:,} words total")

    # Chunk
    print("\n[2/3] Chunking...")
    chunks, sources = chunk_all_documents(documents)
    print(f"  📊 {len(chunks)} chunks created")

    # Embed
    print("\n[3/3] Generating embeddings...")
    embeddings = embed_chunks(chunks)
    print(f"  📊 {len(embeddings)} vectors generated")

    print("\n" + "=" * 55)
    print("  ✅ Ready! Ask anything about your documents.")
    print("  Type 'quit' to exit.")
    print("=" * 55 + "\n")

    # Query loop
    while True:
        question = input("You: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue

        # Retrieve & Generate
        relevant_chunks, relevant_sources = find_similar_chunks(
            question, chunks, sources, embeddings
        )
        answer = ask(question, relevant_chunks)

        # Display
        print(f"\n🤖 {answer}")
        unique_sources = list(dict.fromkeys(relevant_sources))  # preserves order
        print(f"📄 Sources: {', '.join(unique_sources)}\n")


# --- ENTRY POINT ---
if __name__ == "__main__":
    main()
