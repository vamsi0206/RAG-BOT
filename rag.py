

# --- IMPORTS ---
import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


# --- SETUP ---
load_dotenv()
client = Groq()

# Load the embedding model (runs locally on your machine)
# "all-MiniLM-L6-v2" converts any text into a vector of 384 numbers
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")


# --- PART 1: LOAD DOCUMENT ---
# Reads a PDF or text file and returns all the text as one string

def load_document(file_path):
    if file_path.endswith(".pdf"):
        # Read PDF: go through each page and extract text
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        # Read text file: just open and read everything
        with open(file_path, "r") as f:
            text = f.read()
    return text


# --- PART 2: CHUNK THE TEXT ---
# Split the document into smaller pieces
# chunk_size = how many words per chunk
# overlap = how many words shared between consecutive chunks

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()              # Split text into list of words
    chunks = []                        # Empty list to store chunks

    # Loop through words, stepping by (chunk_size - overlap)
    for i in range(0, len(words), chunk_size - overlap):
        # Take 'chunk_size' words starting from position i
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# --- PART 3: EMBED THE CHUNKS ---
# Convert each chunk into a vector (array of 384 numbers)
# This is done ONCE when the document is loaded

def embed_chunks(chunks):
    # .encode() takes a list of strings and returns a list of vectors
    embeddings = embedding_model.encode(chunks)
    return embeddings


# --- PART 4: SEARCH FOR RELEVANT CHUNKS ---
# Given a question, find the most similar chunks

def find_similar_chunks(question, chunks, embeddings, top_k=3):
    # Step 1: Convert question to a vector
    question_embedding = embedding_model.encode([question])[0]

    # Step 2: Calculate similarity between question and every chunk
    # np.dot = dot product (higher number = more similar)
    similarities = np.dot(embeddings, question_embedding)

    # Step 3: Get indices of top_k most similar chunks
    # argsort gives indices from lowest to highest, so we take the last 'top_k'
    # [::-1] reverses it so highest similarity comes first
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Step 4: Return the actual text of those chunks
    return [chunks[i] for i in top_indices]


# --- PART 5: GENERATE ANSWER ---
# Send the relevant chunks + question to the LLM

def ask(question, relevant_chunks):
    # Join the chunks into one context string
    context = "\n\n".join(relevant_chunks)

    # Build the prompt — tell the LLM to only use provided information
    prompt = f"""Based on the following information, answer the question.
If the answer is not in the provided information, say "I don't have enough information to answer that."

Information:
{context}

Question: {question}

Answer:"""

    # Send to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,       # Low temperature = factual, less creative
        max_tokens=1024
    )

    return response.choices[0].message.content


# --- LOAD ALL DOCUMENTS FROM A FOLDER ---
# Scans the folder and loads every .pdf and .txt file

def load_all_documents(folder_path):
    """Load all documents from a folder. Returns list of (filename, text) pairs."""
    documents = []

    # Loop through every file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Only process .pdf and .txt files
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            try:
                text = load_document(file_path)
                documents.append((filename, text))
                print(f"      ✅ Loaded: {filename}")
            except Exception as e:
                print(f"      ❌ Failed: {filename} ({e})")

    return documents


# --- CHUNK ALL DOCUMENTS ---
# Chunks every document and tracks which file each chunk came from

def chunk_all_documents(documents, chunk_size=200, overlap=30):
    """Chunk all documents. Returns (chunks_list, sources_list).
    sources_list tracks which file each chunk belongs to."""
    all_chunks = []
    all_sources = []

    for filename, text in documents:
        chunks = chunk_text(text, chunk_size, overlap)
        all_chunks.extend(chunks)
        # Track the source file for each chunk
        all_sources.extend([filename] * len(chunks))

    return all_chunks, all_sources


# --- MAIN FUNCTION ---
# Ties everything together

def main():
    print("=" * 50)
    print("  DOCUMENT Q&A BOT (RAG)")
    print("  Loads ALL documents from documents/ folder")
    print("  Type 'quit' to exit")
    print("=" * 50)

    folder_path = "documents"

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return

    # STEP 1: Load all documents from the folder
    print("\n[1/3] Loading documents...")
    documents = load_all_documents(folder_path)

    if not documents:
        print("      No .pdf or .txt files found in documents/ folder!")
        return

    total_words = sum(len(text.split()) for _, text in documents)
    print(f"      Total: {len(documents)} files, {total_words} words")

    # STEP 2: Chunk all documents
    print("\n[2/3] Chunking documents...")
    chunks, sources = chunk_all_documents(documents)
    print(f"      Created {len(chunks)} chunks from {len(documents)} files")

    # STEP 3: Embed all chunks
    print("\n[3/3] Creating embeddings...")
    embeddings = embed_chunks(chunks)
    print(f"      Created {len(embeddings)} embeddings")

    print("\n✅ Ready! Ask questions about your documents.\n")

    # QUERY LOOP: Keep asking questions
    while True:
        question = input("You: ")

        if question.lower() == "quit":
            print("\nGoodbye! 👋")
            break

        if not question.strip():
            continue

        # Find relevant chunks
        relevant_chunks = find_similar_chunks(question, chunks, embeddings)

        # Generate answer using LLM
        answer = ask(question, relevant_chunks)

        print(f"\nAssistant: {answer}\n")


# --- RUN ---
if __name__ == "__main__":
    main()
