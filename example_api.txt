# ============================================
# EXAMPLE FastAPI APP — USE THIS AS REFERENCE
# ============================================
# This is a simple book store API to show you the pattern.
# Your RAG-bot API will follow the SAME structure.

# --- STEP 1: IMPORTS ---
from fastapi import FastAPI
from pydantic import BaseModel


# --- STEP 2: CREATE THE APP ---
app = FastAPI()


# --- STEP 3: DEFINE DATA MODELS (Pydantic) ---
# These define what data the client sends and receives.
# Think of them as "forms" that must be filled correctly.

class BookRequest(BaseModel):
    """What the client sends when searching for a book."""
    title: str

class BookResponse(BaseModel):
    """What the API sends back."""
    title: str
    author: str
    summary: str


# --- STEP 4: FAKE DATA (in your RAG bot, this will be your documents) ---
books_db = [
    {"title": "Python Basics", "author": "Vamsi", "summary": "Learn Python from scratch"},
    {"title": "AI Engineering", "author": "Andrew Ng", "summary": "Build AI applications"},
    {"title": "Docker Guide", "author": "DevOps Team", "summary": "Containers explained"},
]


# --- STEP 5: DEFINE ENDPOINTS ---

# GET endpoint — just returns info, no data needed from client
@app.get("/")
def home():
    """Health check — confirms API is running."""
    return {"status": "running", "message": "Book Store API is live!"}


# GET endpoint — returns list of all books
@app.get("/books")
def list_books():
    """Returns all books in the store."""
    return {"total": len(books_db), "books": books_db}


# POST endpoint — client sends data, API processes and responds
@app.post("/search")
def search_book(request: BookRequest):
    """Search for a book by title."""
    # request.title = what the client sent
    query = request.title.lower()

    # Search through our "database"
    results = [book for book in books_db if query in book["title"].lower()]

    if results:
        return {"found": True, "results": results}
    else:
        return {"found": False, "message": f"No book found matching '{request.title}'"}


# --- HOW TO RUN ---
# Save this file, then run:
#   uvicorn example_api:app --reload
#
# Then open browser:
#   http://localhost:8000        → health check
#   http://localhost:8000/books  → see all books
#   http://localhost:8000/docs   → interactive docs (test POST here)


# ============================================
# ARCHITECTURE FOR YOUR RAG-BOT API:
# ============================================
#
# Your RAG-bot API will look like this:
#
# IMPORTS:
#   - fastapi, pydantic
#   - your rag functions (load_document, chunk_text, embed_chunks, find_similar_chunks, ask)
#
# STARTUP:
#   - Load all documents from documents/ folder
#   - Chunk them
#   - Embed them
#   - Store chunks, sources, embeddings in memory (global variables)
#
# ENDPOINTS:
#   GET  /              → "RAG API is running"
#   GET  /documents     → list of loaded document names
#   POST /ask           → client sends {"question": "..."} → returns {"answer": "...", "sources": [...]}
#
# PYDANTIC MODELS:
#   class QuestionRequest(BaseModel):
#       question: str
#
#   class AnswerResponse(BaseModel):
#       answer: str
#       sources: list[str]
#
# THE KEY DIFFERENCE FROM TERMINAL VERSION:
#   Terminal:  input() → print()
#   API:      POST request → JSON response
#
# Everything else (chunking, embedding, searching, asking LLM) stays EXACTLY the same.
# You're just replacing the input/output layer.
