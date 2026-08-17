# Document Q&A Bot with RAG 📄🤖

A production-ready document question-answering system using Retrieval-Augmented Generation (RAG). Ask questions about any PDF or text file and get accurate answers based only on the document's content.

Available as **CLI**, **Web UI (Streamlit)**, and **REST API (FastAPI)**.

## What it does

- Load any PDF or text documents
- Ask questions in natural language
- Gets answers **only from your documents** (no hallucination)
- Shows which document the answer came from (source tracking)
- Uses local embeddings (free, no API cost for search)
- Multiple interfaces: Terminal, Web UI, API

## How it works (RAG Pipeline)

```
SETUP (once):
   Documents → Split into chunks → Convert to vectors → Store in memory

QUERY (every question):
   Question → Convert to vector → Find similar chunks → Send chunks + question to LLM → Answer + Sources
```

## Three Ways to Use

### 1. Terminal (CLI)

```bash
cd ~/rag-chatbot && python3 rag.py
```

Interactive terminal chat — type questions, get answers.

### 2. Web UI (Streamlit)

```bash
cd ~/rag-chatbot && streamlit run streamlit_app.py
```

Opens a ChatGPT-like web interface at `http://localhost:8501`

Features:
- Chat interface with message history
- Sidebar showing loaded documents
- Source citations on every answer
- Cached document loading (fast after first load)

### 3. REST API (FastAPI)

```bash
cd ~/rag-chatbot && uvicorn rag_api:app --reload
```

API available at `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/documents` | GET | List loaded documents |
| `/ask` | POST | Ask a question, get answer + sources |

Example API call:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Ericsson do?"}'
```

Response:
```json
{
  "answer": "Ericsson is a Swedish multinational networking and telecommunications company...",
  "sources": ["archvision.pdf"]
}
```

Interactive API docs: `http://localhost:8000/docs`

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Groq API | LLM provider (LLaMA 3.3 70B) |
| sentence-transformers | Local embedding model (all-MiniLM-L6-v2) |
| NumPy | Vector similarity calculation |
| PyPDF2 | PDF reading |
| FastAPI | REST API framework |
| Streamlit | Web UI framework |
| python-dotenv | Environment variable management |

## Setup

### Prerequisites

- Python 3.8 or higher
- A free Groq API key

### Step 1: Clone the repo

```bash
git clone https://github.com/vamsi0206/RAG-BOT.git
cd RAG-BOT
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
python3 -m pip install -r requirements.txt
```

**Note:** First install takes a few minutes — downloads PyTorch and the embedding model.

### Step 3: Get your API key (free)

1. Go to https://console.groq.com
2. Sign up with Google or GitHub
3. Go to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### Step 4: Create your .env file

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### Step 5: Add documents

Put any `.pdf` or `.txt` files in the `documents/` folder:

```bash
cp your-file.pdf documents/
```

### Step 6: Run

```bash
# Terminal
python3 rag.py

# Web UI
streamlit run streamlit_app.py

# API
uvicorn rag_api:app --reload
```

## Project Structure

```
RAG-BOT/
├── rag.py              ← core RAG engine (loading, chunking, embedding, search, generation)
├── streamlit_app.py    ← web UI (Streamlit)
├── rag_api.py          ← REST API (FastAPI)
├── example_api.py      ← API reference example
├── documents/          ← put your PDFs and text files here
│   ├── archvision.pdf
│   ├── sample.txt
│   └── ...
├── requirements.txt    ← Python packages
├── .env                ← your API key (NOT pushed to GitHub)
├── .gitignore          ← ignores .env
└── README.md           ← this file
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   INTERFACES                      │
├────────────┬─────────────────┬───────────────────┤
│  Terminal  │   Streamlit UI  │    FastAPI REST    │
│  (rag.py)  │(streamlit_app.py)│   (rag_api.py)   │
└─────┬──────┴────────┬────────┴──────────┬────────┘
      │               │                   │
      └───────────────┼───────────────────┘
                      │
         ┌────────────▼────────────┐
         │     RAG ENGINE (rag.py) │
         ├─────────────────────────┤
         │  load_all_documents()   │
         │  chunk_all_documents()  │
         │  embed_chunks()         │
         │  find_similar_chunks()  │
         │  ask()                  │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │       EXTERNAL SERVICES  │
         ├──────────────────────────┤
         │  Groq API (LLM)         │
         │  SentenceTransformers   │
         │  (local embeddings)     │
         └──────────────────────────┘
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| RAG | Retrieve relevant info first, then generate answer |
| Embeddings | Converting text to vectors for semantic search |
| Chunking | Splitting documents into searchable pieces (200 words, 30 word overlap) |
| Vector Similarity | Finding relevant chunks using dot product |
| Source Tracking | Every chunk knows which file it came from |
| Prompt Engineering | LLM instructed to only use provided context |

## Configuration

Edit the top of `rag.py` to customize:

```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"    # embedding model
LLM_MODEL = "llama-3.3-70b-versatile"         # LLM model
CHUNK_SIZE = 200                               # words per chunk
CHUNK_OVERLAP = 30                             # overlap between chunks
TOP_K_RESULTS = 3                              # number of chunks retrieved
LLM_TEMPERATURE = 0.3                          # lower = more factual
```

## Limitations

- Documents re-embedded on every server restart (no persistent vector store)
- Large PDFs take longer to embed
- Token limit means very long chunks may get truncated
- No conversation memory in API mode (each question is independent)

## Future Improvements

- Add ChromaDB/FAISS for persistent vector storage
- Add conversation memory (follow-up questions)
- File upload via Streamlit UI
- Authentication for the API
- Deploy to cloud (Railway/Render)

## Built by

[vamsi0206](https://github.com/vamsi0206)
