# Document Q&A Bot with RAG 📄🤖

A Python-based document question-answering system using Retrieval-Augmented Generation (RAG). Ask questions about any PDF or text file and get accurate answers based only on the document's content.

## What it does

- Load any PDF or text document
- Ask questions in natural language
- Gets answers **only from your document** (no hallucination)
- Uses local embeddings (free, no API cost for search)
- Runs in your terminal

## How it works (RAG Pipeline)

```
SETUP (once):
   Document → Split into chunks → Convert chunks to vectors → Store

QUERY (every question):
   Question → Convert to vector → Find similar chunks → Send chunks + question to LLM → Answer
```

### Step-by-step:

1. **Load** — Reads your PDF or text file
2. **Chunk** — Splits the document into smaller pieces (~200 words each with overlap)
3. **Embed** — Converts each chunk into a vector (array of 384 numbers) using a local AI model
4. **Search** — When you ask a question, finds the 3 most relevant chunks using vector similarity
5. **Generate** — Sends those chunks + your question to the LLM, which answers using only that information

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Groq API | LLM provider (LLaMA 3.3 70B) |
| sentence-transformers | Local embedding model (all-MiniLM-L6-v2) |
| NumPy | Vector similarity calculation |
| PyPDF2 | PDF reading |
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

If that doesn't work, try:

```bash
pip3 install -r requirements.txt
```

Or:

```bash
python3 -m pip install -r requirements.txt
```

**Note:** First install takes a few minutes — it downloads PyTorch and the embedding model.

### Step 3: Get your API key (free)

1. Go to https://console.groq.com
2. Sign up with Google or GitHub
3. Go to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### Step 4: Create your .env file

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Replace `your_key_here` with your actual API key.

### Step 5: Run it

```bash
python3 rag.py
```

It will ask for a document path. Try the included sample:

```
Enter document path: documents/sample.txt
```

Then ask questions!

## Example

```
==================================================
  DOCUMENT Q&A BOT (RAG)
  Type 'quit' to exit
==================================================

Enter document path (e.g., documents/sample.txt): documents/sample.txt

[1/3] Loading document...
      Loaded 512 words
[2/3] Chunking document...
      Created 4 chunks
[3/3] Creating embeddings...
      Created 4 embeddings

✅ Ready! Ask questions about your document.

You: How much does Ericsson invest in R&D?
Assistant: Ericsson invests approximately 20% of its revenue in research and development.

You: Which universities does Ericsson partner with?
Assistant: Ericsson collaborates with MIT, KTH Royal Institute of Technology, and IIT Hyderabad.

You: What is the population of India?
Assistant: I don't have enough information to answer that.

You: quit
Goodbye! 👋
```

## Project Structure

```
RAG-BOT/
├── rag.py              ← main RAG pipeline code
├── documents/          ← put your PDFs or text files here
│   └── sample.txt      ← included sample document
├── requirements.txt    ← Python packages needed
├── .env                ← your API key (NOT pushed to GitHub)
├── .gitignore          ← tells git to ignore .env
└── README.md           ← this file
```

## Using your own documents

1. Put any `.txt` or `.pdf` file in the `documents/` folder
2. Run `python3 rag.py`
3. Enter the path: `documents/your-file.pdf`
4. Ask questions!

## Key Concepts Covered

| Concept | Description |
|---------|-------------|
| RAG | Retrieval-Augmented Generation — search first, then answer |
| Embeddings | Converting text to vectors (numbers) for semantic search |
| Chunking | Splitting documents into searchable pieces |
| Vector Similarity | Finding relevant information using dot product |
| Prompt Engineering | Instructing the LLM to only use provided context |
| Hallucination Prevention | LLM says "I don't know" when info isn't in the document |

## Limitations

- Memory resets when you restart (chunks are not stored permanently)
- Large PDFs take longer to embed
- Token limit means very long chunks might get truncated
- No conversation memory (each question is independent)

## Future Improvements

- Add a vector database (ChromaDB/FAISS) for persistent storage
- Add conversation memory (follow-up questions)
- Web UI with Streamlit or Gradio
- Support for multiple documents at once
- Advanced chunking (by sections/paragraphs instead of word count)

## Built by

[vamsi0206](https://github.com/vamsi0206)
