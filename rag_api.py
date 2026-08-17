from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_all_documents, chunk_all_documents, embed_chunks, find_similar_chunks, ask




app = FastAPI()


#defining the data models(pydantic)

class QueryRequest(BaseModel):
    """ the query that user sends """
    question: str

class DocsResponse(BaseModel):
    """ what repsonse api sends back with related document """
    answer: str
    sources: list[str]


print("loading documents..........")
documents = load_all_documents("documents")
chunks, sources = chunk_all_documents(documents)
embeddings = embed_chunks(chunks)
doc_names = [name for name, _ in documents]
print(f"✅ Loaded {len(documents)} documents, {len(chunks)} chunks")



# defining end points 

@app.get("/")
def home():
    """ rag-bot api is running"""
    return {"status" : "running", "message" : " rag-bot API is live"}

@app.get("/documents")
def list_documents():
    """ return the relevant documents according to the query"""
    return {"total": len(doc_names), "documents": doc_names }


@app.post("/search")
def send_answer(request: QueryRequest):
    """ search for answer from loaded documents"""
    relevant_chunks, relevant_sources = find_similar_chunks(
          request.question, chunks, sources, embeddings
      )

    answer = ask(request.question, relevant_chunks)
 
    unique_sources = list(set(relevant_sources))
    return {"answer": answer, "sources": unique_sources}