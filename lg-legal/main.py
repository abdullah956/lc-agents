import os
import uuid
from typing import TypedDict, Dict, Any, List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV_REGION = os.getenv("PINECONE_REGION")  # e.g., 'us-west-2'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_ENV_REGION:
    raise ValueError("PINECONE_REGION environment variable is not set in .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "lc-legal"

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"[Pinecone] Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV_REGION),
    )

index = pc.Index(index_name)

# Initialize LLM and Embeddings
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# FastAPI app instance
app = FastAPI()

# Define state schema for LangGraph
class StateSchema(TypedDict):
    text: str
    doc_id: str
    insights: List[str]

# Helper: extract raw text from PDF
def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# Helper: split long document into chunks
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Store chunks in Pinecone
def embed_and_store(chunks, doc_id):
    print(f"[Embed] Embedding and storing {len(chunks)} chunks...")
    vectors = embedder.embed_documents(chunks)
    pinecone_vectors = [
        {"id": f"{doc_id}_{i}", "values": vector, "metadata": {"text": chunk, "doc_id": doc_id}}
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    index.upsert(vectors=pinecone_vectors)
    print(f"[Pinecone] Upserted {len(pinecone_vectors)} vectors.")

# Query Pinecone with a real embedding
def query_chunks(doc_id):
    query_text = "Extract legal insights from this contract"
    query_vector = embedder.embed_query(query_text)

    print(f"[Pinecone] Querying with vector for doc_id={doc_id}...")
    results = index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True,
        filter={"doc_id": {"$eq": doc_id}},
    )
    matches = results.get("matches", [])
    print(f"[Pinecone] Retrieved {len(matches)} matching chunks.")
    return [match["metadata"]["text"] for match in matches]

# LangGraph Node: Ingest and embed document
def ingest_node(state: StateSchema) -> StateSchema:
    text = state["text"]
    doc_id = state["doc_id"]
    chunks = chunk_text(text)
    embed_and_store(chunks, doc_id)
    return {"text": text, "doc_id": doc_id, "insights": []}

# LangGraph Node: Extract insights using LLM
def insight_node(state: StateSchema) -> StateSchema:
    doc_id = state["doc_id"]
    template = """Given this legal contract snippet, extract the following:
1. Parties involved
2. Effective date
3. Termination clauses
4. Obligations
5. Renewal terms
6. Governing law

Output the results in a JSON format."""
    
    chunks = query_chunks(doc_id)
    if not chunks:
        print("[LLM] No matching chunks found for this document.")
        return {**state, "insights": []}

    insights = []
    for i, chunk in enumerate(chunks):
        print(f"[LLM] Invoking on chunk {i+1}/{len(chunks)}...")
        response = llm.invoke(template + "\n\n" + chunk)
        insights.append(response.content)

    return {**state, "insights": insights}

# Build LangGraph
workflow = StateGraph(state_schema=StateSchema)
workflow.add_node("ingest", ingest_node)
workflow.add_node("extract", insight_node)
workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "extract")
workflow.add_edge("extract", END)
graph = workflow.compile()

# FastAPI Route
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    if not text.strip():
        return JSONResponse(content={"error": "Uploaded PDF is empty or unreadable."}, status_code=400)

    doc_id = str(uuid.uuid4())
    print(f"[UPLOAD] Received file: {file.filename}, generated doc_id: {doc_id}")
    result = graph.invoke({"text": text, "doc_id": doc_id, "insights": []})
    return JSONResponse(content={"doc_id": doc_id, "insights": result["insights"]})