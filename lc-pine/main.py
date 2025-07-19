import os

from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "lc-pine")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise ValueError("Set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT env vars")

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

app = FastAPI(title="PDF Search Assistant")

class UploadResponse(BaseModel):
    detail: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

def ingest_pdf(file_path: str, namespace: str = None):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(texts))]
    vectorstore.add_texts(texts, metadatas=metadatas, namespace=namespace)
    return len(texts)

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")
    os.makedirs("uploads", exist_ok=True)
    path = f"./uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    count = ingest_pdf(path)
    return {"detail": f"Ingested {count} chunks from {file.filename}"}

@app.post("/query", response_model=QueryResponse)
async def query_pdf(body: QueryRequest):
    answer = qa_chain.run(body.query)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "📄 AI PDF Search Assistant — Upload to /upload, query via /query"}
