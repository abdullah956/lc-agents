import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables from .env
load_dotenv()

# Retrieve OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

# Retrieve PostgreSQL connection info from environment variables
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
PGHOST = os.getenv("PGHOST", "127.0.0.1")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "langchainfastapi")

# Construct SQLAlchemy connection string for PostgreSQL
DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

# Executor for running blocking code asynchronously
executor = ThreadPoolExecutor()

# Initialize OpenAI LLM model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=openai_api_key
)

# FastAPI application instance
app = FastAPI(title="LangChain Memory Chat API")

# Request data model
class ChatRequest(BaseModel):
    message: str
    session_id: str

# Response data model
class ChatResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]]

# Function to get memory instance tied to a session ID
def get_memory(session_id: str) -> ConversationBufferMemory:
    try:
        history = PostgresChatMessageHistory(
            connection_string=DATABASE_URL,
            session_id=session_id
        )
        return ConversationBufferMemory(
            return_messages=True,
            chat_memory=history
        )
    except SQLAlchemyError as e:
        raise RuntimeError(f"Database connection error: {e}")

# Health check root endpoint
@app.get("/")
def root():
    return {"message": "LangChain Memory Chat API is running."}

# Chat endpoint: accepts POST with message and session_id, returns AI response
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    session_id = request.session_id.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        memory = get_memory(session_id)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        # Run blocking chain.predict in executor to avoid blocking event loop
        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: chain.predict(input=user_input)
        )

        # Get all messages from memory
        messages = memory.chat_memory.messages
        # Convert messages to dicts for JSON serialization
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Test initialization outside FastAPI for debugging
if __name__ == "__main__":
    print("Database URL:", DATABASE_URL)
    try:
        history = PostgresChatMessageHistory(connection_string=DATABASE_URL, session_id="testsession1")
        print("PostgresChatMessageHistory initialized successfully.")
    except Exception as e:
        print("Error initializing PostgresChatMessageHistory:", e)
