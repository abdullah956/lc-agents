# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import PostgresChatMessageHistory, ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.google_search.tool import GoogleSearchAPIWrapper, GoogleSearchRun
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
# Load environment variables from .env file
load_dotenv()

# Validate and assign required environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGDATABASE = os.getenv("PGDATABASE")

if not all([OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE]):
    raise EnvironmentError("Missing one or more required environment variables.")

# Set OpenAI key for LangChain
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# PostgreSQL connection string
POSTGRES_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

# FastAPI app
app = FastAPI()

# Allow all CORS origins (for local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    session_id: str
    question: str

# LangChain setup
def get_agent_with_memory(session_id: str):
    message_history = PostgresChatMessageHistory(
        connection_string=POSTGRES_URL,
        session_id=session_id
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=message_history
    )

    # Google Search Tool with explicit API wrapper
    google_wrapper = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID
    )
    search_tool = GoogleSearchRun(api_wrapper=google_wrapper)

    tools = [
        Tool(
            name="google-search",
            func=search_tool.run,
            description="Useful for answering questions using Google Search"
        )
    ]

    llm = ChatOpenAI(temperature=0)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    return agent

# API route
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        logging.info(f"Received question: {request.question}")
        agent = get_agent_with_memory(request.session_id)
        response = agent.run(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))