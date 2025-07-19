import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import load_environment
from services.agent import get_agent_with_memory
from models.query import QueryRequest

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_environment()

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        logging.info(f"Received question: {request.question}")
        agent = get_agent_with_memory(request.session_id)
        response = agent.run(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))