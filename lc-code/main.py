import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as LC_Pinecone  # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

def validate_environment_variables():
    """Validate that all required environment variables are set"""
    missing_vars = []
    
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    
    if not PINECONE_API_KEY:
        missing_vars.append("PINECONE_API_KEY")
    
    if not INDEX_NAME:
        missing_vars.append("PINECONE_INDEX")
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        logger.error("Please check your .env file and ensure these variables are set:")
        for var in missing_vars:
            logger.error(f"  {var}=your_value_here")
        raise ValueError(error_msg)
    
    logger.info("Environment variables validated successfully")
    logger.info(f"INDEX_NAME: {INDEX_NAME}")
    logger.info(f"OPENAI_API_KEY: {'✓' if OPENAI_API_KEY else '✗'}")
    logger.info(f"PINECONE_API_KEY: {'✓' if PINECONE_API_KEY else '✗'}")

# Validate environment variables before proceeding
try:
    validate_environment_variables()
except ValueError as e:
    logger.error("Application startup failed due to missing environment variables")
    sys.exit(1)

app = FastAPI()

# Initialize components with error handling
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    logger.info("LLM and embeddings initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LLM/embeddings: {e}")
    raise e

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if it doesn't
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Match the dimension of OpenAI embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    # Initialize vectorstore
    vectorstore = LC_Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )
    logger.info("Vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Pinecone or vectorstore: {e}")
    raise e

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="answer"
)

# Create the QA chain
try:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    logger.info("QA chain initialized successfully")
except Exception as e:
    logger.error(f"Error initializing QA chain: {e}")
    raise e

class MessageInput(BaseModel):
    message: str
    is_code: bool = False

@app.post("/chat")
async def chat(input: MessageInput):
    try:
        # Validate input
        if not input.message or not input.message.strip():
            return {"error": "Empty message", "response": "Please provide a valid message."}
        
        if input.is_code:
            prompt = f"""
        You are a professional code reviewer.
        
        Please carefully review the following code and return:
        1. Any bugs or issues found with line numbers.
        2. Suggestions to improve code quality and best practices.
        3. A corrected or improved version of the full code.
        
        Wrap fixed code in triple backticks.
        
        Code:
        {input.message}
        """.strip()
        else:
            prompt = input.message

        
        logger.info(f"Processing query: {prompt[:100]}...")
        
        result = qa_chain.invoke({"question": prompt})
        
        # Ensure the answer is not None
        answer = result.get("answer", "I apologize, but I couldn't generate a response.")
        if answer is None:
            answer = "I apologize, but I couldn't generate a response."
        
        return {"response": answer}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {"error": str(e), "response": "An error occurred while processing your request. Please try again."}

@app.get("/health")
async def health_check():
    try:
        # Test the vectorstore connection
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        return {
            "status": "healthy", 
            "vectorstore": "connected",
            "index_name": INDEX_NAME,
            "vector_count": stats.get('total_vector_count', 0)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive data)"""
    return {
        "index_name": INDEX_NAME,
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "model": "gpt-4"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)