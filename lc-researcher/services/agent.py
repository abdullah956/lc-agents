from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import PostgresChatMessageHistory, ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.google_search.tool import GoogleSearchAPIWrapper, GoogleSearchRun
from config.settings import load_environment

def get_agent_with_memory(session_id: str):
    """Initialize and return a LangChain agent with memory."""
    config = load_environment()
    
    # Initialize PostgreSQL chat message history
    message_history = PostgresChatMessageHistory(
        connection_string=config["POSTGRES_URL"],
        session_id=session_id
    )
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=message_history
    )
    
    # Initialize Google Search tool
    google_wrapper = GoogleSearchAPIWrapper(
        google_api_key=config["GOOGLE_API_KEY"],
        google_cse_id=config["GOOGLE_CSE_ID"]
    )
    search_tool = GoogleSearchRun(api_wrapper=google_wrapper)
    
    tools = [
        Tool(
            name="google-search",
            func=search_tool.run,
            description="Useful for answering questions using Google Search"
        )
    ]
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0)
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    return agent