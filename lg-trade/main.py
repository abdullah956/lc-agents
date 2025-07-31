import os
import aiohttp
from typing import TypedDict, Dict, Any
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.schema import HumanMessage

from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHA_API_KEY")

# Initialize FastAPI
app = FastAPI(title="AI Trading Assistant")

# ✅ Initialize Pinecone client and connect to existing index
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(PINECONE_INDEX)

# ✅ LangChain VectorStore
vectorstore = LangchainPinecone(
    index_name=PINECONE_INDEX,
    embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    text_key="text"
)

# ✅ OpenAI Chat Model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.2)

# LangGraph state
class GraphState(TypedDict):
    stock_data: Dict[str, Any]
    insights: str
    messages: list

# ✅ Async stock fetch function
async def fetch_stock_price(ticker: str) -> Dict[str, Any]:
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"AlphaVantage error: {await response.text()}")
            data = await response.json()

    quote = data.get("Global Quote", {})
    if not quote or "05. price" not in quote:
        raise ValueError(f"Invalid or missing data for ticker '{ticker}' from AlphaVantage")

    return {
        "symbol": quote.get("01. symbol", ticker.upper()),
        "price": float(quote.get("05. price")),
        "timestamp": quote.get("07. latest trading day")
    }

# ✅ Node 1: Fetch stock info (sync wrapper for async call)
def fetch_node(state: GraphState) -> GraphState:
    import asyncio
    ticker = state["messages"][-1].content.strip().upper()
    stock_data = asyncio.run(fetch_stock_price(ticker))
    return {**state, "stock_data": stock_data}

# ✅ Node 2: Generate LLM-based insight
def insight_node(state: GraphState) -> GraphState:
    stock = state["stock_data"]
    prompt = (
        f"Provide a brief trading insight for:\n"
        f"Symbol: {stock['symbol']}\n"
        f"Current Price: ${stock['price']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "insights": response.content}

# ✅ Build LangGraph
graph_builder = StateGraph(GraphState)
graph_builder.add_node("fetch", fetch_node)
graph_builder.add_node("insight", insight_node)
graph_builder.set_entry_point("fetch")
graph_builder.add_edge("fetch", "insight")
graph_builder.add_edge("insight", END)
graph = graph_builder.compile()

# ✅ FastAPI endpoint
@app.get("/analyze")
async def analyze(ticker: str = Query(..., description="Stock ticker like AAPL or TSLA")):
    try:
        initial_state = {
            "messages": [HumanMessage(content=ticker)],
            "stock_data": {},
            "insights": ""
        }
        import asyncio
        result = await asyncio.to_thread(graph.invoke, initial_state)
        return JSONResponse(content={
            "ticker": result["stock_data"]["symbol"],
            "price": result["stock_data"]["price"],
            "insight": result["insights"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})