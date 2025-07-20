import os
import re
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WeatherAPI Key hardcoded as requested
WEATHER_API_KEY =os.getenv("WEATHER_API_KEY")

pc = PineconeClient(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="default"
)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.3)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

app = FastAPI()

class Query(BaseModel):
    question: str

def fetch_real_time_data(query: str) -> str:
    if "weather" in query.lower():
        match = re.search(r"weather.*(?:in|at|of)?\s*([a-zA-Z\s]+)", query.lower())
        city = match.group(1).strip() if match else "Lahore"
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            temp = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            return f"{city.title()} weather: {condition}, {temp}°C"
        return "Could not fetch weather info."
    return None

@app.post("/chat")
async def chat(query: Query):
    real_time = fetch_real_time_data(query.question)
    if real_time:
        return {"response": real_time}
    result = qa_chain.run(query.question)
    return {"response": result}
