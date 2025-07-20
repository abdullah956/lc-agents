import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone as PineconeClient

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

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

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def fetch_real_time_data(query: str) -> str | None:
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

def chat_with_llm(user_input: str):
    real_time = fetch_real_time_data(user_input)
    if real_time:
        return real_time

    memory = st.session_state.memory
    messages = memory.chat_memory.messages
    history = [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages]
    messages_input = history + [{"role": "user", "content": user_input}]
    res = llm.invoke(messages_input)

    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(res.content)
    return res.content

# UI
st.title("🧠 Chat Assistant with Memory")
st.text_input("Session ID", key="session_id_input", value=st.session_state.session_id, on_change=lambda: st.session_state.update({
    "session_id": st.session_state.session_id_input,
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)
}))

user_input = st.text_input("You:", key="user_input")
if user_input:
    response = chat_with_llm(user_input)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)
