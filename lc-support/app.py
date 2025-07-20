import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient
import requests

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

st.set_page_config(page_title="AI Customer Support", page_icon="🤖")
st.title("🤖 AI Customer Support Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something...")
if user_input:
    response = fetch_real_time_data(user_input)
    if not response:
        response = qa_chain.run(user_input)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("ai", response))

for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)
