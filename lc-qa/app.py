import streamlit as st
from qa_chain import get_conversational_chain

st.set_page_config(page_title="AI Q&A with Memory", layout="centered")
st.title("🧠 AI-Powered Q&A System with Memory")

# Session state to store the chain and history
if "chain" not in st.session_state:
    st.session_state.chain = get_conversational_chain()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
query = st.text_input("Ask a question:")

# Handle user input
if query:
    response = st.session_state.chain.run(query)
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("AI", response))

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation:")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
