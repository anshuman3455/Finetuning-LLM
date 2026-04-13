import streamlit as st
from agent.simple_agent import run_agent

st.title("💰 Finance AI Assistant (Local LLM + MCP)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Ask something...")

if user_input:
    st.chat_message("user").write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        response = run_agent(user_input)

    st.chat_message("assistant").write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })