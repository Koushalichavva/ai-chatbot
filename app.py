# app.py
# Single Responsibility: Handles ONLY the Streamlit UI.
#   - Page config
#   - Chat rendering
#   - User input handling
#   - Delegating to RAGPipeline for answers
#
# Interface Segregation: app.py only imports what it needs.
#   It doesn't know about ChromaDB, embeddings, chunking, or prompts.
#   All of that is behind the RAGPipeline facade.
#
# This file is intentionally thin. If you ever swap Streamlit for FastAPI
# or a CLI, you only rewrite this file — everything else stays intact.

import streamlit as st
from database import load_db
from rag import load_llm, RAGPipeline

# ------------------------------------------------------------------ #
#  Page Setup                                                         #
# ------------------------------------------------------------------ #
st.set_page_config(page_title="HR Assistant", page_icon="👔")
st.title("👔 HR Onboarding Assistant")
st.caption("Ask me anything about HR policies, leave, attendance, and benefits.")

# ------------------------------------------------------------------ #
#  Load Resources (Singleton — loaded once, cached across messages)   #
# ------------------------------------------------------------------ #
db = load_db()
llm = load_llm()
pipeline = RAGPipeline(db=db, llm=llm)  # Dependency Inversion: inject db + llm

# ------------------------------------------------------------------ #
#  Session State — Conversation History                               #
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------ #
#  Render Chat History                                                #
# ------------------------------------------------------------------ #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------ #
#  Handle New User Input                                              #
# ------------------------------------------------------------------ #
if question := st.chat_input("Ask your HR question..."):

    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Get answer from RAGPipeline (Facade — one call does everything)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = pipeline.answer(
                question=question,
                messages=st.session_state.messages
            )
        st.markdown(answer)

    # 3. Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})