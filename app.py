import os
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.set_page_config(page_title="HR Assistant", page_icon="👔")
st.title("👔 HR Onboarding Assistant")
st.caption("Ask me anything about HR policies, leave, attendance, and benefits.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
if question := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    results = db.similarity_search(question, k=3)
    context = "\n\n".join([r.page_content for r in results])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an HR Onboarding and Policy Explanation Assistant.
Your purpose is to explain company onboarding steps, HR policies,
attendance rules, leave policies, and benefits in a clear and simple manner.
You must follow these rules strictly:
- Provide INFORMATION ONLY.
- Do NOT approve or reject leave, payroll, or benefits.
- Do NOT handle exceptions or personal employee cases.
- Do NOT request or process personal employee data.
- If a request requires approval or HR intervention,
  politely ask the user to contact the HR department.
Your tone must be:
- Professional
- Clear
- Employee-friendly"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)