import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
import streamlit as st

SYSTEM_PROMPT = """You are an HR Onboarding and Policy Explanation Assistant.
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

def build_db():
    loader = PyPDFLoader("docs/hr_policy.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )
    return db

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists("chroma_db"):
        st.info("Setting up knowledge base for first time...")
        return build_db()
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3.1")

st.set_page_config(page_title="HR Assistant", page_icon="👔")
st.title("👔 HR Onboarding Assistant")
st.caption("Ask me anything about HR policies, leave, attendance, and benefits.")

db = load_db()
llm = load_llm()

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

    # Build conversation history
    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    full_prompt = f"""{SYSTEM_PROMPT}

Conversation so far:
{history}

Context from HR Policy:
{context}

Question: {question}"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(full_prompt)

            FALLBACK_KEYWORDS = ["i don't know", "i'm not sure", "not mentioned", "no information"]
            if any(phrase in answer.lower() for phrase in FALLBACK_KEYWORDS):
                answer = "I'm sorry, I don't have enough information to answer that. Please contact the HR department directly for assistance."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})