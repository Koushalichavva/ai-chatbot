# database.py
# Single Responsibility: Handles ONLY vector database setup and loading.
# Singleton Pattern (GoF): Only one ChromaDB instance is ever created per session.
#   - Streamlit's @st.cache_resource is the Singleton mechanism here.
#   - Without it, every user message would reload the entire DB — wasteful.

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_PATH,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Single Responsibility: Creates the embedding model.
    Kept private (underscore prefix) — only database.py needs this.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _build_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Single Responsibility: Builds ChromaDB from scratch by:
      1. Loading the PDF
      2. Splitting into chunks
      3. Embedding and storing in ChromaDB

    Dependency Inversion: Accepts `embeddings` as a parameter instead of
    creating it internally — caller controls which embedding model is used.
    """
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DIR
    )
    return db


@st.cache_resource   # <-- Singleton Pattern: Streamlit ensures this runs only once per session.
def load_db() -> Chroma:
    """
    Facade Pattern (GoF): Hides the complexity of:
      - checking if DB exists
      - deciding to build vs load
      - initializing embeddings
    Caller (app.py) just calls load_db() and gets back a ready-to-use DB.

    Liskov Substitution: Returns a Chroma object. If you swap to Pinecone,
    return a Pinecone object — app.py won't need any changes as long as
    .similarity_search() is supported.
    """
    embeddings = _get_embeddings()

    if not os.path.exists(CHROMA_DIR):
        st.info("Setting up knowledge base for first time...")
        return _build_db(embeddings)

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )