from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1 - Load PDF
loader = PyPDFLoader("docs/Koushali_resume.pdf")
pages = loader.load()

# Step 2 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"Total chunks created: {len(chunks)}")

# Step 3 - Embed and store in ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("PDF embedded and stored in ChromaDB ✅")