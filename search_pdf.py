from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the stored ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Ask a question
question = "What are Koushali's key skills and experiences mentioned in the resume?"

# Search for relevant chunks
results = db.similarity_search(question, k=3)  # top 3 relevant chunks

print(f"Top {len(results)} relevant chunks found:\n")
for i, chunk in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print()