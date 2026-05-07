from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

question = "What are Koushali's key skills and experiences mentioned in the resume?"
results = db.similarity_search(question, k=3)
print(f"Top {len(results)} relevant chunks found:\n")
for i, chunk in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print()