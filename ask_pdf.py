import os
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Load ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Your question
question = "What are Koushali's key skills and experiences mentioned in the resume?"

# Step 1 - Search relevant chunks
results = db.similarity_search(question, k=3)
context = "\n\n".join([r.page_content for r in results])

# Step 2 - Send to Groq with context
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions using only the context provided."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
)

print("Answer:")
print(response.choices[0].message.content)