# config.py
# Single Responsibility: Only holds configuration constants.
# Open/Closed: Add new config values here without touching other files.

# --- Paths ---
PDF_PATH = "docs/hr_policy.pdf"
CHROMA_DIR = "chroma_db"

# --- Embedding & LLM ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K_RESULTS = 5          # Fetch more chunks initially — reranker will trim to TOP_K_AFTER_RERANK
TOP_K_AFTER_RERANK = 3    # Keep only the top 3 after reranking

# --- Refine ---
# Sentences containing these keywords are noise — removed during Refine stage.
# Add domain-specific noise phrases here without touching rag.py.
NOISE_KEYWORDS = [
    "contact it",
    "contact admin",
    "refer to the manual",
    "see appendix",
    "for more details contact",
]

# --- Context Window Management ---
# Only the last N messages are sent to LLM to stay within context window limits.
# LLaMA 3.1 has an 8k token context window.
# Each message ≈ ~100-150 tokens on average.
# 6 messages ≈ 600-900 tokens — leaves plenty of room for system prompt + context + answer.
CONTEXT_WINDOW_MESSAGE_LIMIT = 6

# --- Fallback ---
# Strategy Pattern: These keywords drive the fallback detection strategy.
# To change strategy, update this list — no other file needs to change.
FALLBACK_KEYWORDS = [
    "i don't know",
    "i'm not sure",
    "not mentioned",
    "no information"
]
FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information to answer that. "
    "Please contact the HR department directly for assistance."
)

# --- System Prompt ---
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