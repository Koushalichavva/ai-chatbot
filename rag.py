# rag.py
# Full 6-Stage RAG Pipeline:
#   Stage 1 — Rewrite  : Improve user question before searching
#   Stage 2 — Retrieve : Search ChromaDB for relevant chunks
#   Stage 3 — Rerank   : Re-order chunks by actual relevance
#   Stage 4 — Refine   : Remove noise/irrelevant sentences from chunks
#   Stage 5 — Insert   : Build the final prompt with context + history
#   Stage 6 — Generate : Call LLM and apply fallback strategy
#
# Facade Pattern (GoF): RAGPipeline.answer() is a single clean interface.
#   app.py calls one method — all 6 stages are hidden inside.
#
# Strategy Pattern (GoF): Fallback detection and noise filtering are
#   strategies driven by config.py — swap behaviour without touching logic.
#
# Singleton Pattern (GoF): load_llm() uses @st.cache_resource.

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma

from config import (
    LLM_MODEL,
    TOP_K_RESULTS,
    TOP_K_AFTER_RERANK,
    CONTEXT_WINDOW_MESSAGE_LIMIT,
    FALLBACK_KEYWORDS,
    FALLBACK_MESSAGE,
    NOISE_KEYWORDS,
    SYSTEM_PROMPT
)


@st.cache_resource  # Singleton: LLM loaded only once per session.
def load_llm() -> OllamaLLM:
    return OllamaLLM(model=LLM_MODEL)


class RAGPipeline:
    """
    Facade Pattern: Single interface to the entire 6-stage RAG process.
    Dependency Inversion: Accepts db and llm — doesn't create them.
    """

    def __init__(self, db: Chroma, llm: OllamaLLM):
        self.db = db
        self.llm = llm

    # ------------------------------------------------------------------ #
    #  STAGE 1 — REWRITE                                                  #
    # ------------------------------------------------------------------ #

    def _rewrite_question(self, question: str) -> str:
        """
        Stage 1 — Rewrite:
        User questions are often vague or short (e.g. "leaves?", "salary?").
        A vague question → vague ChromaDB search → irrelevant chunks returned.

        FIX: Ask the LLM to rewrite the question into a clear, detailed
        version BEFORE sending it to ChromaDB.

        Example:
          Input : "leaves?"
          Output: "How many annual and sick leaves does an employee get per year?"

        The rewritten question is used ONLY for ChromaDB search.
        The original question is still shown to the user and sent to the LLM.

        WHY a separate LLM call here?
        Because ChromaDB does semantic search — better query = better chunks.
        This one small LLM call can dramatically improve retrieval quality.
        """
        rewrite_prompt = f"""You are a search query optimizer for an HR policy assistant.
Rewrite the following user question into a clear, specific search query.
Return ONLY the rewritten question. No explanation, no preamble.

Original question: {question}
Rewritten question:"""

        rewritten = self.llm.invoke(rewrite_prompt).strip()

        # Safety fallback: if LLM returns empty or garbage, use original
        if not rewritten or len(rewritten) > 300:
            return question

        return rewritten

    # ------------------------------------------------------------------ #
    #  STAGE 2 — RETRIEVE                                                 #
    # ------------------------------------------------------------------ #

    def _retrieve_chunks(self, search_query: str) -> list:
        """
        Stage 2 — Retrieve:
        Search ChromaDB using the rewritten question.
        Returns raw Document objects (not joined strings yet) so
        Rerank and Refine stages can process them individually.

        Note: We fetch TOP_K_RESULTS (5) here — more than we'll use —
        because Rerank will trim to the best TOP_K_AFTER_RERANK (3).
        Fetching more gives Rerank more to work with.

        Also combines last assistant response for follow-up awareness
        (same improvement from before — preserved here).
        """
        return self.db.similarity_search(search_query, k=TOP_K_RESULTS)

    def _build_search_query(self, rewritten: str, messages: list) -> str:
        """
        Combines rewritten question with last assistant response
        for better follow-up question retrieval.
        """
        if len(messages) > 1:
            last_response = next(
                (m["content"] for m in reversed(messages[:-1])
                 if m["role"] == "assistant"), ""
            )
            return f"{last_response} {rewritten}"
        return rewritten

    # ------------------------------------------------------------------ #
    #  STAGE 3 — RERANK                                                   #
    # ------------------------------------------------------------------ #

    def _rerank_chunks(self, chunks: list, question: str) -> list:
        """
        Stage 3 — Rerank:
        ChromaDB returns chunks ordered by vector similarity — but vector
        similarity ≠ actual relevance to the user's question.

        Example problem:
          ChromaDB returns: [chunk_about_IT, chunk_about_leave, chunk_about_salary]
          But user asked about leave — chunk_about_leave should be first!

        FIX: Score each chunk by counting how many words from the question
        appear in the chunk. Higher score = more relevant = ranked higher.

        This is a lightweight lexical reranker (no extra model needed).
        Production systems use cross-encoders (e.g. ms-marco) for this,
        but for a local Ollama setup, this works well without extra cost.

        Result: chunks are re-sorted so the most relevant ones come first.
        Then we trim to TOP_K_AFTER_RERANK (3) — best 3 only go to LLM.
        """
        question_words = set(question.lower().split())

        def relevance_score(chunk):
            chunk_words = set(chunk.page_content.lower().split())
            # Score = number of question words found in this chunk
            return len(question_words & chunk_words)

        reranked = sorted(chunks, key=relevance_score, reverse=True)
        return reranked[:TOP_K_AFTER_RERANK]   # Keep only the top 3

    # ------------------------------------------------------------------ #
    #  STAGE 4 — REFINE                                                   #
    # ------------------------------------------------------------------ #

    def _refine_chunks(self, chunks: list) -> str:
        """
        Stage 4 — Refine:
        Retrieved chunks often contain noise — sentences unrelated to HR
        policies mixed in (e.g. "Contact IT for laptop issues").

        Sending noise to the LLM wastes context window tokens and can
        confuse the model into generating irrelevant answers.

        FIX: Filter out sentences containing NOISE_KEYWORDS (from config.py).
        Then deduplicate — chunks sometimes overlap due to chunk_overlap=50,
        so identical sentences appear multiple times.

        Before refine:
          "Employees get 10 leaves. Contact IT for laptop issues. Leave policy applies..."
        After refine:
          "Employees get 10 leaves. Leave policy applies..."

        Strategy Pattern: NOISE_KEYWORDS is the "strategy" — change it in
        config.py to change what gets filtered, no code changes needed.
        """
        seen = set()
        refined_chunks = []

        for chunk in chunks:
            sentences = chunk.page_content.split(". ")
            clean_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Skip noise sentences
                if any(noise in sentence.lower() for noise in NOISE_KEYWORDS):
                    continue

                # Skip duplicate sentences (from chunk overlap)
                if sentence in seen:
                    continue

                seen.add(sentence)
                clean_sentences.append(sentence)

            if clean_sentences:
                refined_chunks.append(". ".join(clean_sentences))

        return "\n\n".join(refined_chunks)

    # ------------------------------------------------------------------ #
    #  STAGE 5 — INSERT                                                   #
    # ------------------------------------------------------------------ #

    def _build_history(self, messages: list) -> str:
        """
        Context Window Management — trims history to last N messages.
        See earlier explanation in previous version of this file.
        """
        recent = messages[-CONTEXT_WINDOW_MESSAGE_LIMIT:-1]
        history = ""
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        return history

    def _insert_into_prompt(self, question: str, context: str, history: str) -> str:
        """
        Stage 5 — Insert:
        Assembles the final prompt from all pieces:
          - System prompt (role + rules)
          - Conversation history (context window managed)
          - Refined context (from ChromaDB, reranked + cleaned)
          - Original user question

        This is now a DEDICATED stage — not mixed with retrieval or LLM logic.
        Single Responsibility: only builds the prompt string, nothing else.

        Note: We insert the ORIGINAL question here (not the rewritten one).
        The rewritten question was only for ChromaDB search quality.
        """
        return f"""{SYSTEM_PROMPT}

Conversation so far:
{history}

Context from HR Policy:
{context}

Question: {question}"""

    # ------------------------------------------------------------------ #
    #  STAGE 6 — GENERATE                                                 #
    # ------------------------------------------------------------------ #

    def _generate_answer(self, prompt: str) -> str:
        """
        Stage 6 — Generate:
        Sends the final assembled prompt to the LLM and gets the answer.
        Applies fallback strategy if the answer signals uncertainty.

        Strategy Pattern: FALLBACK_KEYWORDS drives detection — change in
        config.py to change behaviour without touching this method.
        """
        raw_answer = self.llm.invoke(prompt)

        if any(phrase in raw_answer.lower() for phrase in FALLBACK_KEYWORDS):
            return FALLBACK_MESSAGE

        return raw_answer

    # ------------------------------------------------------------------ #
    #  PUBLIC INTERFACE — The Facade                                      #
    # ------------------------------------------------------------------ #

    def answer(self, question: str, messages: list) -> str:
        """
        Facade: One method call runs all 6 stages in order.

        Stage 1 — Rewrite  : question → better search query
        Stage 2 — Retrieve : search ChromaDB with rewritten query
        Stage 3 — Rerank   : re-order chunks by relevance, trim to top 3
        Stage 4 — Refine   : remove noise + duplicates from chunks
        Stage 5 — Insert   : assemble final prompt
        Stage 6 — Generate : call LLM, apply fallback, return answer

        app.py calls only this — it knows nothing about the 6 stages.
        """
        # Stage 1 — Rewrite
        rewritten = self._rewrite_question(question)

        # Stage 2 — Retrieve
        search_query = self._build_search_query(rewritten, messages)
        chunks = self._retrieve_chunks(search_query)

        # Stage 3 — Rerank
        reranked_chunks = self._rerank_chunks(chunks, question)

        # Stage 4 — Refine
        context = self._refine_chunks(reranked_chunks)

        # Stage 5 — Insert
        history = self._build_history(messages)
        prompt = self._insert_into_prompt(question, context, history)

        # Stage 6 — Generate
        return self._generate_answer(prompt)