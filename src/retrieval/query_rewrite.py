"""Query rewrite retriever.

Retriever block: rewrite the user's question with an LLM before embedding,
so vague or conversational queries embed into the right region of vector
space. See Topics/Project-21-Query-Rewrite/README.md.
"""

from langchain_core.documents import Document


# The teaching surface of this project. Edit freely. Keep the output
# contract: exactly one standalone question, no extra prose.
REWRITE_PROMPT = """You are a search query rewriter for a RAG system.

Given the original user question, produce ONE standalone, specific search
query that would find the most relevant passages in a document store.

Rules:
- Resolve pronouns and conversational references ("it", "the second one").
- Keep the same language as the original question.
- Output only the rewritten query, nothing else.

Original question: {question}
Rewritten query:"""


class QueryRewriteRetriever:
    """Retrieve by embedding an LLM-rewritten version of the question.

    Wraps any retriever exposing ``retrieve(question) -> list[Document]``
    (e.g. a ``SimilarityRetriever`` from Project 01). Only the rewritten
    query reaches the inner retriever, so the vector store is untouched.
    """

    def __init__(self, rewriter_llm, retriever, top_k: int = 5):
        # rewriter_llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.openai.OpenAILLM, llms.gemini.GeminiLLM, or a local
        # Ollama wrapper). Imported by the caller, never at module level.
        self.rewriter_llm = rewriter_llm
        self.retriever = retriever
        self.top_k = top_k

    def _rewrite(self, question: str) -> str:
        """Rewrite the question with the LLM, falling back to the original.

        Calls ``self.rewriter_llm.invoke(REWRITE_PROMPT.format(question=question))``,
        strips the result, and falls back to the original question when the
        output is empty or unusable. That fallback is the safety net a real
        system needs: a broken LLM call must never take retrieval down with it.
        """
        try:
            out = self.rewriter_llm.invoke(
                REWRITE_PROMPT.format(question=question)
            )
        except Exception:
            return question
        out = (out or "").strip()
        return out if out else question

    def retrieve(self, question: str) -> list[Document]:
        """Rewrite the question, then delegate retrieval to the inner retriever."""
        rewritten = self._rewrite(question)
        return self.retriever.retrieve(rewritten)[: self.top_k]


if __name__ == "__main__":
    # Fake LLM + fake retriever — no network, runs anywhere.
    class _FakeRewriterLLM:
        """Replaces the question with a fixed 'better' question."""

        def invoke(self, prompt_text: str) -> str:
            return "What is the capital of France?"

    class _FakeRetriever:
        """Records the question it received and echoes it back as a document."""

        def __init__(self):
            self.last_question = None

        def retrieve(self, question: str) -> list[Document]:
            self.last_question = question
            return [Document(page_content=f"answer to: {question}")]

    inner = _FakeRetriever()
    retriever = QueryRewriteRetriever(_FakeRewriterLLM(), inner, top_k=1)
    docs = retriever.retrieve("what about it?")

    # The inner retriever must have seen the REWRITTEN query, not the raw one.
    assert inner.last_question == "What is the capital of France?", inner.last_question
    assert docs[0].page_content == "answer to: What is the capital of France?"
    print("OK: retrieve() embedded the rewritten query, not the raw one")
