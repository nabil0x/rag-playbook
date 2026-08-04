"""Decomposition retriever.

Retriever block: split a multi-hop question into independent sub-questions,
retrieve with the original question AND each sub-question, then merge the
deduplicated results. Multi-hop questions need facts from 2+ chunks; plain
top-k returns the single most-similar chunk and the other facts are out of
reach. See Topics/Project-22-HyDE-Decomposition/README.md.
"""

from langchain_core.documents import Document


# The teaching surface of this project. Edit freely. Keep the output
# contract: 2-4 independent sub-questions, one per line, nothing else.
DECOMPOSE_PROMPT = """You are a question decomposer for a RAG system.

Given a multi-hop user question, split it into 2-4 INDEPENDENT sub-questions
whose answers together answer the original question.

Example:
  Question: "Who wrote the book Waiting, and where were they born?"
  Sub-questions:
  Who wrote the book Waiting?
  Where was that author born?

Rules:
- Each sub-question must be answerable on its own by one retrieved chunk.
- Do not include the original question.
- Output only the sub-questions, one per line, nothing else.

Question: {question}
Sub-questions:"""


class DecomposeRetriever:
    """Retrieve per sub-question, merged with original-query results.

    Wraps any retriever exposing ``retrieve(question) -> list[Document]``.
    Retrieving with the original question AND each sub-question pulls every
    fact's chunk into the candidate set. Duplicates (same page content) are
    removed, first occurrence wins, and the final list is cut to ``top_k``.
    """

    def __init__(self, decomposer_llm, retriever, top_k: int = 5):
        # decomposer_llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.openai.OpenAILLM, llms.gemini.GeminiLLM, or a local
        # Ollama wrapper). Imported by the caller, never at module level.
        self.decomposer_llm = decomposer_llm
        self.retriever = retriever
        self.top_k = top_k

    def _decompose(self, question: str) -> list[str]:
        """TODO(Project 22): implement the decomposition step.

        Call self.decomposer_llm.invoke(DECOMPOSE_PROMPT.format(question=question)),
        split the result into lines, strip each one, and drop empty lines.
        Fall back to [question] when nothing usable comes back - retrieving
        with the original question alone is still a valid retrieval.
        """
        raise NotImplementedError(
            "TODO(Project 22): implement DecomposeRetriever._decompose"
        )

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve with the original + sub-questions, dedupe, return top-k."""
        queries = [question] + self._decompose(question)

        merged: list[Document] = []
        seen: set[str] = set()
        for sub in queries:
            for doc in self.retriever.retrieve(sub):
                key = doc.page_content
                if key and key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged[: self.top_k]


if __name__ == "__main__":
    # Fake LLM + fake retriever - no network, runs anywhere.
    class _FakeDecomposerLLM:
        """Returns two fixed sub-questions."""

        def invoke(self, prompt_text: str) -> str:
            return "Who wrote the book Waiting?\nWhere was that author born?"

    class _FakeRetriever:
        """Echoes each query as a document, plus one chunk shared by all."""

        def __init__(self):
            self.questions_seen: list[str] = []

        def retrieve(self, question: str) -> list[Document]:
            self.questions_seen.append(question)
            return [
                Document(page_content="shared context chunk"),
                Document(page_content=f"answer to: {question}"),
            ]

    inner = _FakeRetriever()
    retriever = DecomposeRetriever(_FakeDecomposerLLM(), inner, top_k=4)
    docs = retriever.retrieve("Who wrote Waiting, and where were they born?")

    # All three queries must have been issued: original + two sub-questions.
    assert len(inner.questions_seen) == 3, inner.questions_seen
    assert "Who wrote the book Waiting?" in inner.questions_seen
    assert "Where was that author born?" in inner.questions_seen
    # The shared chunk must be deduplicated (appears exactly once).
    contents = [d.page_content for d in docs]
    assert contents.count("shared context chunk") == 1, contents
    assert len(docs) == 4
    print("OK: retrieve() issued every sub-question and deduplicated the results")
