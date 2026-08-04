"""Step-back retriever.

Retriever block: abstract the question into a broader "step-back" question,
retrieve with it, then merge with the original-query results. Narrow
questions ("how does tokenization work in the parser?") often miss the
general chunk that actually explains the topic; the step-back question
finds it. See Topics/Project-21-Query-Rewrite/README.md.
"""

from langchain_core.documents import Document


# The teaching surface of this project. Edit freely. Keep the output
# contract: exactly one broader question, no extra prose.
STEPBACK_PROMPT = """You are a search query abstractor for a RAG system.

Given a specific user question, produce ONE broader "step-back" question
that would retrieve the general background material the specific question
depends on.

Example:
  Specific: "How does the recursive character splitter set overlap?"
  Step-back: "How does the recursive character text splitter work?"

Rules:
- The step-back question must be broader, not a paraphrase.
- Keep the same language as the original question.
- Output only the step-back question, nothing else.

Specific question: {question}
Step-back question:"""


class StepBackRetriever:
    """Retrieve with a step-back question, merged with original-query results.

    Wraps any retriever exposing ``retrieve(question) -> list[Document]``.
    Retrieving with the step-back question surfaces general context; merging
    with the original-query results keeps precision. Duplicates (same page
    content) are removed, first occurrence wins, and the final list is cut
    to ``top_k``.
    """

    def __init__(self, stepback_llm, retriever, top_k: int = 5):
        # stepback_llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.openai.OpenAILLM, llms.gemini.GeminiLLM, or a local
        # Ollama wrapper). Imported by the caller, never at module level.
        self.stepback_llm = stepback_llm
        self.retriever = retriever
        self.top_k = top_k

    def _step_back(self, question: str) -> str:
        """TODO(Project 21): implement the step-back generation.

        Call self.stepback_llm.invoke(STEPBACK_PROMPT.format(question=question)),
        strip the result, and fall back to the original question when the
        output is empty. A step-back question identical to the original is
        fine (the merge then just dedupes).
        """
        raise NotImplementedError(
            "TODO(Project 21): implement StepBackRetriever._step_back"
        )

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve with step-back + original queries, dedupe, return top-k."""
        stepback = self._step_back(question)

        merged: list[Document] = []
        seen: set[str] = set()
        for doc in self.retriever.retrieve(stepback) + self.retriever.retrieve(question):
            key = doc.page_content
            if key and key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged[: self.top_k]


if __name__ == "__main__":
    # Fake LLM + fake retriever — no network, runs anywhere.
    class _FakeStepbackLLM:
        """Returns a fixed broader question."""

        def invoke(self, prompt_text: str) -> str:
            return "How does the recursive character text splitter work?"

    class _FakeRetriever:
        """Maps a question to a fixed ranked list."""

        def __init__(self):
            self.questions_seen: list[str] = []

        def retrieve(self, question: str) -> list[Document]:
            self.questions_seen.append(question)
            if "recursive character" in question:  # step-back question
                return [Document(page_content="general splitter doc")]
            return [Document(page_content="specific overlap doc")]

    inner = _FakeRetriever()
    retriever = StepBackRetriever(_FakeStepbackLLM(), inner, top_k=2)
    docs = retriever.retrieve("How does the splitter set overlap?")

    # Both the step-back and the original query must have been issued.
    assert len(inner.questions_seen) == 2, inner.questions_seen
    assert "general splitter doc" in {d.page_content for d in docs}
    assert "specific overlap doc" in {d.page_content for d in docs}
    print("OK: retrieve() merged step-back + original results without duplicates")
