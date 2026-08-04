"""HyDE retriever (Hypothetical Document Embeddings).

Retriever block: the LLM writes a hypothetical passage that would answer
the question, and we embed THAT passage instead of the query. Hypothetical
document text looks like source-document text, so its embedding lands near
the real chunks even when the question shares no vocabulary with them.
See Topics/Project-22-HyDE-Decomposition/README.md.
"""

from langchain_core.documents import Document


# The teaching surface of this project. Edit freely. Keep the output
# contract: exactly one hypothetical passage, no extra prose.
HYDE_PROMPT = """You are a HyDE generator for a RAG system.

Given a user question, write ONE short hypothetical passage that would
answer it. The passage must look like a real excerpt from a source
document - factual, declarative sentences, third person, no preamble -
not like a chatbot reply to the user.

Example:
  Question: "How does the recursive character splitter set overlap?"
  Passage: "The RecursiveCharacterTextSplitter accepts a chunk_overlap
  parameter that controls how many characters are shared between
  consecutive chunks, preserving sentence boundaries across splits."

Rules:
- Write the passage as it would appear in a source document, never as
  an answer addressed to the user ("here is", "the answer is").
- Keep it plausible and factual in tone; it does not need to be correct.
- Output only the passage, nothing else.

Question: {question}
Hypothetical passage:"""


class HyDERetriever:
    """Retrieve by embedding a hypothetical answer passage instead of the query.

    Wraps any retriever exposing ``retrieve(question) -> list[Document]``
    (e.g. a ``SimilarityRetriever`` from Project 01). Only the hypothetical
    passage reaches the inner retriever, so the vector store is untouched.
    """

    def __init__(self, hyde_llm, retriever, top_k: int = 5):
        # hyde_llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.openai.OpenAILLM, llms.gemini.GeminiLLM, or a local
        # Ollama wrapper). Imported by the caller, never at module level.
        self.hyde_llm = hyde_llm
        self.retriever = retriever
        self.top_k = top_k

    def _hypothetical(self, question: str) -> str:
        """TODO(Project 22): implement the hypothetical-passage generation.

        Call self.hyde_llm.invoke(HYDE_PROMPT.format(question=question)),
        strip the result, and fall back to the original question when the
        output is empty or unusable. That fallback is the safety net a real
        system needs.
        """
        raise NotImplementedError(
            "TODO(Project 22): implement HyDERetriever._hypothetical"
        )

    def retrieve(self, question: str) -> list[Document]:
        """Generate a hypothetical passage, then delegate retrieval to the inner retriever."""
        hypothetical = self._hypothetical(question)
        return self.retriever.retrieve(hypothetical)[: self.top_k]


if __name__ == "__main__":
    # Fake LLM + fake retriever - no network, runs anywhere.
    class _FakeHyDELLM:
        """Returns a fixed hypothetical passage."""

        def invoke(self, prompt_text: str) -> str:
            return "The capital of France is Paris, a city on the Seine."

    class _FakeRetriever:
        """Records the question it received and echoes it back as a document."""

        def __init__(self):
            self.last_question = None

        def retrieve(self, question: str) -> list[Document]:
            self.last_question = question
            return [Document(page_content=f"answer to: {question}")]

    inner = _FakeRetriever()
    retriever = HyDERetriever(_FakeHyDELLM(), inner, top_k=1)
    docs = retriever.retrieve("What is the capital of France?")

    # The inner retriever must have seen the HYPOTHETICAL passage, not the query.
    assert inner.last_question == "The capital of France is Paris, a city on the Seine.", inner.last_question
    assert docs[0].page_content == "answer to: The capital of France is Paris, a city on the Seine."
    print("OK: retrieve() embedded the hypothetical passage, not the raw query")
