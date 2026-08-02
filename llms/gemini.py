"""Gemini LLM wrapper.

LLM block: only `llm = ...` changes when swapping models.
See Topics/Project-01-Baseline-RAG/README.md.
"""


class GeminiLLM:
    """Generate answers with Google's Gemini model."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model

    def invoke(self, prompt: str) -> str:
        # TODO: implement via langchain_google_genai ChatGoogleGenerativeAI
        raise NotImplementedError
