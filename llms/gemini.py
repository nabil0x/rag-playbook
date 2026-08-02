"""Gemini LLM wrapper.

LLM block: only `llm = ...` changes when swapping models.
See Topics/Project-01-Baseline-RAG/README.md.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class GeminiLLM:
    """Generate answers with Google's Gemini model."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self._llm = None

    def _get_llm(self):
        """Lazily build (and cache) the ChatGoogleGenerativeAI instance."""
        if self._llm is None:
            if not os.getenv("GOOGLE_API_KEY"):
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. Add GOOGLE_API_KEY=<your key> to "
                    "the .env file next to this repo."
                )
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "GeminiLLM needs langchain-google-genai: "
                    "pip install langchain-google-genai"
                )
            self._llm = ChatGoogleGenerativeAI(model=self.model, temperature=0.2)
        return self._llm

    def invoke(self, prompt: str) -> str:
        response = self._get_llm().invoke(prompt)
        return response.content


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("SKIP: GOOGLE_API_KEY not set")
    else:
        print(GeminiLLM().invoke("Say hello in one short sentence."))
