"""Groq LLM wrapper.

LLM block: swap Gemini -> Groq with one line — same ``invoke(prompt) -> str``
contract as ``llms/gemini.py`` and ``llms/openai.py``, so any retriever or
prompt that takes an "LLM exposing invoke(str) -> str" works unchanged.
See Topics/Project-01-Baseline-RAG/README.md.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class GroqLLM:
    """Generate answers with Groq's hosted Llama models."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        """Lazily build (and cache) the ChatGroq instance."""
        if self._llm is None:
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add GROQ_API_KEY=<your key> to "
                    "the .env file next to this repo."
                )
            try:
                from langchain_groq import ChatGroq
            except ImportError:
                raise ImportError(
                    "GroqLLM needs langchain-groq: pip install langchain-groq"
                )
            self._llm = ChatGroq(model=self.model, temperature=self.temperature)
        return self._llm

    def invoke(self, prompt: str) -> str:
        response = self._get_llm().invoke(prompt)
        return response.content


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("SKIP: GROQ_API_KEY not set")
    else:
        print(GroqLLM().invoke("Say hello in one short sentence."))
