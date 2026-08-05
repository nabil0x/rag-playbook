"""OpenAI LLM wrapper.

LLM block: swap Gemini -> OpenAI with one line.
See Topics/Project-18-RAG-Benchmark-Suite/README.md.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class OpenAILLM:
    """Generate answers with OpenAI's chat models."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._llm = None

    def _get_llm(self):
        """Lazily build (and cache) the ChatOpenAI instance."""
        if self._llm is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add OPENAI_API_KEY=<your key> to "
                    "the .env file next to this repo."
                )
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAILLM needs langchain-openai: pip install langchain-openai"
                )
            self._llm = ChatOpenAI(model=self.model, temperature=0.2)
        return self._llm

    def invoke(self, prompt: str) -> str:
        response = self._get_llm().invoke(prompt)
        return response.content


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
    else:
        try:
            print(OpenAILLM().invoke("Say hello in one short sentence."))
        except ImportError:
            print("SKIP: langchain-openai not installed (pip install langchain-openai)")
