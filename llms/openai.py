"""OpenAI LLM wrapper.

LLM block: swap Gemini -> OpenAI with one line.
See Topics/Project-18-RAG-Benchmark-Suite/README.md.
"""


class OpenAILLM:
    """Generate answers with OpenAI's chat models."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def invoke(self, prompt: str) -> str:
        # TODO: implement via langchain_openai ChatOpenAI
        raise NotImplementedError
