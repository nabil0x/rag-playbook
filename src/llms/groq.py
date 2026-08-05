"""Groq LLM wrapper.

LLM block: swap Gemini -> Groq with one line — same ``invoke(prompt) -> str``
contract as ``src/llms/gemini.py`` and ``src/llms/openai.py``, so any retriever or
prompt that takes an "LLM exposing invoke(str) -> str" works unchanged.
See Topics/Project-01-Baseline-RAG/README.md.
"""

import json
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

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove a surrounding markdown code fence (```json ... ```).

        Hosted models routinely wrap their JSON answers in a fenced block;
        ``json.loads`` cannot parse the raw text. Strips the leading `````
        line (and optional ``json`` label) and the trailing ````` line,
        leaving the bare JSON.
        """
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def json_object(self, prompt: str, retries: int = 2) -> dict:
        """Ask the model to output ONLY a JSON object and parse it.

        Mirrors ``LLMJudge.judge``'s contract: strips a markdown code fence
        if present, and on parse failure re-prompts with a JSON-only
        instruction up to ``retries`` times. Returns ``{"error": ...}`` when
        the model never produces parseable JSON.
        """
        text = ""
        for attempt in range(retries + 1):
            full = prompt if attempt == 0 else prompt + (
                "\n\nRespond with ONLY valid JSON, no markdown.")
            text = self.invoke(full)
            try:
                parsed = json.loads(self._strip_code_fence(text))
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {"error": f"could not parse JSON after {retries + 1} attempts",
                "raw": text}


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("SKIP: GROQ_API_KEY not set")
    else:
        print(GroqLLM().invoke("Say hello in one short sentence."))
