"""Gemini LLM wrapper.

LLM block: only `llm = ...` changes when swapping models.
See Topics/Project-01-Baseline-RAG/README.md.
"""

import json
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

        Mirrors ``GroqLLM.json_object`` (and ``LLMJudge.judge``): strips a
        markdown code fence if present, and on parse failure re-prompts with
        a JSON-only instruction up to ``retries`` times. Returns
        ``{"error": ...}`` when the model never produces parseable JSON.
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
    if not os.getenv("GOOGLE_API_KEY"):
        print("SKIP: GOOGLE_API_KEY not set")
    else:
        print(GeminiLLM().invoke("Say hello in one short sentence."))
