"""Local Ollama LLM wrapper (fully offline, zero quota).

LLM block: same ``invoke(prompt) -> str`` / ``json_object(prompt) -> dict``
contract as ``llms/groq.py``, ``llms/gemini.py`` and ``llms/nvidia.py``, so
any retriever or prompt that takes an "LLM exposing invoke(str) -> str"
works unchanged. Backed by ``langchain_ollama.ChatOllama`` running the
locally served ``qwen2.5-coder:7b`` model (the same model the repo's
``evaluation/judge.py`` LLMJudge uses) — a deterministic, no-quota
fallback when hosted providers are rate-limited.
"""

import json

from dotenv import load_dotenv

load_dotenv()


class OllamaLLM:
    """Generate answers with a locally served Ollama model."""

    def __init__(self, model: str = "qwen2.5-coder:7b", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        """Lazily build (and cache) the ChatOllama instance."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise ImportError(
                    "OllamaLLM needs langchain-ollama: pip install langchain-ollama"
                )
            self._llm = ChatOllama(model=self.model, temperature=self.temperature)
        return self._llm

    def invoke(self, prompt: str) -> str:
        response = self._get_llm().invoke(prompt)
        return response.content

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove a surrounding markdown code fence (```json ... ```).

        Local models routinely wrap their JSON answers in a fenced block;
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

        Mirrors ``GroqLLM.json_object`` / ``GeminiLLM.json_object``: strips a
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
    print(OllamaLLM().invoke("Say hello in one short sentence."))
