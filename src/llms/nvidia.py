"""NVIDIA NIM LLM wrapper (OpenAI-compatible endpoint).

LLM block: same ``invoke(prompt) -> str`` / ``json_object(prompt) -> dict``
contract as ``llms/groq.py`` and ``llms/gemini.py``, so any retriever or
prompt that takes an "LLM exposing invoke(str) -> str" works unchanged.
Backed by the NVIDIA hosted NIM endpoint (``integrate.api.nvidia.com``)
with the DeepSeek chat model, thinking disabled for fast deterministic
answers.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaLLM:
    """Generate answers with NVIDIA's hosted NIM chat models."""

    def __init__(self, model: str = "deepseek-ai/deepseek-v4-pro",
                 temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        """Lazily build (and cache) the ChatOpenAI instance for the NIM endpoint."""
        if self._llm is None:
            if not os.getenv("NVIDIA_API_KEY"):
                raise RuntimeError(
                    "NVIDIA_API_KEY is not set. Add NVIDIA_API_KEY=<your key> "
                    "to the .env file next to this repo."
                )
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "NvidiaLLM needs langchain-openai: pip install langchain-openai"
                )
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                base_url=NVIDIA_BASE_URL,
                api_key=os.getenv("NVIDIA_API_KEY"),
                extra_body={"chat_template_kwargs": {"thinking": False}},
                max_tokens=4096,
            )
        return self._llm

    def invoke(self, prompt: str, retries: int = 3) -> str:
        """Generate text, retrying transient API failures with backoff.

        Hosted NIM endpoints intermittently return 500/429 or drop the
        connection mid-batch; retrying a few times with a short sleep
        normally rides through. Non-transient errors (auth, bad request)
        propagate immediately.
        """
        import time

        from openai import (APIConnectionError, APITimeoutError,
                            InternalServerError, RateLimitError)

        transient = (InternalServerError, RateLimitError,
                     APITimeoutError, APIConnectionError)
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self._get_llm().invoke(prompt)
                return response.content
            except transient as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"NVIDIA NIM failed after {retries + 1} attempts: {last_error}"
        ) from last_error

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
    if not os.getenv("NVIDIA_API_KEY"):
        print("SKIP: NVIDIA_API_KEY not set")
    else:
        print(NvidiaLLM().invoke("Say hello in one short sentence."))
