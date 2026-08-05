"""LLM-as-judge wrapper around a local Ollama model.

Evaluation block: the judge that scores answers. Wraps
``langchain_ollama.ChatOllama`` (qwen2.5-coder:7b, temperature 0) plus a
``LocalEmbeddings`` adapter over fastembed (BAAI/bge-base-en-v1.5). Everything
runs locally — no API key, no rate limits.
See Topics/Project-20-Deep-Eval/README.md.
"""

from __future__ import annotations

import json

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """LangChain-compatible local embedder (fastembed / BAAI/bge-base-en-v1.5).

    Implements the langchain ``Embeddings`` interface (``embed_documents`` +
    ``embed_query``) so ``Chroma.from_documents`` and ragas accept it directly.
    Never use langchain_community ``FastEmbedEmbeddings`` — it leaks a fastembed
    model object into usage metadata and breaks ragas.

    Args:
        model: fastembed model name.
    """

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        self.model = model
        self._emb = None

    def _get_emb(self):
        """Build and cache the fastembed TextEmbedding client."""
        if self._emb is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "LocalEmbeddings needs fastembed: pip install fastembed"
                ) from exc
            self._emb = TextEmbedding(model_name=self.model)
        return self._emb

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of vectors."""
        return [v.tolist() for v in self._get_emb().embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text into one vector."""
        return next(self._get_emb().query_embed(text)).tolist()


class LLMJudge:
    """Local LLM-as-judge for generation evaluation.

    Wraps ``langchain_ollama.ChatOllama`` (qwen2.5-coder:7b, temperature 0).
    The client is built lazily so importing this module never requires Ollama
    to be reachable — only runtime calls do.

    Args:
        model: Ollama model name.
        temperature: sampling temperature (0 = deterministic).
        embedder: a ``LocalEmbeddings`` instance for ``embed``; created lazily
            when None.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.0,
        embedder=None,
    ):
        self.model = model
        self.temperature = temperature
        self.embedder = embedder
        self._llm = None

    def _get_llm(self):
        """Build and cache the ChatOllama client."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError(
                    "LLMJudge needs langchain-ollama: pip install langchain-ollama"
                ) from exc
            self._llm = ChatOllama(model=self.model, temperature=self.temperature)
        return self._llm

    def ask(self, question: str, context: str) -> str:
        """Return a plain-text answer to ``question`` given ``context``.

        No JSON — a free-form answer for the RAG pipeline.
        """
        prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer concisely using only the context:"
        )
        return self._get_llm().invoke(prompt).content.strip()

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove a surrounding markdown code fence (```json ... ```).

        Coder-oriented models (qwen2.5-coder) routinely wrap their JSON
        answers in a fenced block; ``json.loads`` cannot parse the raw text.
        Strips the leading ````` line (and optional ``json`` label) and the
        trailing ````` line, leaving the bare JSON.
        """
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def judge(self, instruction: str, prompt: str) -> dict:
        """Ask the model to output ONLY a JSON object and parse it.

        Retries once on parse failure appending "Respond with ONLY valid
        JSON."; on a second failure returns ``{"error": ...}``.

        Raises:
            Nothing — parse failures are returned as ``{"error": ...}``.
        """
        last_error = ""
        for attempt in range(2):
            full = f"{instruction}\n\n{prompt}\n\nRespond with ONLY a valid JSON object."
            if attempt == 1:
                full += " Respond with ONLY valid JSON."
            try:
                text = self._get_llm().invoke(full).content
                return json.loads(self._strip_code_fence(text))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
        return {
            "error": f"LLMJudge: could not parse JSON after 2 attempts: {last_error}"
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with the local fastembed model."""
        if self.embedder is None:
            self.embedder = LocalEmbeddings()
        return self.embedder.embed_documents(texts)


if __name__ == "__main__":
    judge = LLMJudge()
    print("LLMJudge instantiated (no Ollama call yet)")
    print("LocalEmbeddings:", LocalEmbeddings.__name__)