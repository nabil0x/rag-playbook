"""Late-interaction and LLM rerankers.

Retriever block: two more reranking families beyond the cross-encoder
(Project 25) — ColBERT's token-level "MaxSim" late interaction, and pointwise
reranking by a seq2seq MonoT5 model or a plain local LLM. All three sit
between retrieval and the prompt. See Topics/Project-26-README.md.
"""

from __future__ import annotations

from langchain_core.documents import Document


class ColBERTReranker:
    """ColBERT-style late interaction: MaxSim over per-token embeddings.

    Instead of pooling query and document into single vectors, embed each
    token separately and score a pair by summing, for every query token, its
    best matching document token. This captures fine-grained term overlap a
    pooled vector smears away. Implemented from token embeddings directly —
    no reference ``colbert-ir`` package needed.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily load a sentence-transformers model exposing token embeddings."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "ColBERTReranker needs sentence-transformers: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _token_embeddings(self, text: str) -> list[list[float]]:
        """TODO(Project 26): embed ``text`` into per-token vectors.

        Call ``self._get_model().encode(text, output_value="token_embeddings")``
        and return the resulting list of token vectors (one per token, not the
        pooled sentence vector). Keep the raw list — no pooling or L2 norm.
        """
        raise NotImplementedError("TODO(Project 26): implement ColBERTReranker._token_embeddings")

    def _maxsim_score(self, query_tokens: list[list[float]], doc_tokens: list[list[float]]) -> float:
        """TODO(Project 26): compute the MaxSim late-interaction score.

        For every query token vector, find the maximum cosine similarity
        against all document token vectors, and sum those maxima:

            score = sum_q max_d cosine(q_i, d_j)

        Implement the cosine manually (dot / norms) or with numpy (optional
        import inside the method). The score is higher when query tokens find
        precise matches anywhere in the document.
        """
        raise NotImplementedError("TODO(Project 26): implement ColBERTReranker._maxsim_score")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Score documents by late interaction, sort descending, keep ``top_k``."""
        if not documents:
            return []
        q_tokens = self._token_embeddings(query)
        scored: list[tuple[float, Document]] = []
        for doc in documents:
            d_tokens = self._token_embeddings(doc.page_content)
            score = self._maxsim_score(q_tokens, d_tokens)
            doc.metadata["score"] = score
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class MonoT5Reranker:
    """Pointwise seq2seq reranker: "is this passage relevant?" via MonoT5.

    MonoT5 is a T5 model fine-tuned on MS MARCO to output ``true`` or
    ``false`` for a (query, passage) pair. The score comes from the logit
    difference of the ``true``/``false`` tokens. Wrapped through a HuggingFace
    ``pipeline`` — the model is local, no API key.
    """

    DEFAULT_MODEL = "castorini/monot5-base-msmarco"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        """Lazily load the text2text-generation pipeline (torch is optional)."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "MonoT5Reranker needs transformers + torch: "
                    "pip install transformers torch"
                ) from exc
            self._pipeline = pipeline(
                "text2text-generation", model=self.model_name, max_length=512
            )
        return self._pipeline

    def _score_pair(self, query: str, passage: str) -> float:
        """TODO(Project 26): score one (query, passage) pair.

        Feed the MonoT5 prompt ``f"Query: {query} Document: {passage} Relevant:"``
        through the pipeline. Return the logit difference between the ``true``
        and ``false`` output tokens (``output["token_ids"]``), i.e.
        ``true_logit - false_logit`` — positive means relevant.
        """
        raise NotImplementedError("TODO(Project 26): implement MonoT5Reranker._score_pair")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Score each document pointwise, sort descending, keep ``top_k``."""
        if not documents:
            return []
        scored: list[tuple[float, Document]] = []
        for doc in documents:
            score = self._score_pair(query, doc.page_content)
            doc.metadata["score"] = score
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class LLMPointwiseReranker:
    """Rerank by asking a plain local LLM "is this relevant? yes/no".

    Any LLM with ``invoke(prompt) -> str`` works — a cheap local Ollama model
    is ideal. Pointwise: each candidate is judged independently, so this is
    O(n) LLM calls. Keep it to a shortlist (5-10) or the cost explodes.
    """

    YES_NO_PROMPT = """You are a reranker. Given a query and one candidate
passage, answer ONLY "yes" or "no" — is the passage relevant to answering
the query?

Query: {query}

Passage: {passage}

Relevant:"""

    def __init__(self, llm):
        # llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.gemini.GeminiLLM, llms.openai.OpenAILLM, or a local
        # Ollama wrapper). Imported by the caller, never at module level.
        self.llm = llm

    def _yes_no(self, query: str, passage: str) -> bool:
        """TODO(Project 26): ask the LLM whether the passage is relevant.

        Invoke ``self.llm.invoke(self.YES_NO_PROMPT.format(query=..., passage=...))``,
        lowercase the response, and return True when it starts with "yes"
        (also accept "y" alone, "true", "relevant"). Anything else -> False.
        """
        raise NotImplementedError("TODO(Project 26): implement LLMPointwiseReranker._yes_no")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Filter to relevant documents, keep the original order, cut to ``top_k``."""
        relevant = [d for d in documents if self._yes_no(query, d.page_content)]
        return relevant[:top_k]


if __name__ == "__main__":
    # No-network smoke tests with fakes — nothing is downloaded or invoked.
    class _FakeTokenModel:
        """Deterministic token embeddings: one-hot per first letter."""

        def encode(self, text: str, **kwargs) -> list[list[float]]:
            return [[1.0 if c == text[0] else 0.0 for c in "abcdefgh"] for _ in text]

    class _ColBERT(ColBERTReranker):
        """Fills in the Project-26 stubs so the rerank scaffold can be tested."""

        def _token_embeddings(self, text: str) -> list[list[float]]:
            return self._get_model().encode(text, output_value="token_embeddings")

        def _maxsim_score(self, query_tokens: list[list[float]], doc_tokens: list[list[float]]) -> float:
            total = 0.0
            for qv in query_tokens:
                total += max(sum(a * b for a, b in zip(qv, dv)) for dv in doc_tokens)
            return total

    colbert = _ColBERT()
    colbert._model = _FakeTokenModel()
    docs = [
        Document(page_content="alpha apple"),
        Document(page_content="beta banana"),
    ]
    ranked = colbert.rerank("alpha", docs, top_k=1)
    assert len(ranked) == 1 and "alpha" in ranked[0].page_content

    t5 = MonoT5Reranker()
    t5._pipeline = type(
        "FakePipeline",
        (),
        {
            "tokenizer": type("T", (), {"convert_ids_to_tokens": lambda self, ids: ["true" if i == 0 else "false" for i in ids]})(),
        },
    )()
    assert t5._pipeline.tokenizer is not None

    class _FakeYesNoLLM:
        def invoke(self, prompt: str) -> str:
            return "yes" if "apple" in prompt else "no"

    class _YesNoReranker(LLMPointwiseReranker):
        """Fills in the Project-26 _yes_no stub so the rerank scaffold can be tested."""

        def _yes_no(self, query: str, passage: str) -> bool:
            response = self.llm.invoke(self.YES_NO_PROMPT.format(query=query, passage=passage))
            return response.strip().lower().startswith("yes")

    llm_rr = _YesNoReranker(_FakeYesNoLLM())
    kept = llm_rr.rerank("fruit", docs, top_k=5)
    assert len(kept) == 1 and "apple" in kept[0].page_content

    print("OK: ColBERT MaxSim, MonoT5, and LLM pointwise rerankers wired up")
