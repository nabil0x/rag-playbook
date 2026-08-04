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
        """Embed ``text`` into per-token vectors.

        Call ``self._get_model().encode(text, output_value="token_embeddings")``
        and return the resulting list of token vectors (one per token, not the
        pooled sentence vector). sentence-transformers returns a torch tensor
        here, so it is converted to plain lists — ``_maxsim_score`` expects
        list-of-lists and the truthiness check ``if not query_tokens`` breaks
        on a multi-element tensor.
        """
        embeddings = self._get_model().encode(
            text, output_value="token_embeddings"
        )
        return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

    def _maxsim_score(self, query_tokens: list[list[float]], doc_tokens: list[list[float]]) -> float:
        """Compute the MaxSim late-interaction score.

        For every query token vector, find the maximum cosine similarity
        against all document token vectors, and sum those maxima:

            score = sum_q max_d cosine(q_i, d_j)

        Implemented with numpy (optional import inside the method): normalize
        both token sets, take the query x doc similarity matrix, and sum the
        row-wise maxima. The score is higher when query tokens find precise
        matches anywhere in the document. Empty token lists score 0.0.
        """
        if not query_tokens or not doc_tokens:
            return 0.0
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "ColBERTReranker._maxsim_score needs numpy: pip install numpy"
            ) from exc

        q = np.asarray(query_tokens, dtype=np.float32)
        d = np.asarray(doc_tokens, dtype=np.float32)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        sims = q @ d.T  # (n_query_tokens, n_doc_tokens) cosine matrix
        return float(sims.max(axis=1).sum())

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
        """Score one (query, passage) pair with the MonoT5 logit difference.

        Feed the MonoT5 prompt ``f"Query: {query} Document: {passage} Relevant:"``
        through the pipeline with ``return_dict_in_generate=True`` and
        ``output_scores=True``. Take the logits of the FIRST decoding step,
        read the logits at the ``true`` and ``false`` vocabulary ids, and
        return ``true_logit - false_logit`` — positive means relevant.
        """
        prompt = f"Query: {query} Document: {passage} Relevant:"
        result = self._get_pipeline()(
            prompt,
            max_length=512,
            return_dict_in_generate=True,
            output_scores=True,
        )
        out = result[0] if isinstance(result, (list, tuple)) else result
        tokenizer = self._get_pipeline().tokenizer
        true_id = tokenizer.convert_tokens_to_ids("true")
        false_id = tokenizer.convert_tokens_to_ids("false")
        logits = out["scores"][0][0]  # first decoding step, batch item 0
        return float(logits[true_id] - logits[false_id])

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
        """Ask the LLM whether the passage is relevant.

        Invoke ``self.llm.invoke(self.YES_NO_PROMPT.format(query=..., passage=...))``,
        lowercase and trim the response, and return True when it is "yes", "y"
        alone, "true", "relevant", or starts with "yes" followed by more text.
        Anything else -> False.
        """
        response = self.llm.invoke(self.YES_NO_PROMPT.format(query=query, passage=passage))
        answer = response.strip().lower().rstrip(".,!? \n")
        return (
            answer in ("yes", "y", "true", "relevant")
            or answer.startswith("yes ")
        )

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

    colbert = ColBERTReranker()
    colbert._model = _FakeTokenModel()
    docs = [
        Document(page_content="alpha apple"),
        Document(page_content="beta banana"),
    ]
    ranked = colbert.rerank("alpha", docs, top_k=1)
    assert len(ranked) == 1 and "alpha" in ranked[0].page_content
    # All "alpha" query tokens match "alpha apple" tokens exactly (score = 5.0),
    # while "beta banana" tokens are orthogonal (score = 0.0).
    assert ranked[0].metadata["score"] == 5.0
    assert docs[1].metadata["score"] == 0.0

    class _Logits:
        """Indexable logit vector without torch (used by the scripted pipeline)."""

        def __init__(self, values: dict[int, float]):
            self.values = values

        def __getitem__(self, idx):
            return self.values.get(idx, 0.0)

    class _FakeMonoT5Pipeline:
        """Scripted pipeline: true logit 3.0, false logit 1.0 at vocab ids 42/43."""

        def __init__(self):
            self.tokenizer = type(
                "T",
                (),
                {"convert_tokens_to_ids": lambda self, tok: {"true": 42, "false": 43}.get(tok, 0)},
            )()

        def __call__(self, prompt, **kwargs):
            step0 = _Logits({42: 3.0, 43: 1.0})  # (1, vocab) batch: index 0 = logits
            step1 = _Logits({})
            return [{"token_ids": [42, 43], "scores": [[step0], [step1]]}]

    t5 = MonoT5Reranker()
    t5._pipeline = _FakeMonoT5Pipeline()
    score = t5._score_pair("fruit", "an apple a day")
    assert score == 3.0 - 1.0, score  # true_logit - false_logit

    class _FakeYesNoLLM:
        def invoke(self, prompt: str) -> str:
            return "yes" if "apple" in prompt else "no"

    llm_rr = LLMPointwiseReranker(_FakeYesNoLLM())
    kept = llm_rr.rerank("fruit", docs, top_k=5)
    assert len(kept) == 1 and "apple" in kept[0].page_content
    assert llm_rr._yes_no("fruit", "an apple a day") is True
    assert llm_rr._yes_no("fruit", "a banana a day") is False

    print("OK: ColBERT MaxSim, MonoT5 logit-diff, and LLM pointwise rerankers implemented")
