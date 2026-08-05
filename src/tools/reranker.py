"""Cross-encoder reranking (LangChain gap — no native cross-encoder).

Retriever block: retrieve a wide candidate list with a cheap bi-encoder, then
re-score the short candidate list with a cross-encoder that reads each
(query, document) PAIR as one input. Pair-level attention is far better at
relevance than cosine over pooled vectors — at the cost of one forward pass
per pair, which is why the cross-encoder only ever sees a short candidate
list. See the 06-re-ranking labs for the empirical lift vs the bi-encoder
baseline (nDCG@k, gold-position recovery, lost-in-the-middle).
"""

from __future__ import annotations

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Re-score documents with a sentence-transformers ``CrossEncoder``.

    Scores each (query, document) pair in one batched ``predict`` call,
    attaches ``metadata["score"]`` (higher = more relevant) and returns a new
    list sorted by score descending. The input documents are not reordered
    and no store is touched — the reranker sits between retrieval and the
    prompt. The model is loaded lazily on first use and cached.

    Args:
        model_name: sentence-transformers cross-encoder checkpoint. The
            default is a small MS-MARCO relevance model (~90 MB, local).
        batch_size: pairs per ``predict`` batch (throughput vs memory).
    """

    #: Default model: small, local, trained on MS MARCO relevance pairs.
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazily load the cross-encoder (sentence-transformers is optional)."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker needs sentence-transformers: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, documents: list[Document]) -> list[Document]:
        """Attach ``metadata["score"]`` to every document (order preserved).

        Returns the SAME list with scores attached — the caller decides what
        to do with the ranking. Prefer ``rerank`` for the common
        retrieve-wide-then-keep-short flow.
        """
        if not documents:
            return documents
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._get_model().predict(
            pairs,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        for doc, score in zip(documents, scores):
            doc.metadata["score"] = float(score)
        return documents

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Score the candidates, sort by score descending, keep ``top_k``.

        Each kept document carries ``metadata["score"]`` (a float). The input
        list is not mutated — a new list is returned. Empty candidate lists
        return an empty list.
        """
        if not documents:
            return []
        self.score(query, documents)
        ranked = sorted(
            documents,
            key=lambda doc: doc.metadata["score"],
            reverse=True,
        )
        return ranked[:top_k]


if __name__ == "__main__":
    # No-network smoke test with a fake cross-encoder: scores by word overlap.
    class _FakeCrossEncoder:
        """Pretends to be a CrossEncoder; scores = query-term overlap."""

        def predict(self, pairs, **kwargs) -> list[float]:
            return [sum(1 for w in q.split() if w in d) for q, d in pairs]

    reranker = CrossEncoderReranker()
    reranker._model = _FakeCrossEncoder()  # inject the fake, no download

    docs = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="RAG retrieves relevant documents."),
        Document(page_content="France is a country in Europe."),
    ]
    original_order = [d.page_content for d in docs]

    # score() keeps the input order and attaches scores.
    scored = reranker.score("Paris France", docs)
    assert [d.page_content for d in scored] == original_order
    assert all("score" in d.metadata for d in scored)

    # rerank() returns a NEW descending list, top-k only.
    top = reranker.rerank("Paris France", docs, top_k=2)
    assert len(top) == 2
    assert "Paris is the capital" in top[0].page_content  # 2/2 words match
    assert top[0].metadata["score"] >= top[1].metadata["score"]
    assert [d.page_content for d in docs] == original_order  # inputs untouched

    print("OK: score() preserves order, rerank() sorts descending and keeps top_k")
