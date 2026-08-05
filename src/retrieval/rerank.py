"""Cross-encoder reranking.

Retriever block: retrieve a wide candidate list with a cheap bi-encoder, then
re-score the top candidates with a cross-encoder that reads each query-document
pair. The reranker sits between retrieval and the prompt, so the vector store
is untouched. See Topics/Project-25-CrossEncoder-Reranking/README.md.
"""

from __future__ import annotations

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Re-score retrieved documents with a sentence-transformers cross-encoder.

    A cross-encoder feeds the *pair* (query, document) into the model, so it
    is far more accurate than the bi-encoder cosine used at retrieval time —
    and far slower, which is why it only ever sees a short candidate list.
    Scores are attached to ``doc.metadata["score"]`` (descending order wins).
    """

    #: Default model: small, local, trained on MS MARCO relevance pairs.
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
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

    def _score(self, query: str, documents: list[Document]) -> list[float]:
        """Score every (query, document) pair with the cross-encoder.

        Build the pair list — ``[(query, doc.page_content) for doc in
        documents]`` — and call ``self._get_model().predict(pairs)``, returning
        the raw score list (one float per document; higher = more relevant).
        """
        pairs = [(query, doc.page_content) for doc in documents]
        return list(self._get_model().predict(pairs))

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Score the candidates, sort by score descending, keep ``top_k``.

        Each kept document carries ``metadata["score"]`` (a float). The input
        list is not mutated — a new list is returned.
        """
        if not documents:
            return []
        scores = self._score(query, documents)
        ranked = sorted(
            zip(scores, documents),
            key=lambda pair: pair[0],
            reverse=True,
        )
        kept = ranked[:top_k]
        for score, doc in kept:
            doc.metadata["score"] = float(score)
        return [doc for _, doc in kept]


class RerankRetriever:
    """Retrieve wide with an inner retriever, rerank short with a cross-encoder.

    ``retrieve()`` pulls ``k_retrieve`` candidates from the inner retriever
    (cheap), then reranks to ``top_k`` (expensive but precise). The classic
    retrieve-then-rerank shape every production RAG system uses.
    """

    def __init__(self, inner_retriever, reranker: CrossEncoderReranker, k_retrieve: int = 20, top_k: int = 5):
        # inner_retriever: any object exposing retrieve(question) -> list[Document]
        # (e.g. retrieval.similarity.SimilarityRetriever from Project 01).
        self.inner_retriever = inner_retriever
        self.reranker = reranker
        self.k_retrieve = k_retrieve
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve ``k_retrieve`` candidates, rerank to ``top_k``."""
        candidates = self.inner_retriever.retrieve(question)[: self.k_retrieve]
        return self.reranker.rerank(question, candidates, top_k=self.top_k)


if __name__ == "__main__":
    # No-network smoke test with a fake cross-encoder: scores by word overlap.
    class _FakeCrossEncoder:
        """Pretends to be a CrossEncoder; scores = query-term overlap."""

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [sum(1 for w in q.split() if w in d) for q, d in pairs]

    class _ScoringReranker(CrossEncoderReranker):
        """Runs the real ``_score`` against the injected fake model."""

    reranker = _ScoringReranker()
    reranker._model = _FakeCrossEncoder()  # inject the fake, no download

    docs = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="RAG retrieves relevant documents."),
        Document(page_content="France is a country in Europe."),
    ]
    top = reranker.rerank("Paris France", docs, top_k=2)

    assert len(top) == 2
    assert "Paris is the capital" in top[0].page_content  # 2/2 words match
    assert top[0].metadata["score"] >= top[1].metadata["score"]
    assert "score" in docs[0].metadata  # original docs carry the score too

    inner = type(
        "FakeRetriever",
        (),
        {"retrieve": lambda self, q: docs},
    )()
    rr = RerankRetriever(inner, reranker, k_retrieve=20, top_k=2)
    assert len(rr.retrieve("Paris France")) == 2

    print("OK: rerank() sorts by score descending and RerankRetriever narrows the list")
