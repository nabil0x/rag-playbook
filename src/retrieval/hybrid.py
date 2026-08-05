"""Hybrid retriever (dense + sparse / keyword).

Retriever block: combine vector similarity with BM25-style keyword
matching, then fuse the rankings.
See Topics/Project-15-Hybrid-Search/README.md.
"""

from langchain_core.documents import Document


class HybridRetriever:
    """Fuse dense and sparse retrieval results into one ranked list.

    Dense retrieval (embedding similarity) is strong on semantic matches;
    sparse retrieval (BM25-style keyword matching) is strong on exact term
    matches. The two rankings are fused with Reciprocal Rank Fusion (RRF):
    every document earns 1 / (60 + rank) for each ranked list it appears in,
    and the summed scores decide the final order. Documents found by both
    retrievers are deduplicated and get a boost — the classic hybrid win.

    `sparse_retriever` is any object exposing
    retrieve(question) -> list[Document]; a typical choice wraps
    langchain_community.retrievers.BM25Retriever.
    """

    # RRF constant: how strongly lower ranks are discounted (k=60 is standard).
    RRF_K: int = 60

    def __init__(self, dense_retriever, sparse_retriever, top_k: int = 5):
        # sparse_retriever: e.g. langchain_community.retrievers.BM25Retriever,
        # which needs rank-bm25 (`pip install rank-bm25`). It is imported by
        # the caller, never at module level, so this file has no heavy deps.
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """RRF dedup key: the page content, or a content hash when empty."""
        return doc.page_content or str(hash(repr(doc)))

    def retrieve(self, question: str) -> list[Document]:
        """Run both retrievers, fuse with RRF, return the top-k documents."""
        dense_docs = self.dense_retriever.retrieve(question)
        sparse_docs = self.sparse_retriever.retrieve(question)

        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}

        for ranked in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranked, start=1):
                key = self._doc_key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank)
                by_key[key] = doc

        fused = sorted(by_key, key=lambda key: scores[key], reverse=True)
        return [by_key[key] for key in fused[: self.top_k]]


if __name__ == "__main__":
    # Two tiny fake retrievers returning overlapping ranked lists.
    # No network, no langchain_community/rank-bm25 — the demo runs anywhere.
    class _FakeDenseRetriever:
        """Stands in for a dense (vector similarity) retriever."""

        def retrieve(self, question: str) -> list[Document]:
            return [
                Document(page_content="D1 cats sleep"),
                Document(page_content="D2 dogs bark"),
                Document(page_content="D3 birds fly"),
            ]

    class _FakeSparseRetriever:
        """Stands in for a BM25-style keyword retriever."""

        def retrieve(self, question: str) -> list[Document]:
            return [
                Document(page_content="D2 dogs bark"),
                Document(page_content="D4 fish swim"),
            ]

    retriever = HybridRetriever(
        _FakeDenseRetriever(), _FakeSparseRetriever(), top_k=4
    )
    results = retriever.retrieve("dogs")

    print("Fused ranking:")
    for doc in results:
        print(f"  - {doc.page_content}")

    # D2 appears in both lists (rank 2 and rank 1), so RRF must put it on
    # top, and it must be deduplicated (appears exactly once in the output).
    assert [doc.page_content for doc in results] == [
        "D2 dogs bark",
        "D1 cats sleep",
        "D4 fish swim",
        "D3 birds fly",
    ], results
    assert len(results) == 4
    assert len({doc.page_content for doc in results}) == 4
    print("OK: fused order dedupes and ranks sensibly")
