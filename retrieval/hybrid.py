"""Hybrid retriever (dense + sparse / keyword).

Retriever block: combine vector similarity with BM25-style keyword
matching, then fuse the rankings.
See Topics/Project-15-Hybrid-Search/README.md.
"""


class HybridRetriever:
    """Fuse dense and sparse retrieval results into one ranked list."""

    def __init__(self, dense_retriever, sparse_retriever, top_k: int = 5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k

    def retrieve(self, question: str) -> list:
        # TODO: run both retrievers, then fuse (e.g. reciprocal rank fusion)
        raise NotImplementedError
