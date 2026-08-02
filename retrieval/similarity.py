"""Naive similarity retriever.

Retriever block: the baseline every other strategy improves on.
See Topics/Project-01-Baseline-RAG/README.md.
"""


class SimilarityRetriever:
    """Return top-k chunks by plain cosine similarity."""

    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> list:
        # TODO: embed the query, search the store, return top-k chunks
        raise NotImplementedError
