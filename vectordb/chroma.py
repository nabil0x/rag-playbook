"""Chroma vector store.

Vector DB block: swap the store — no other code changes.
See Topics/Project-07-Compare-Vector-Databases/README.md.
"""


class ChromaVectorStore:
    """Persist and query embeddings with Chroma."""

    def __init__(self, collection_name: str = "rag"):
        self.collection_name = collection_name

    def add(self, chunks: list, embeddings: list) -> None:
        # TODO: implement with langchain_chroma Chroma
        raise NotImplementedError

    def query(self, query_embedding: list, top_k: int = 5) -> list:
        # TODO: implement similarity search
        raise NotImplementedError
