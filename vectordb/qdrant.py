"""Qdrant vector store.

Vector DB block: `Qdrant.from_documents(...)`.
See Topics/Project-07-Compare-Vector-Databases/README.md.
"""


class QdrantVectorStore:
    """Persist and query embeddings with Qdrant."""

    def __init__(self, collection_name: str = "rag"):
        self.collection_name = collection_name

    def add(self, chunks: list, embeddings: list) -> None:
        # TODO: implement with langchain_qdrant Qdrant
        raise NotImplementedError

    def query(self, query_embedding: list, top_k: int = 5) -> list:
        # TODO: implement similarity search
        raise NotImplementedError
