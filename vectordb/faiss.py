"""FAISS vector store.

Vector DB block: `FAISS.from_documents(splits, embedding)` — offline search.
See Topics/Project-07-Compare-Vector-Databases/README.md.
"""


class FAISSVectorStore:
    """In-memory vector store backed by FAISS."""

    def __init__(self):
        self._index = None

    def add(self, chunks: list, embeddings: list) -> None:
        # TODO: implement with langchain_community FAISS.from_documents
        raise NotImplementedError

    def query(self, query_embedding: list, top_k: int = 5) -> list:
        # TODO: implement similarity search
        raise NotImplementedError
