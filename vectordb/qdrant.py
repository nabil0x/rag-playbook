"""Qdrant vector store.

Vector DB block: `Qdrant.from_documents(...)`.
See Topics/Project-07-Compare-Vector-Databases/README.md.
"""

import sys

from langchain_core.documents import Document


class _PrecomputedEmbeddings:
    """Tiny passthrough adapter returning precomputed vectors by index."""

    def __init__(self, embeddings: list[list[float]]):
        self.embeddings = embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(
            "precomputed embeddings cannot embed queries; pass a query vector instead"
        )


class QdrantVectorStore:
    """Persist and query embeddings with Qdrant."""

    def __init__(
        self,
        collection_name: str = "rag",
        path: str = "./qdrant_storage",
        embedding=None,
    ):
        self.collection_name = collection_name
        self.path = path
        self.embedding = embedding
        self._store = None

    def add(self, chunks: list[Document], embeddings: list | None = None) -> None:
        try:
            from langchain_qdrant import Qdrant
        except ImportError as exc:
            raise ImportError(
                "QdrantVectorStore needs langchain-qdrant: pip install langchain-qdrant"
            ) from exc

        if embeddings is not None:
            embedder = _PrecomputedEmbeddings(embeddings)
        elif self.embedding is not None:
            embedder = self.embedding
        else:
            raise ValueError("add() needs embeddings or a store-level embedding model")

        self._store = Qdrant.from_documents(
            chunks,
            embedding=embedder,
            collection_name=self.collection_name,
            path=self.path,
        )

    def query(self, query_embedding: list, top_k: int = 5) -> list[Document]:
        if self._store is None:
            raise RuntimeError("call add() first")
        return self._store.similarity_search_by_vector(query_embedding, k=top_k)

    def query_mmr(
        self, query_embedding: list, top_k: int = 5, lambda_mult: float = 0.5
    ) -> list[Document]:
        if self._store is None:
            raise RuntimeError("call add() first")
        mmr = getattr(self._store, "max_marginal_relevance_search_by_vector", None)
        if mmr is None:
            print(
                "warning: Qdrant backend has no MMR support; "
                "falling back to plain similarity search",
                file=sys.stderr,
            )
            return self.query(query_embedding, top_k=top_k)
        return mmr(query_embedding, k=top_k, lambda_mult=lambda_mult)

    def embed_query(self, text: str) -> list[float]:
        if self.embedding is None:
            raise ValueError("vector store has no embedding model")
        return self.embedding.embed_query(text)


if __name__ == "__main__":
    try:
        import langchain_qdrant  # noqa: F401
    except ImportError:
        print(
            "SKIP: QdrantVectorStore demo needs langchain-qdrant: "
            "pip install langchain-qdrant"
        )
        raise SystemExit(0)

    class _FakeEmbedding:
        """Keyword-presence fake embedding (identity-ish: 3 vocab slots)."""

        VOCAB = ["rag", "vector", "mmr"]

        def embed_query(self, text: str) -> list[float]:
            lowered = text.lower()
            return [1.0 if w in lowered else 0.0 for w in self.VOCAB]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    store = QdrantVectorStore(embedding=_FakeEmbedding())
    chunks = [
        Document(page_content="RAG retrieves relevant documents."),
        Document(page_content="Vector stores index embeddings."),
        Document(page_content="MMR balances relevance and diversity."),
    ]
    store.add(chunks)
    query_vec = store.embed_query("RAG retrieval")
    hits = store.query(query_vec)
    print("query: RAG retrieval")
    print("top hit:", hits[0].page_content if hits else None)
    mmr_hits = store.query_mmr(query_vec)
    print("mmr hits:", len(mmr_hits))
