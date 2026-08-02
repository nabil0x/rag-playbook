"""FAISS vector store.

Vector DB block: `FAISS.from_documents(splits, embedding)` — offline search.
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


class FAISSVectorStore:
    """In-memory vector store backed by FAISS."""

    def __init__(self, embedding=None):
        self.embedding = embedding
        self._index = None

    def add(self, chunks: list[Document], embeddings: list | None = None) -> None:
        try:
            import faiss
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "FAISSVectorStore needs faiss-cpu: pip install faiss-cpu"
            ) from exc
        if getattr(faiss, "IndexFlatL2", None) is None:
            # guards against a broken/partial faiss (or self-shadowing when
            # this script is run directly): the real faiss-cpu always has it
            raise ImportError(
                "FAISSVectorStore needs faiss-cpu: pip install faiss-cpu"
            )

        if embeddings is not None:
            embedder = _PrecomputedEmbeddings(embeddings)
        elif self.embedding is not None:
            embedder = self.embedding
        else:
            raise ValueError("add() needs embeddings or a store-level embedding model")

        self._index = FAISS.from_documents(chunks, embedding=embedder)

    def query(self, query_embedding: list, top_k: int = 5) -> list[Document]:
        if self._index is None:
            raise RuntimeError("call add() first")
        return self._index.similarity_search_by_vector(query_embedding, k=top_k)

    def query_mmr(
        self, query_embedding: list, top_k: int = 5, lambda_mult: float = 0.5
    ) -> list[Document]:
        if self._index is None:
            raise RuntimeError("call add() first")
        mmr = getattr(self._index, "max_marginal_relevance_search_by_vector", None)
        if mmr is None:
            print(
                "warning: FAISS backend has no MMR support; "
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
        import faiss
        from langchain_community.vectorstores import FAISS  # noqa: F401
    except ImportError:
        print("SKIP: FAISSVectorStore demo needs faiss-cpu: pip install faiss-cpu")
        raise SystemExit(0)
    if getattr(faiss, "IndexFlatL2", None) is None:
        print("SKIP: FAISSVectorStore demo needs faiss-cpu: pip install faiss-cpu")
        raise SystemExit(0)

    class _FakeEmbedding:
        """Keyword-presence fake embedding (identity-ish: 3 vocab slots)."""

        VOCAB = ["rag", "vector", "mmr"]

        def embed_query(self, text: str) -> list[float]:
            lowered = text.lower()
            return [1.0 if w in lowered else 0.0 for w in self.VOCAB]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    store = FAISSVectorStore(embedding=_FakeEmbedding())
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
