"""Chroma vector store.

Vector DB block: swap the store — no other code changes.
See Topics/Project-07-Compare-Vector-Databases/README.md.
"""

import sys

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class _PrecomputedEmbeddings(Embeddings):
    """Passthrough adapter mapping precomputed vectors back to their texts.

    Subclasses ``Embeddings`` so vector-store integrations accept it — they
    dispatch on ``isinstance(embedding, Embeddings)``. Vectors are looked up
    BY TEXT, not by call order: stores may call ``embed_documents`` once with
    the full list (FAISS, Chroma) or in batches (langchain-qdrant batches at
    64), and a positional offset would silently misalign vectors past the
    first batch.
    """

    def __init__(self, texts: list[str], embeddings: list[list[float]]):
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must be parallel lists")
        self._table: dict[str, list[float]] = dict(zip(texts, embeddings))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._table]
        if missing:
            raise ValueError(
                f"{len(missing)} text(s) have no precomputed vector; "
                "the store asked to embed text it was not given at add() time"
            )
        return [self._table[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(
            "precomputed embeddings cannot embed queries; pass a query vector instead"
        )


class ChromaVectorStore:
    """Persist and query embeddings with Chroma.

    Note: ``Chroma.from_documents`` reuses a collection by name — calling
    add() twice with the same ``collection_name``/``persist_dir`` accumulates
    (upserts) into the existing collection rather than starting fresh. Pass a
    new ``collection_name`` or ``persist_dir`` to start clean.
    """

    def __init__(
        self,
        collection_name: str = "rag",
        persist_dir: str = "./chroma_langchain_db",
        embedding=None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding = embedding
        self._store = None

    def add(self, chunks: list[Document], embeddings: list | None = None) -> None:
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore needs langchain-chroma: pip install langchain-chroma"
            ) from exc

        if embeddings is not None:
            embedder = _PrecomputedEmbeddings(
                [c.page_content for c in chunks], embeddings
            )
        elif self.embedding is not None:
            embedder = self.embedding
        else:
            raise ValueError("add() needs embeddings or a store-level embedding model")

        self._store = Chroma.from_documents(
            chunks,
            embedding=embedder,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )

    def load(self) -> None:
        """Reopen an existing persistent collection from disk (no re-adding).

        The query methods take query VECTORS, so reading a collection back
        needs no embedding model — the precomputed adapter is a no-op
        stand-in. Requires the collection to exist in ``persist_dir``.
        """
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore needs langchain-chroma: pip install langchain-chroma"
            ) from exc

        self._store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=_PrecomputedEmbeddings([], []),
        )

    def query(self, query_embedding: list, top_k: int = 5) -> list[Document]:
        if self._store is None:
            raise RuntimeError("call add() first")
        return self._store.similarity_search_by_vector(query_embedding, k=top_k)

    def query_with_scores(
        self, query_embedding: list, top_k: int = 5, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Return top-k as (Document, score) pairs, optionally metadata-filtered.

        With Chroma's default collection (l2 space) the score is the raw
        (squared) L2 distance — LOWER is more similar, with the same values
        FAISS reports. The ``filter`` dict uses Chroma ``where`` syntax, e.g.
        ``{"bucket": "b1"}``.
        """
        if self._store is None:
            raise RuntimeError("call add() first")
        return self._store.similarity_search_by_vector_with_relevance_scores(
            query_embedding, k=top_k, filter=filter
        )

    def query_mmr(
        self, query_embedding: list, top_k: int = 5, lambda_mult: float = 0.5
    ) -> list[Document]:
        if self._store is None:
            raise RuntimeError("call add() first")
        mmr = getattr(self._store, "max_marginal_relevance_search_by_vector", None)
        if mmr is None:
            print(
                "warning: Chroma backend has no MMR support; "
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
    import shutil
    import tempfile

    try:
        import langchain_chroma  # noqa: F401
    except ImportError:
        print(
            "SKIP: ChromaVectorStore demo needs langchain-chroma: "
            "pip install langchain-chroma"
        )
        raise SystemExit(0)

    class _FakeEmbedding(Embeddings):
        """Keyword-presence fake embedding (identity-ish: 3 vocab slots)."""

        VOCAB = ["rag", "vector", "mmr"]

        def embed_query(self, text: str) -> list[float]:
            lowered = text.lower()
            return [1.0 if w in lowered else 0.0 for w in self.VOCAB]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    store = ChromaVectorStore(
        collection_name="demo",
        persist_dir=tempfile.mkdtemp(prefix="chroma_demo_"),
        embedding=_FakeEmbedding(),
    )
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
    import shutil

    shutil.rmtree(store.persist_dir, ignore_errors=True)
