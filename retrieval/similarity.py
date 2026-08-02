"""Naive similarity retriever.

Retriever block: the baseline every other strategy improves on.
See Topics/Project-01-Baseline-RAG/README.md.
"""

from langchain_core.documents import Document


class SimilarityRetriever:
    """Return top-k chunks by plain cosine similarity."""

    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Document]:
        """Embed the question and return the top-k most similar documents.

        Raises ValueError (with context) if the store cannot embed the
        query or search it — e.g. no embedding model is configured.
        """
        try:
            query_embedding = self.vector_store.embed_query(question)
            return self.vector_store.query(query_embedding, top_k=self.top_k)
        except ValueError as exc:
            raise ValueError(
                "SimilarityRetriever: the vector store could not embed or "
                f"search {question!r} (is an embedding model configured?): "
                f"{exc}"
            ) from exc


if __name__ == "__main__":
    # Tiny inline stand-in for a vector store: keyword-overlap "embeddings".
    # No network, no langchain/chromadb — the demo runs anywhere.
    class _FakeStore:
        """Minimal vector store: cosine similarity over keyword-overlap vectors."""

        VOCAB: tuple[str, ...] = ("cat", "dog", "bird", "sleep", "eat", "play")

        def __init__(self, docs: dict[str, str]):
            # page_content -> source label
            self._docs = dict(docs)
            self._embeddings = {text: self._embed(text) for text in self._docs}

        def _embed(self, text: str) -> list[float]:
            return [1.0 if tok in text.lower() else 0.0 for tok in self.VOCAB]

        def embed_query(self, question: str) -> list[float]:
            return self._embed(question)

        def _cosine(self, emb_a: list[float], emb_b: list[float]) -> float:
            dot = sum(a * b for a, b in zip(emb_a, emb_b))
            if not any(emb_a) or not any(emb_b):
                return 0.0
            return dot / (sum(emb_a) * sum(emb_b)) ** 0.5

        def query(
            self, query_embedding: list[float], top_k: int = 5
        ) -> list[Document]:
            ranked = sorted(
                self._docs,
                key=lambda text: self._cosine(
                    query_embedding, self._embeddings[text]
                ),
                reverse=True,
            )
            return [
                Document(page_content=text, metadata={"source": self._docs[text]})
                for text in ranked[:top_k]
            ]

    store = _FakeStore(
        {
            "cats sleep a lot": "animals",
            "dogs love long walks": "animals",
            "birds build nests": "animals",
            "python is a snake": "reptiles",
        }
    )
    retriever = SimilarityRetriever(store, top_k=3)

    for question in ("sleepy cat", "walking a dog", "what is a klingon?"):
        print(f"Q: {question!r}")
        for doc in retriever.retrieve(question):
            print(f"  - {doc.page_content!r} (source={doc.metadata['source']})")

    # Demonstrate the documented failure mode: a store without embeddings.
    class _NoEmbedderStore:
        def embed_query(self, text: str) -> list[float]:
            raise ValueError("no embedding model configured")

    try:
        SimilarityRetriever(_NoEmbedderStore()).retrieve("anything")
    except ValueError as exc:
        print(f"Expected error wrapped with context: {exc}")
