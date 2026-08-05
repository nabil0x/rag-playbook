"""MMR retriever (Maximum Marginal Relevance).

Retriever block: trade relevance against diversity.
`retriever = vectorstore.as_retriever(search_type="mmr")`.
See Topics/Project-08-MMR-Retrieval/README.md.
"""

from langchain_core.documents import Document


class MMRRetriever:
    """Return diverse, relevant chunks using MMR."""

    def __init__(self, vector_store, top_k: int = 5, lambda_mult: float = 0.5):
        self.vector_store = vector_store
        self.top_k = top_k
        self.lambda_mult = lambda_mult

    def retrieve(self, question: str) -> list[Document]:
        """Embed the question and run MMR search on the vector store.

        lambda_mult=1.0 keeps pure relevance; lambda_mult=0.0 maximizes
        diversity against the already-selected documents.

        Raises ValueError (with context) if the store cannot embed the
        query or search it — e.g. no embedding model is configured.
        """
        try:
            query_embedding = self.vector_store.embed_query(question)
            return self.vector_store.query_mmr(
                query_embedding,
                top_k=self.top_k,
                lambda_mult=self.lambda_mult,
            )
        except ValueError as exc:
            raise ValueError(
                "MMRRetriever: the vector store could not embed or "
                f"search {question!r} (is an embedding model configured?): "
                f"{exc}"
            ) from exc


if __name__ == "__main__":
    # Tiny inline stand-in for a vector store: keyword-overlap "embeddings"
    # plus a greedy query_mmr implementation. No network, no dependencies.
    class _FakeStore:
        """Minimal vector store with a greedy MMR query."""

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

        def query_mmr(
            self,
            query_embedding: list[float],
            top_k: int = 5,
            lambda_mult: float = 0.5,
        ) -> list[Document]:
            # Greedy MMR: repeatedly pick the candidate maximizing
            # lambda * relevance - (1 - lambda) * similarity to selected docs.
            pool: list[str] = list(self._docs)
            selected: list[str] = []

            while len(selected) < top_k and pool:
                def mmr_score(text: str) -> float:
                    relevance = self._cosine(
                        query_embedding, self._embeddings[text]
                    )
                    diversity = 0.0
                    if selected:
                        diversity = max(
                            self._cosine(
                                self._embeddings[text], self._embeddings[s]
                            )
                            for s in selected
                        )
                    return lambda_mult * relevance - (1.0 - lambda_mult) * diversity

                chosen = max(pool, key=mmr_score)
                pool.remove(chosen)
                selected.append(chosen)

            return [
                Document(page_content=text, metadata={"source": self._docs[text]})
                for text in selected
            ]

    store = _FakeStore(
        {
            "cats and dogs sleep": "animals",
            "dogs chase cats": "animals",
            "cats eat birds": "animals",
            "birds sleep in trees": "animals",
        }
    )

    # Same store, same question: only lambda_mult changes the ranking.
    for lambda_mult in (1.0, 0.5, 0.0):
        retriever = MMRRetriever(store, top_k=3, lambda_mult=lambda_mult)
        docs = retriever.retrieve("sleepy cat")
        print(f"lambda_mult={lambda_mult}: {[doc.page_content for doc in docs]}")
