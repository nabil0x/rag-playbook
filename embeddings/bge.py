"""BGE embedding model (BAAI/bge-base-en-v1.5).

Embedding block: replace Gemini with a local open-source embedder.
See Topics/Project-06-Better-Embeddings/README.md.

Depends on langchain-huggingface + sentence-transformers (+ torch).
These are imported lazily inside the methods so that this module can be
imported even on machines where the heavy deps are not installed.
"""


class BGEEmbedding:
    """Embed text with BAAI/bge-base-en-v1.5 via sentence-transformers.

    The model is built lazily on first use and cached; the heavy
    dependencies (langchain-huggingface, sentence-transformers, torch)
    are only imported at that point.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Build and cache the HuggingFaceBgeEmbeddings model."""
        if self._model is None:
            try:
                from langchain_huggingface import HuggingFaceBgeEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "BGEEmbedding needs langchain-huggingface: "
                    "pip install langchain-huggingface sentence-transformers"
                ) from exc

            try:
                # bge models require normalized embeddings for cosine similarity.
                self._model = HuggingFaceBgeEmbeddings(
                    model_name=self.model_name,
                    encode_kwargs={"normalize_embeddings": True},
                )
            except (ImportError, RuntimeError) as exc:
                raise ImportError(
                    "BGEEmbedding needs sentence-transformers and torch: "
                    "pip install sentence-transformers"
                ) from exc
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().embed_documents(texts)


if __name__ == "__main__":
    try:
        emb = BGEEmbedding()
        vector = emb.embed_query("hello world")
        print(f"dim={len(vector)} first_3={vector[:3]}")
    except ImportError:
        print("SKIP: pip install sentence-transformers")
