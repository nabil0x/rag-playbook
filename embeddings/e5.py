"""E5 embedding model (intfloat/multilingual-e5-base).

Embedding block: swap the embedding model.
See Topics/Project-06-Better-Embeddings/README.md.

E5 models are trained with instruction prefixes: queries are prefixed
"query: " and passages "passage: " before embedding.

Depends on langchain-huggingface + sentence-transformers (+ torch).
These are imported lazily inside the methods so that this module can be
imported even on machines where the heavy deps are not installed.
"""


class E5Embedding:
    """Embed text with intfloat/multilingual-e5-base.

    Applies the E5 instruction prefixes ("query: " / "passage: ") before
    embedding. The model is built lazily on first use and cached.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Build and cache the HuggingFaceEmbeddings model."""
        if self._model is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "E5Embedding needs langchain-huggingface: "
                    "pip install langchain-huggingface sentence-transformers"
                ) from exc

            try:
                self._model = HuggingFaceEmbeddings(model_name=self.model_name)
            except (ImportError, RuntimeError) as exc:
                raise ImportError(
                    "E5Embedding needs sentence-transformers and torch: "
                    "pip install sentence-transformers"
                ) from exc
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().embed_query(f"query: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().embed_documents(
            [f"passage: {text}" for text in texts]
        )


if __name__ == "__main__":
    try:
        emb = E5Embedding()
        vector = emb.embed_query("hello world")
        print(f"dim={len(vector)} first_3={vector[:3]}")
    except ImportError:
        print("SKIP: pip install sentence-transformers")
