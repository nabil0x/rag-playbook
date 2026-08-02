"""E5 embedding model (intfloat/multilingual-e5-base).

Embedding block: swap the embedding model.
See Topics/Project-06-Better-Embeddings/README.md.
"""


class E5Embedding:
    """Embed text with intfloat/multilingual-e5-base."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name

    def embed_query(self, text: str) -> list:
        # TODO: implement with HuggingFaceEmbeddings / sentence_transformers
        raise NotImplementedError
