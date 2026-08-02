"""BGE embedding model (BAAI/bge-base-en-v1.5).

Embedding block: replace Gemini with a local open-source embedder.
See Topics/Project-06-Better-Embeddings/README.md.
"""


class BGEEmbedding:
    """Embed text with BAAI/bge-base-en-v1.5 via sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name

    def embed_query(self, text: str) -> list:
        # TODO: implement with HuggingFaceEmbeddings / sentence_transformers
        raise NotImplementedError
