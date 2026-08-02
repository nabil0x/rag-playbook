"""Gemini embedding model.

Embedding block: swap the embedding model — only one line changes.
See Topics/Project-06-Better-Embeddings/README.md.
"""


class GeminiEmbedding:
    """Embed text with Google's Gemini embedding model."""

    def __init__(self, model: str = "models/embedding-001"):
        self.model = model

    def embed_query(self, text: str) -> list:
        # TODO: implement via langchain_google_genai GoogleGenerativeAIEmbeddings
        raise NotImplementedError
