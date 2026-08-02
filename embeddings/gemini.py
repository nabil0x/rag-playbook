"""Gemini embedding model.

Embedding block: swap the embedding model — only one line changes.
See Topics/Project-06-Better-Embeddings/README.md.

Requires a GOOGLE_API_KEY in the environment (e.g. in a .env file).
"""

import os

from dotenv import load_dotenv

load_dotenv()


class GeminiEmbedding:
    """Embed text with Google's Gemini embedding model.

    The ``GoogleGenerativeAIEmbeddings`` client is created lazily on first
    use so importing this module never requires the API key or a network
    call.
    """

    def __init__(self, model: str = "gemini-embedding-2-preview"):
        self.model = model
        self._client = None

    def _get_embeddings(self):
        """Build and cache the GoogleGenerativeAIEmbeddings client."""
        if self._client is None:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "GeminiEmbedding needs langchain-google-genai: "
                    "pip install langchain-google-genai"
                ) from exc

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. Add GOOGLE_API_KEY=your-key "
                    "to your .env file and load it with load_dotenv()."
                )

            self._client = GoogleGenerativeAIEmbeddings(model=self.model)
        return self._client

    def embed_query(self, text: str) -> list[float]:
        return self._get_embeddings().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings().embed_documents(texts)


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("SKIP: GOOGLE_API_KEY not set")
    else:
        emb = GeminiEmbedding()
        vector = emb.embed_query("hello world")
        print(f"dim={len(vector)} first_3={vector[:3]}")
