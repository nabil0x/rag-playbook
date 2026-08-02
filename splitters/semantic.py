"""Semantic text splitter.

Splitter block: chunk on meaning instead of fixed character counts.
Unlike character- or token-based splitters, which cut at arbitrary
positions, a semantic splitter groups *sentences* that belong together:
sentences whose embeddings are close are kept in one chunk, and a chunk
boundary is drawn where the meaning shifts abruptly (a large distance
spike between consecutive sentence embeddings).

Algorithm (pure Python, no langchain-experimental dependency):
  1. Split each Document into sentences (light regex split).
  2. Embed every sentence with a sentence-transformer model.
  3. Compute pairwise cosine distances between consecutive embeddings.
  4. Find the breakpoint_percentile percentile of those distances; every
     distance above it marks a chunk boundary.
  5. Join sentences between breakpoints into one chunk Document,
     carrying over the source metadata.

See Topics/Project-17-Modular-RAG-Framework/README.md.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

# Sentence-boundary split: a period/question mark/exclamation point
# followed by whitespace (or end of string). Kept deliberately simple —
# no external sentence-tokenizer dependency.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Default model: small, fast, good-enough for chunk-boundary detection.
_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class SemanticSplitter:
    """Split documents into semantically coherent chunks.

    Chunks group consecutive sentences whose embeddings stay close to each
    other; a chunk boundary is inserted where the embedding distance between
    two neighbouring sentences exceeds ``breakpoint_percentile`` percent of
    all such distances.
    """

    def __init__(self, embedding=None, breakpoint_percentile: float = 95.0):
        """Store the embedding provider and the breakpoint threshold.

        Args:
            embedding: optional embedding object. If None, a local
                SentenceTransformer ("BAAI/bge-base-en-v1.5") is created on
                first use. Otherwise it must provide either
                ``embed_documents(texts) -> list[list[float]]`` (LangChain
                Embeddings-style) or ``encode(texts) -> array``
                (sentence-transformers-style); if it exposes ``embed_query``
                but not ``embed_documents``, sentences are embedded one by one.
            breakpoint_percentile: percentile (0-100) of consecutive-sentence
                distances above which a chunk breakpoint is placed.
        """
        self.embedding = embedding
        self.breakpoint_percentile = breakpoint_percentile

    def _get_encoder(self) -> Any:
        """Return a callable ``encoder(sentences: list[str]) -> numpy.ndarray``.

        Lazily imported so the module (and the rest of the pipeline) works
        without sentence-transformers installed.
        """
        if self.embedding is not None:
            return self._embed_with_provider

        try:
            import sentence_transformers  # noqa: F401  (lazy import)
        except ImportError as exc:
            raise ImportError(
                "SemanticSplitter needs sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc
        model = sentence_transformers.SentenceTransformer(_DEFAULT_MODEL)
        return lambda sentences: model.encode(sentences)

    def _embed_with_provider(self, sentences: list[str]) -> Any:
        """Embed sentences using the user-supplied ``embedding`` provider."""
        import numpy as np

        if hasattr(self.embedding, "embed_documents"):
            vectors = self.embedding.embed_documents(sentences)
            return np.asarray(vectors, dtype="float32")
        if hasattr(self.embedding, "embed_query"):
            return np.asarray(
                [self.embedding.embed_query(sentence) for sentence in sentences],
                dtype="float32",
            )
        if hasattr(self.embedding, "encode"):
            return np.asarray(self.embedding.encode(sentences), dtype="float32")
        raise TypeError(
            "SemanticSplitter.embedding must provide embed_documents(), "
            "embed_query(), or encode(); got "
            f"{type(self.embedding).__name__}"
        )

    def _split_one(self, doc: Document, encoder: Any) -> list[Document]:
        """Split a single Document into semantic chunks."""
        import numpy as np

        sentences = _SENTENCE_RE.split(doc.page_content.strip())
        sentences = [s for s in sentences if s]
        if not sentences:
            return []

        # Short text: a single sentence never spans a breakpoint, so it is
        # already one chunk — no need to embed anything.
        if len(sentences) == 1:
            return [Document(page_content=sentences[0], metadata=dict(doc.metadata))]

        embeddings = np.asarray(encoder(sentences), dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard against zero-vector sentences
        normalized = embeddings / norms

        # Cosine distance between consecutive sentences = 1 - cosine similarity.
        dots = np.sum(normalized[:-1] * normalized[1:], axis=1)
        distances = 1.0 - np.clip(dots, -1.0, 1.0)

        threshold = float(
            np.percentile(distances, self.breakpoint_percentile)
        )
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        # Build chunks: sentence indices are split at the breakpoints
        # (a breakpoint at i means the boundary is *after* sentence i).
        boundaries = [0] + [b + 1 for b in breakpoints] + [len(sentences)]
        chunks: list[Document] = []
        for start, end in zip(boundaries, boundaries[1:]):
            joined = " ".join(sentences[start:end])
            chunks.append(
                Document(page_content=joined, metadata=dict(doc.metadata))
            )
        return chunks

    def split_docs(self, docs: list[Document]) -> list[Document]:
        """Split documents into semantic chunks (canonical entry point)."""
        encoder = self._get_encoder()
        chunks: list[Document] = []
        for doc in docs:
            chunks.extend(self._split_one(doc, encoder))
        return chunks

    def split(self, docs: list[Document]) -> list[Document]:
        """Split documents into semantic chunks (alias of ``split_docs``)."""
        return self.split_docs(docs)


if __name__ == "__main__":
    # Demo: no embedding needed until split() runs; the model is downloaded
    # lazily on first use.
    try:
        sample = Document(
            page_content=(
                "RAG stands for retrieval-augmented generation. "
                "It combines a retriever with a generator. "
                "The retriever finds relevant passages from a knowledge base. "
                "The generator writes an answer conditioned on those passages. "
                "This is a totally different topic now. "
                "Semantic splitters group sentences by meaning. "
                "Nearby sentences with similar meaning stay in one chunk. "
            ),
            metadata={"source": "demo.txt"},
        )
        splitter = SemanticSplitter()
        result = splitter.split([sample])
        print(f"Split 1 document into {len(result)} semantic chunk(s)")
        if result:
            print(f"First chunk ({len(result[0].page_content.split())} words):")
            print(result[0].page_content[:120] + "...")
    except ImportError:
        print(
            "SKIP: SemanticSplitter demo needs sentence-transformers: "
            "pip install sentence-transformers"
        )
