"""Semantic text splitter.

Splitter block: chunk on meaning instead of fixed character counts.
See Topics/Project-17-Modular-RAG-Framework/README.md.
"""


class SemanticSplitter:
    """Split documents into semantically coherent chunks."""

    def __init__(self, embedding=None, breakpoint_percentile: float = 95.0):
        self.embedding = embedding
        self.breakpoint_percentile = breakpoint_percentile

    def split(self, docs: list) -> list:
        # TODO: implement (e.g. via langchain_experimental SemanticChunker)
        raise NotImplementedError
