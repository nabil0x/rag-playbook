"""Token-based text splitter.

Splitter block: chunk by token count instead of characters.
Compare chunk count / average size / retrieval quality against
RecursiveCharacterTextSplitter.
See Topics/Project-18-RAG-Benchmark-Suite/README.md.
"""


class TokenSplitter:
    """Split documents on token boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: list) -> list:
        # TODO: implement with TokenTextSplitter (langchain_text_splitters)
        raise NotImplementedError
