"""Token-based text splitter.

Splitter block: chunk by token count instead of characters.
TokenTextSplitter counts tokens (via tiktoken, the tokenizer used by
OpenAI models) instead of raw characters, so every chunk stays within a
token budget that the LLM's context window actually understands.

Compare chunk count / average size / retrieval quality against
RecursiveCharacterTextSplitter — token-based splitting produces more
uniformly sized chunks, while recursive character splitting keeps words
and paragraphs intact.

See Topics/Project-18-RAG-Benchmark-Suite/README.md.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter


class TokenSplitter:
    """Split documents on token boundaries.

    Wraps langchain_text_splitters.TokenTextSplitter so it fits the
    pipeline's splitter contract (``split_docs``/``split``), letting you
    swap it for RecursiveCharacterTextSplitter (``DocumentProcessor``) or
    ``SemanticSplitter`` without touching the rest of the pipeline.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Store the token budget and build the underlying TokenTextSplitter.

        Args:
            chunk_size: maximum number of tokens per chunk.
            chunk_overlap: number of tokens shared between neighbouring
                chunks, preserving context across boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_docs(self, docs: list[Document]) -> list[Document]:
        """Split documents into token-budgeted chunks (canonical entry point)."""
        return self.text_splitter.split_documents(docs)

    def split(self, docs: list[Document]) -> list[Document]:
        """Split documents into token-budgeted chunks (alias of ``split_docs``)."""
        return self.split_docs(docs)


if __name__ == "__main__":
    sample = Document(
        page_content=(
            "Token-based splitting counts tokens, not characters, when it "
            "decides where to cut. Because tiktoken mirrors the tokenizer "
            "the model itself uses, each chunk fits the model's context "
            "window exactly. Character-based splitters can produce chunks "
            "that are far larger, in token terms, than their character "
            "count suggests. " * 5
        ),
        metadata={"source": "demo.txt"},
    )
    splitter = TokenSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split([sample])
    import tiktoken  # same tokenizer TokenTextSplitter uses by default

    enc = tiktoken.encoding_for_model("gpt-4")
    avg_tokens = sum(len(enc.encode(c.page_content)) for c in chunks) / len(chunks) if chunks else 0
    print(f"Split 1 document into {len(chunks)} token-budgeted chunk(s)")
    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    if chunks:
        print(f"First chunk starts: {chunks[0].page_content[:80]}...")
