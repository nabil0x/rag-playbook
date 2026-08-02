"""PDF document loader.

Loader block: replace `WebBaseLoader` with `PyPDFLoader`.
Every loader returns `List[Document]` — nothing else changes.
See Topics/Project-02-PDF-Knowledge-Base/README.md.
"""


class PDFLoader:
    """Load text from PDF files. Returns a list of langchain Documents."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list:
        # TODO: implement with PyPDFLoader (langchain_community.document_loaders)
        raise NotImplementedError
