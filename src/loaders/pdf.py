"""PDF document loader.

Loader block: replace `WebBaseLoader` with `PyPDFLoader`.
Every loader returns `List[Document]` — nothing else changes.
See Topics/Project-02-PDF-Knowledge-Base/README.md.
"""

import sys

from langchain_core.documents import Document


class PDFLoader:
    """Load text from PDF files. Returns a list of langchain Documents."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list[Document]:
        try:
            import pypdf  # noqa: F401  (checked before use, see below)
        except ImportError as exc:
            raise ImportError("PDFLoader needs pypdf: pip install pypdf") from exc

        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as exc:
            raise ImportError(
                "PDFLoader needs langchain-community: pip install langchain-community"
            ) from exc

        try:
            return PyPDFLoader(self.path).load()
        except (ImportError, RuntimeError) as exc:
            raise ImportError("PDFLoader needs pypdf: pip install pypdf") from exc


if __name__ == "__main__":
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else "Data/sample.pdf"
        docs = PDFLoader(path).load()
        print(f"Loaded {len(docs)} document(s) from {path}")
        if docs:
            print(docs[0].page_content[:200])
    except ImportError:
        print("SKIP: PDFLoader demo needs pypdf: pip install pypdf")
