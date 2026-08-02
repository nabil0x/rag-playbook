"""CSV document loader.

Loader block: swap the loader, keep the rest of the pipeline.
See Topics/Project-13-Multi-format-RAG/README.md.
"""

import sys

from langchain_core.documents import Document


class CSVLoader:
    """Load rows from a CSV file. Returns a list of langchain Documents."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list[Document]:
        try:
            from langchain_community.document_loaders import CSVLoader
        except ImportError as exc:
            raise ImportError(
                "CSVLoader needs langchain-community: pip install langchain-community"
            ) from exc
        return CSVLoader(self.path).load()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "NoteBooks/Data/sample.csv"
    docs = CSVLoader(path).load()
    print(f"Loaded {len(docs)} document(s) from {path}")
    if docs:
        print(docs[0].page_content)
