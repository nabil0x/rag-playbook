"""CSV document loader.

Loader block: swap the loader, keep the rest of the pipeline.
See Topics/Project-13-Multi-format-RAG/README.md.
"""


class CSVLoader:
    """Load rows from a CSV file. Returns a list of langchain Documents."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list:
        # TODO: implement with CSVLoader (langchain_community.document_loaders)
        raise NotImplementedError
