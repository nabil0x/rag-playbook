"""MMR retriever (Maximum Marginal Relevance).

Retriever block: trade relevance against diversity.
`retriever = vectorstore.as_retriever(search_type="mmr")`.
See Topics/Project-08-MMR-Retrieval/README.md.
"""


class MMRRetriever:
    """Return diverse, relevant chunks using MMR."""

    def __init__(self, vector_store, top_k: int = 5, lambda_mult: float = 0.5):
        self.vector_store = vector_store
        self.top_k = top_k
        self.lambda_mult = lambda_mult

    def retrieve(self, question: str) -> list:
        # TODO: implement MMR over the vector store
        raise NotImplementedError
