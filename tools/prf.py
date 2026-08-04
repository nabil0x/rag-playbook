"""Pseudo-relevance feedback (PRF) retriever.

Retriever block: harvest the top terms from the top-k documents of a
first-stage retrieval, append them to the query, and retrieve again with
the expanded query. The extra terms pull the second-stage retrieval toward
the vocabulary of the documents the first stage already judged relevant —
a cheap, LLM-free stand-in for query expansion. See the 05-query-transformation
labs for the empirical comparison against the plain top-k baseline.
"""

import re
from collections import Counter

from langchain_core.documents import Document


# Small default stopword set — enough for the labs' corpus; pass your own
# ``stopwords`` to PRFRetriever to swap it.
PRF_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his how i if in
    is it its of on or she that the their them then there these they this to
    was were what when where which who why with you your""".split()
)


class PRFRetriever:
    """Two-stage retrieval with term-level query expansion.

    Wraps any retriever exposing ``retrieve(question) -> list[Document]``
    (e.g. a ``SimilarityRetriever``). Stage 1 retrieves with the raw query
    and keeps the top ``feedback_k`` documents; stage 2 appends the top
    ``n_terms`` tokens harvested from those documents (excluding stopwords
    and terms already in the query) and retrieves again, returning the top
    ``top_k`` of the expanded retrieval.
    """

    def __init__(
        self,
        retriever,
        top_k: int = 5,
        feedback_k: int = 3,
        n_terms: int = 5,
        min_term_len: int = 3,
        stopwords: set[str] | None = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.feedback_k = feedback_k
        self.n_terms = n_terms
        self.min_term_len = min_term_len
        self.stopwords = stopwords if stopwords is not None else PRF_STOPWORDS

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase alphanumeric tokens of ``text``."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _feedback_terms(
        self, question: str, docs: list[Document], n_terms: int
    ) -> list[str]:
        """Top ``n_terms`` tokens by frequency across ``docs``.

        Query terms, stopwords, and tokens shorter than ``min_term_len`` are
        excluded — an expansion term must add information the query does not
        already carry.
        """
        query_tokens = set(self._tokenize(question))
        counts: Counter[str] = Counter()
        for doc in docs:
            for tok in self._tokenize(doc.page_content):
                if (
                    tok in query_tokens
                    or tok in self.stopwords
                    or len(tok) < self.min_term_len
                ):
                    continue
                counts[tok] += 1
        return [tok for tok, _ in counts.most_common(n_terms)]

    def retrieve(self, question: str) -> list[Document]:
        """Stage-1 feedback, expand the query, stage-2 retrieval."""
        feedback = self.retriever.retrieve(question)[: self.feedback_k]
        terms = self._feedback_terms(question, feedback, self.n_terms)
        expanded = " ".join([question, *terms]) if terms else question
        return self.retriever.retrieve(expanded)[: self.top_k]


if __name__ == "__main__":
    # Fake retriever - no network, runs anywhere. Returns docs whose
    # vocabulary overlaps the query so the expansion terms are deterministic.
    class _FakeRetriever:
        """Echoes each query as a document with fixed extra vocabulary."""

        def __init__(self):
            self.questions_seen: list[str] = []

        def retrieve(self, question: str) -> list[Document]:
            self.questions_seen.append(question)
            return [
                Document(page_content="federal reserve banking regulation"),
                Document(page_content="bank capital requirements oversight"),
                Document(page_content="monetary policy inflation target"),
            ]

    inner = _FakeRetriever()
    prf = PRFRetriever(inner, top_k=2, feedback_k=2, n_terms=3)
    question = "How does the fed set interest rates?"
    docs = prf.retrieve(question)

    # Stage 1 issued the raw query; stage 2 the expanded one.
    assert len(inner.questions_seen) == 2, inner.questions_seen
    expanded = inner.questions_seen[1]
    suffix = expanded[len(question):].strip()  # the expansion portion only
    # Harvested terms must come from the feedback docs' vocabulary.
    assert suffix == "federal reserve banking", suffix
    # Query terms must not reappear as expansion.
    for term in ("fed", "interest", "rates"):
        assert term not in suffix.split(), suffix
    assert len(docs) == 2
    print(f"OK: stage-2 query = {expanded!r}")
