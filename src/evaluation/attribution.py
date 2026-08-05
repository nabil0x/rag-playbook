"""Citations and attribution.

Evaluation block: split an LLM answer into atomic claims, check each claim
against the retrieved context chunks (exact match first, embedding similarity
as the fuzzy fallback), and format the answer with inline [1] [2] citations.
See Topics/Project-28-Citations-Attribution/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class Citation:
    """One claim-level attribution result.

    Attributes:
        claim: an atomic statement extracted from the answer.
        source_doc_id: metadata id of the chunk that (may) support it.
        span: the matching text span inside the chunk, if any.
        supported: whether the claim is grounded in the context.
    """

    claim: str
    source_doc_id: str = ""
    span: str = ""
    supported: bool = False


class AttributionEvaluator:
    """Check each claim of an answer against the retrieved context.

    Two-stage support check per claim: exact substring match against any
    chunk text first (cheap, decisive), then embedding cosine similarity
    above ``threshold`` against the best-matching chunk (fuzzy fallback).
    """

    #: Cosine threshold for the fuzzy (embedding) support check.
    DEFAULT_THRESHOLD = 0.8

    def __init__(self, embedder=None, threshold: float = DEFAULT_THRESHOLD):
        # embedder: any object exposing embed_documents(texts) -> list[list[float]]
        # (e.g. evaluation.judge.LocalEmbeddings). None disables the fuzzy step
        # — only exact matches are then considered supported.
        self.embedder = embedder
        self.threshold = threshold

    def _split_claims(self, answer: str) -> list[str]:
        """TODO(Project 28): split ``answer`` into atomic claims.

        Naive but effective first pass: split on sentence boundaries ("." ),
        strip each piece, and drop empty or one-word fragments. A claim is
        the unit that gets checked — finer than the whole answer, coarser
        than words. (An LLM claim extractor is the Stretch upgrade.)
        """
        raise NotImplementedError("TODO(Project 28): implement AttributionEvaluator._split_claims")

    def _exact_match(self, claim: str, chunks: list[Document]) -> tuple[str, str] | None:
        """TODO(Project 28): find an exact substring match for ``claim``.

        Return ``(doc_id, span)`` for the first chunk whose ``page_content``
        contains ``claim`` (or contains a significant fragment of it, e.g.
        the claim without trailing punctuation) — or None when no chunk
        matches. The span can be the matched substring itself.
        """
        raise NotImplementedError("TODO(Project 28): implement AttributionEvaluator._exact_match")

    def _fuzzy_match(self, claim: str, chunks: list[Document]) -> tuple[str, str] | None:
        """TODO(Project 28): find the best embedding-similarity match.

        When ``self.embedder`` is set, embed the claim and all chunk texts in
        one ``embed_documents`` call, compute cosine similarities, and return
        ``(doc_id, span)`` for the best chunk when its similarity exceeds
        ``self.threshold`` — else None. Without an embedder, return None.
        """
        raise NotImplementedError("TODO(Project 28): implement AttributionEvaluator._fuzzy_match")

    def evaluate(self, answer: str, context_documents: list[Document]) -> list[Citation]:
        """Return a Citation per claim of ``answer``.

        Runs exact match first, falls back to fuzzy match, and marks the
        claim supported only when one of the two finds a chunk.
        """
        citations: list[Citation] = []
        for claim in self._split_claims(answer):
            hit = self._exact_match(claim, context_documents) or self._fuzzy_match(claim, context_documents)
            if hit is None:
                citations.append(Citation(claim=claim))
            else:
                doc_id, span = hit
                citations.append(Citation(claim=claim, source_doc_id=doc_id, span=span, supported=True))
        return citations

    def groundedness(self, citations: list[Citation]) -> float:
        """Fraction of claims that are supported (1.0 = fully grounded)."""
        if not citations:
            return 0.0
        return sum(1 for c in citations if c.supported) / len(citations)


class CitationFormatter:
    """Format an answer with inline [1] [2] markers and a sources list.

    The pipeline prompt (``src/prompts/citation.py``) asks the LLM to emit
    citations; this formatter normalizes them into a trailing source list
    so every ``[n]`` maps to a real chunk.
    """

    def __init__(self, source_key: str = "source"):
        # source_key: metadata key holding a human-readable source name.
        self.source_key = source_key

    def format(self, answer: str, sources: list[Document]) -> str:
        """TODO(Project 28): return the answer plus a numbered source list.

        Expects ``answer`` to already contain inline ``[n]`` markers (as
        produced by the citation prompt). Append:

            Sources:
            [1] <source name or chunk preview>
            [2] ...

        built from ``sources`` in order, where the name comes from
        ``doc.metadata.get(self.source_key)`` with a truncated
        ``page_content`` fallback. Strip markers that point past the list.
        """
        raise NotImplementedError("TODO(Project 28): implement CitationFormatter.format")


if __name__ == "__main__":
    # No-network smoke test — the Project-28 stubs are filled in by a tiny
    # subclass so the evaluate()/groundedness() scaffold can be tested.
    chunks = [
        Document(page_content="Paris is the capital of France.", metadata={"id": "c1", "source": "notes.md"}),
        Document(page_content="The Seine is a river in Paris.", metadata={"id": "c2", "source": "wiki.md"}),
    ]

    class _Evaluator(AttributionEvaluator):
        """Minimal claim splitting + exact-match check, no fuzzy fallback."""

        def _split_claims(self, answer: str) -> list[str]:
            return [c for c in answer.split(". ") if c.strip()]

        def _exact_match(self, claim: str, docs: list[Document]) -> tuple[str, str] | None:
            for doc in docs:
                if claim in doc.page_content:
                    return (doc.metadata["id"], claim)
            return None

        def _fuzzy_match(self, claim: str, docs: list[Document]) -> tuple[str, str] | None:
            return None

    ev = _Evaluator(embedder=None)
    exact = ev.evaluate("Paris is the capital of France. Elephants fly.", chunks)
    assert exact[0].supported and "c1" in exact[0].source_doc_id
    assert not exact[1].supported  # hallucinated claim
    assert ev.groundedness(exact) == 0.5

    fmt = CitationFormatter()
    assert fmt.source_key == "source"

    print("OK: attribution marks supported/unsupported claims and groundedness reports the fraction")
