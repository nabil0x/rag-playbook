"""Evidence-based claim verification for the SciFact pipeline.

LangChain has no turnkey "given a claim and candidate passages, decide
SUPPORTED / REFUTED / NOT_ENOUGH_INFO" flow, so this module fills the gap:
dense retrieval over a corpus (any object exposing ``embed_documents``) plus
an LLM verdict pass (any object exposing ``json_object``).

The verdict contract mirrors SciFact's three-way label set:
- ``SUPPORTED``      — the retrieved evidence backs the claim,
- ``REFUTED``        — the retrieved evidence contradicts the claim,
- ``NOT_ENOUGH_INFO``— the evidence neither supports nor refutes it
  (irrelevant, or a parse failure on the LLM side).

Everything is deliberately tolerant: a local 7B model is not a reliable
structured-output device, so ``verify_claim`` never raises and never returns
a verdict outside the three-value set. Retrieval is cosine over embeddings
computed in pure Python (no vector store required).

Used by curriculum/10-agentic-rag/ lab 03.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class JsonLLM(Protocol):
    def json_object(self, prompt: str) -> dict | list: ...


#: The only verdicts the pipeline knows how to report.
VALID_VERDICTS = frozenset({"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"})


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def retrieve_evidence(
    question_or_claim: str,
    corpus_texts: list[str],
    embedder: Embedder,
    top_k: int = 5,
    corpus_embeddings: list[list[float]] | None = None,
) -> list[dict]:
    """Rank ``corpus_texts`` against the claim and return the top ``top_k``.

    ``corpus_embeddings`` (optional) lets a caller embed a large corpus once
    and reuse the vectors across many claims — the whole point for the
    5183-doc SciFact corpus, where per-claim re-embedding is wasteful.
    """
    query_vec = embedder.embed_documents([question_or_claim])[0]
    if corpus_embeddings is None:
        corpus_embeddings = embedder.embed_documents(list(corpus_texts))
    scored = [
        (_cosine(query_vec, vec), index)
        for index, vec in enumerate(corpus_embeddings)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "text": corpus_texts[index],
            "index": index,
            "score": round(score, 4),
        }
        for score, index in scored[:top_k]
    ]


def verify_claim(
    llm: JsonLLM,
    claim: str,
    evidence_texts: list[str],
) -> dict:
    """Ask the LLM for a three-way verdict on ``claim`` against the evidence.

    Never crashes: a bad/absent JSON verdict falls back to
    ``{"verdict": "NOT_ENOUGH_INFO", "reason": "parse failure"}``.
    """
    evidence_block = "\n\n---\n\n".join(
        f"[{i + 1}] {text}" for i, text in enumerate(evidence_texts)
    )
    prompt = (
        "You are an evidence verifier for scientific claims.\n"
        "Decide whether the evidence passages support, refute, or say "
        "nothing about the claim.\n"
        "Rules:\n"
        '- Verdict must be EXACTLY one of: "SUPPORTED", "REFUTED", '
        '"NOT_ENOUGH_INFO".\n'
        '- "SUPPORTED": the evidence explicitly backs the claim.\n'
        '- "REFUTED": the evidence explicitly contradicts the claim.\n'
        '- "NOT_ENOUGH_INFO": the evidence neither supports nor refutes '
        "the claim (including when it is irrelevant).\n"
        '- Output ONLY JSON: {"verdict": "SUPPORTED", '
        '"reason": "one-sentence justification"}\n'
        "\n"
        f"Claim:\n{claim}\n\n"
        f"Evidence:\n{evidence_block}"
    )
    result = llm.json_object(prompt)
    if isinstance(result, dict) and "error" not in result:
        verdict = str(result.get("verdict", "")).strip().upper()
        if verdict in VALID_VERDICTS:
            reason = str(result.get("reason", "")).strip()
            return {"verdict": verdict, "reason": reason or "no reason given"}
    return {"verdict": "NOT_ENOUGH_INFO", "reason": "parse failure"}


def verify_loop(
    llm: JsonLLM,
    embedder: Embedder,
    claims: list[str],
    corpus_texts: list[str],
    top_k: int = 5,
    progress: Callable[[int, int], None] | None = None,
    corpus_embeddings: list[list[float]] | None = None,
) -> list[dict]:
    """Verify every claim: retrieve top-k evidence, then ask the LLM.

    The corpus is embedded exactly once (either precomputed via
    ``corpus_embeddings`` or here on first use) and reused for all claims.
    Returns one dict per claim:
    ``{"claim", "verdict", "reason", "evidence_indices"}``.
    """
    if corpus_embeddings is None:
        corpus_embeddings = embedder.embed_documents(list(corpus_texts))
    results: list[dict] = []
    total = len(claims)
    for done, claim in enumerate(claims, start=1):
        hits = retrieve_evidence(
            claim,
            corpus_texts,
            embedder,
            top_k=top_k,
            corpus_embeddings=corpus_embeddings,
        )
        verdict = verify_claim(llm, claim, [hit["text"] for hit in hits])
        results.append(
            {
                "claim": claim,
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
                "evidence_indices": [hit["index"] for hit in hits],
            }
        )
        if progress is not None:
            progress(done, total)
    return results


if __name__ == "__main__":
    # No-network smoke test: a keyword-overlap fake embedder proves the
    # retrieval ordering, and canned-JSON fakes prove the verdict parsing
    # (including the never-crash parse-failure fallback).

    class _KeywordEmbedder:
        """Fake embedder: one dimension per keyword, 1.0 when present."""

        _TOKENS = ("paris", "france", "capital")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [
                [1.0 if token in text.lower() else 0.0 for token in self._TOKENS]
                for text in texts
            ]

    class _GoodLLM:
        def json_object(self, prompt: str) -> dict:
            return {"verdict": "SUPPORTED", "reason": "the evidence confirms it"}

    class _BadLLM:
        def json_object(self, prompt: str) -> dict:
            return {"error": "could not parse JSON after 3 attempts", "raw": "..."}

    corpus = [
        "Paris is a city in France.",
        "The capital of France is Paris.",
        "Bananas are yellow fruit.",
        "France borders Germany.",
    ]
    claim = "Paris is the capital of France."

    top = retrieve_evidence(claim, corpus, _KeywordEmbedder(), top_k=2)
    assert len(top) == 2, f"expected 2 hits, got {len(top)}"
    assert all({"text", "index", "score"} <= set(hit) for hit in top), (
        "hits must carry text/index/score"
    )
    assert top[0]["text"] == "The capital of France is Paris.", (
        f"best hit should be the exact sentence, got {top[0]['text']!r}"
    )
    assert top[0]["score"] > top[1]["score"]

    good = verify_claim(_GoodLLM(), claim, [hit["text"] for hit in top])
    assert good["verdict"] == "SUPPORTED" and good["reason"], good

    bad = verify_claim(_BadLLM(), claim, [hit["text"] for hit in top])
    assert bad == {"verdict": "NOT_ENOUGH_INFO", "reason": "parse failure"}, bad

    results = verify_loop(
        _GoodLLM(),
        _KeywordEmbedder(),
        [claim],
        corpus,
        top_k=2,
        corpus_embeddings=_KeywordEmbedder().embed_documents(corpus),
    )
    assert len(results) == 1
    assert results[0]["verdict"] == "SUPPORTED"
    assert results[0]["evidence_indices"] == [top[0]["index"], top[1]["index"]]

    print("OK: retrieve_evidence returns top-k dicts and verify_claim parses verdicts")
