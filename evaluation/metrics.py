"""From-scratch RAGAS-style generation metrics.

Evaluation block: the educational core — faithfulness, answer relevance,
context precision, and context recall implemented by hand (no ragas imports).
Each metric holds an ``LLMJudge`` (and ``LocalEmbeddings`` where needed) as a
collaborator passed in ``__init__``.
See Topics/Project-20-Deep-Eval/README.md.
"""

from __future__ import annotations


class FaithfulnessMetric:
    """Score how faithful an answer is to the context (from scratch).

    Asks the judge to (1) list the claims in the answer and (2) mark each as
    supported by the context — in ONE judge call per claim-set using a strict
    JSON schema ``{"claims": [...], "supported": [true/false, ...]}``.
    Score = supported / claims (0.0 if no claims).

    Args:
        judge: an ``LLMJudge`` instance.
    """

    def __init__(self, judge):
        self.judge = judge

    def score(self, question: str, context: str, answer: str) -> float:
        """Return the fraction of the answer's claims supported by context."""
        instruction = (
            "You are a faithfulness judge. Break the answer into atomic "
            "claims, then mark each claim as supported (true) or not (false) "
            "by the context."
        )
        prompt = (
            f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
            'Return JSON: {"claims": ["..."], "supported": [true, false, ...]}'
        )
        result = self.judge.judge(instruction, prompt)
        if "error" in result:
            return 0.0
        claims = result.get("claims", [])
        supported = result.get("supported", [])
        n = min(len(claims), len(supported))
        if n == 0:
            return 0.0
        return sum(1 for flag in supported[:n] if flag) / n


class AnswerRelevanceMetric:
    """Embedding cosine similarity between the question and the answer.

    Short exact answers score low — that is expected and discussed in the
    notebook: a terse "610.00" shares little token overlap with the question's
    embedding.

    Args:
        judge: an ``LLMJudge`` instance (used for its local embedder).
    """

    def __init__(self, judge):
        self.judge = judge

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors (0.0 if either is zero)."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def score(self, question: str, answer: str) -> float:
        """Return the cosine similarity between question and answer embeddings."""
        q = self.judge.embed([question])[0]
        a = self.judge.embed([answer])[0]
        return self._cosine(q, a)


class ContextPrecisionMetric:
    """Score how precisely the retrieved context answers the question.

    The judge labels each context chunk relevant/not to the question in ONE
    call (JSON ``{"relevant": [bool, ...]}``). precision@k = mean of
    (relevant_count_at_k / k) over k where chunk k is relevant (RAGAS
    definition).

    Args:
        judge: an ``LLMJudge`` instance.
    """

    def __init__(self, judge):
        self.judge = judge

    def score(self, question: str, context: list[str]) -> float:
        """Return the RAGAS context precision over the ranked chunks."""
        instruction = (
            "You are a retrieval judge. For each context chunk, decide whether "
            "it is relevant to answering the question."
        )
        numbered = "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(context))
        prompt = (
            f"Question: {question}\n\nContext chunks:\n{numbered}\n\n"
            'Return JSON: {"relevant": [true, false, ...]}'
        )
        result = self.judge.judge(instruction, prompt)
        if "error" in result:
            return 0.0
        relevant = result.get("relevant", [])
        if not relevant:
            return 0.0
        # RAGAS precision@k: mean over k where chunk k is relevant of
        # (relevant_count_at_k / k).
        scores = []
        count = 0
        for k, flag in enumerate(relevant, start=1):
            if flag:
                count += 1
                scores.append(count / k)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


class ContextRecallMetric:
    """Score whether the context contains the facts needed for the answer.

    The judge checks whether each claim of the reference answer appears in the
    context (JSON ``{"present": [bool, ...]}``). Score = present / total.

    Args:
        judge: an ``LLMJudge`` instance.
    """

    def __init__(self, judge):
        self.judge = judge

    def score(
        self, question: str, context: str, reference_answer: str
    ) -> float:
        """Return the fraction of reference claims present in the context."""
        instruction = (
            "You are a recall judge. Break the reference answer into atomic "
            "claims, then mark each claim present (true) or absent (false) in "
            "the context."
        )
        prompt = (
            f"Context:\n{context}\n\nReference answer:\n{reference_answer}\n\n"
            'Return JSON: {"present": [true, false, ...]}'
        )
        result = self.judge.judge(instruction, prompt)
        if "error" in result:
            return 0.0
        present = result.get("present", [])
        if not present:
            return 0.0
        return sum(1 for flag in present if flag) / len(present)


if __name__ == "__main__":
    print("metrics module imports cleanly (no judge calls made).")