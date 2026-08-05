"""Evaluation harness: run generation metrics over a golden question set.

Evaluation block: the orchestrator. For each golden question it retrieves
context, generates an answer, scores it with the from-scratch metrics, then
aggregates results and prints aligned tables. Also provides a from-scratch
Cohen's kappa for LLM-as-judge vs human agreement.
See Topics/Project-20-Deep-Eval/README.md.
"""

from __future__ import annotations


class MetricResult:
    """One evaluated question.

    Args:
        doc: document key the question belongs to.
        question: the question asked.
        reference: the hand-checked reference answer.
        context: the retrieved context joined as text (top-3 chunks).
        answer: the generated answer.
        faithfulness: from-scratch faithfulness score (0..1).
        answer_relevance: from-scratch answer relevance score (0..1).
        context_precision: from-scratch context precision score (0..1).
        context_recall: from-scratch context recall score (0..1).
    """

    def __init__(
        self,
        doc: str,
        question: str,
        reference: str,
        context: str,
        answer: str,
        faithfulness: float = 0.0,
        answer_relevance: float = 0.0,
        context_precision: float = 0.0,
        context_recall: float = 0.0,
    ):
        self.doc = doc
        self.question = question
        self.reference = reference
        self.context = context
        self.answer = answer
        self.faithfulness = faithfulness
        self.answer_relevance = answer_relevance
        self.context_precision = context_precision
        self.context_recall = context_recall


class EvaluationHarness:
    """Run the four from-scratch metrics over a golden question list.

    Args:
        judge: an ``LLMJudge`` instance (ask + embed).
        retriever: duck-typed retriever exposing ``retrieve(question) -> list[Document]``.
        faithfulness_metric: a ``FaithfulnessMetric`` instance.
        answer_relevance_metric: an ``AnswerRelevanceMetric`` instance.
        context_precision_metric: a ``ContextPrecisionMetric`` instance.
        context_recall_metric: a ``ContextRecallMetric`` instance.
        top_k: how many retrieved chunks to feed as context.
    """

    def __init__(
        self,
        judge,
        retriever,
        faithfulness_metric,
        answer_relevance_metric,
        context_precision_metric,
        context_recall_metric,
        top_k: int = 3,
    ):
        self.judge = judge
        self.retriever = retriever
        self.faithfulness_metric = faithfulness_metric
        self.answer_relevance_metric = answer_relevance_metric
        self.context_precision_metric = context_precision_metric
        self.context_recall_metric = context_recall_metric
        self.top_k = top_k

    @staticmethod
    def _to_text(chunks) -> str:
        """Join retrieved chunks (Documents or plain strings) into one text."""
        texts = []
        for chunk in chunks:
            text = getattr(chunk, "page_content", chunk)
            texts.append(text)
        return "\n\n".join(texts)

    def run(self, questions: list[dict]) -> list[MetricResult]:
        """Evaluate every question: retrieve, generate, score.

        Args:
            questions: list of dicts with keys ``doc``, ``question``, ``reference``.

        Raises:
            RuntimeError: if retrieval returns nothing for a question.
        """
        results: list[MetricResult] = []
        for q in questions:
            docs = self.retriever.retrieve(q["question"])[: self.top_k]
            if not docs:
                raise RuntimeError(
                    f"EvaluationHarness: retriever returned no chunks for {q['question']!r}"
                )
            context = self._to_text(docs)
            answer = self.judge.ask(q["question"], context)
            results.append(
                MetricResult(
                    doc=q["doc"],
                    question=q["question"],
                    reference=q["reference"],
                    context=context,
                    answer=answer,
                    faithfulness=self.faithfulness_metric.score(
                        q["question"], context, answer
                    ),
                    answer_relevance=self.answer_relevance_metric.score(
                        q["question"], answer
                    ),
                    context_precision=self.context_precision_metric.score(
                        q["question"], docs[: self.top_k]
                    ),
                    context_recall=self.context_recall_metric.score(
                        q["question"], context, q["reference"]
                    ),
                )
            )
        return results

    def aggregate(self, results: list[MetricResult]) -> dict:
        """Return overall and per-doc mean scores.

        Returns:
            dict with ``overall`` (dict of metric means) and ``by_doc``
            (dict mapping doc key to its metric means).
        """
        metrics = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
        ]

        def means(rows: list[MetricResult]) -> dict:
            out: dict[str, float] = {}
            for name in metrics:
                values = [getattr(r, name) for r in rows]
                out[name] = sum(values) / len(values) if values else 0.0
            return out

        by_doc: dict[str, list[MetricResult]] = {}
        for r in results:
            by_doc.setdefault(r.doc, []).append(r)
        return {
            "overall": means(results),
            "by_doc": {doc: means(rows) for doc, rows in by_doc.items()},
        }

    @staticmethod
    def kappa(labels_a: list[int], labels_b: list[int]) -> float:
        """Cohen's kappa for two binary label lists, computed from scratch.

        Args:
            labels_a: first annotator's binary labels (0/1).
            labels_b: second annotator's binary labels (0/1).

        Raises:
            ValueError: if the label lists differ in length.
        """
        if len(labels_a) != len(labels_b):
            raise ValueError(
                "EvaluationHarness.kappa: label lists must have equal length"
            )
        n = len(labels_a)
        if n == 0:
            return 0.0
        a11 = sum(1 for x, y in zip(labels_a, labels_b) if x == 1 and y == 1)
        a10 = sum(1 for x, y in zip(labels_a, labels_b) if x == 1 and y == 0)
        a01 = sum(1 for x, y in zip(labels_a, labels_b) if x == 0 and y == 1)
        a00 = sum(1 for x, y in zip(labels_a, labels_b) if x == 0 and y == 0)
        po = (a11 + a00) / n
        pe = ((a11 + a10) * (a11 + a01) + (a01 + a00) * (a10 + a00)) / (n * n)
        if pe == 1.0:
            return 0.0
        return (po - pe) / (1 - pe)

    def print_table(self, results: list[MetricResult]) -> None:
        """Print an aligned text table of every scored question."""
        header = f"{'doc':<20} {'question':<38} {'faith':>5} {'rel':>5} {'prec':>5} {'rec':>5} answer"
        print(header)
        print("-" * len(header))
        for r in results:
            answer = r.answer.replace("\n", " ")[:40]
            print(
                f"{r.doc:<20} {r.question[:38]:<38} "
                f"{r.faithfulness:>5.2f} {r.answer_relevance:>5.2f} "
                f"{r.context_precision:>5.2f} {r.context_recall:>5.2f} {answer}"
            )


if __name__ == "__main__":
    # Sanity check: kappa on toy labels.
    a = [1, 1, 0, 0, 1]
    b = [1, 1, 0, 1, 1]
    print(f"toy kappa = {EvaluationHarness.kappa(a, b):.3f}")
