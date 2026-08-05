"""RAGAS cross-validation wrapper for the from-scratch metrics.

Evaluation block: run the same four generation metrics through the ragas 0.4.3
library, using the SAME local judge (Ollama) and SAME local embeddings, so the
notebook can compare our from-scratch scores against an independent
implementation.
See Topics/Project-20-Deep-Eval/README.md.
"""

from __future__ import annotations


def ragas_scores(samples: list[dict], llm, embeddings) -> list[dict]:
    """Score a list of RAG samples with ragas 0.4.3.

    Each sample dict must have the keys ``user_input``, ``response``,
    ``retrieved_contexts`` (list of strings), and ``reference``. Returns one
    dict per sample with the four scores: ``faithfulness``,
    ``answer_relevancy``, ``context_precision``, ``context_recall``.

    Args:
        samples: list of sample dicts (see above).
        llm: a langchain chat model instance (e.g. ChatOllama) used as judge.
        embeddings: a ``LocalEmbeddings`` instance (never the community
            FastEmbedEmbeddings — it leaks usage metadata and breaks ragas).

    Raises:
        ImportError: if ragas is not installed.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "ragas_scores needs ragas: pip install ragas"
        ) from exc

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
            )
            for s in samples
        ]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return df.to_dict(orient="records")


if __name__ == "__main__":
    print("ragas_metrics module imports cleanly (no ragas call made).")
