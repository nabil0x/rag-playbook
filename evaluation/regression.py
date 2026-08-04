"""Golden-set prompt regression gate.

Evaluation block (capstone): every prompt edit becomes a CI event — run the
golden QA set through the pipeline with the candidate prompt, score the
answers with the Project 20 judge, and fail the change when faithfulness or
relevance drops below the baseline. See Topics/Project-36-Drift-Prompt-Regression/README.md.
"""

from __future__ import annotations

from typing import Any


def _flatten_golden(golden: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Turn the ``evaluation.golden.GOLDEN_QA`` dict into one flat (question, expected) list."""
    pairs: list[tuple[str, str]] = []
    for questions in golden.values():
        pairs.extend(questions)
    return pairs


class PromptRegressionSuite:
    """Run a candidate prompt over the golden set and compare against baseline.

    ``pipeline_factory(prompt)`` builds a RAG pipeline using that prompt (the
    Project 17 swap-a-block idea applied to the prompt block); ``judge`` scores
    each answer (the Project 20 LLM-as-judge). The suite's job is to run, score,
    compare, and pass/fail — the same shape a CI gate needs.
    """

    def __init__(
        self,
        pipeline_factory,
        judge,
        golden: dict[str, list[tuple[str, str]]] | None = None,
        baseline: dict[str, float] | None = None,
    ):
        # pipeline_factory: callable(prompt) -> object with .ask(question) -> str.
        # judge: any object exposing judge(instruction, prompt) -> dict with
        # faithfulness/relevance keys (evaluation.judge.LLMJudge).
        # golden: GOLDEN_QA dict; defaults to the Project 20 golden set.
        # baseline: {"faithfulness": f, "relevance": r}; None -> computed on
        # the first run via compare(scores_from_current_prompt).
        self.pipeline_factory = pipeline_factory
        self.judge = judge
        if golden is None:
            try:
                from evaluation.golden import GOLDEN_QA
            except ImportError:
                GOLDEN_QA = {}
            golden = GOLDEN_QA
        self.golden = _flatten_golden(golden)
        self.baseline = baseline

    def run(self, prompt) -> list[float]:
        """TODO(Project 36): score every golden pair through the pipeline.

        For each (question, expected): build the pipeline with ``prompt``,
        ask, judge the answer (faithfulness + relevance averaged, or whatever
        metric your Project 20 setup produces), and collect one score per
        pair. Return the per-pair score list — the input to ``compare``.
        """
        raise NotImplementedError("TODO(Project 36): implement PromptRegressionSuite.run")

    def compare(self, scores: list[float]) -> dict[str, Any]:
        """TODO(Project 36): the regression verdict.

        Return ``{"passed": bool, "mean": float, "delta": float}`` where
        ``delta`` is the mean score minus the baseline mean (negative =
        regression) and ``passed`` is True when there is no baseline yet or
        when delta >= 0. This dict is what a CI step prints and exits on.
        """
        raise NotImplementedError("TODO(Project 36): implement PromptRegressionSuite.compare")


if __name__ == "__main__":
    # No-network smoke test — construction and golden-set loading only
    # (run/compare are learner TODOs and are not invoked).
    import os
    import sys

    # Direct execution puts evaluation/ on sys.path; the golden set and the
    # evaluation package live one level up at the repo root.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    flat = _flatten_golden({"doc": [("q1", "a1"), ("q2", "a2")]})
    assert flat == [("q1", "a1"), ("q2", "a2")]

    suite = PromptRegressionSuite(pipeline_factory=lambda prompt: None, judge=object())
    assert len(suite.golden) > 0  # the Project 20 golden set ships with the repo

    suite_empty = PromptRegressionSuite(pipeline_factory=lambda prompt: None, judge=object(), golden={})
    assert suite_empty.golden == []

    print(f"OK: PromptRegressionSuite loaded {len(suite.golden)} golden QA pairs")
