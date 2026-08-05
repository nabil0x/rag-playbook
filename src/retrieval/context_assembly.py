"""Context assembly and lost-in-the-middle.

Retriever block: the last mile between a raw retrieval list and the context
the LLM actually reads — deduplicate, reorder, and truncate to a token budget,
then measure the "lost in the middle" position bias (Liu et al. 2023).
See Topics/Project-27-Context-Assembly/README.md.
"""

from __future__ import annotations

from langchain_core.documents import Document


class ContextAssembler:
    """Dedupe, reorder, and truncate a retrieved document list.

    Wraps an optional embedder for near-duplicate detection (cosine > 0.9)
    and an optional token counter. Exact duplicates are always removed;
    near-duplicates only when an embedder is provided.
    """

    #: Cosine threshold above which two chunks count as near-duplicates.
    NEAR_DUP_THRESHOLD = 0.9

    def __init__(self, embedder=None, token_counter=None):
        # embedder: any object exposing embed_documents(texts) -> list[list[float]]
        # (e.g. evaluation.judge.LocalEmbeddings). None disables near-dup removal.
        # token_counter: any callable text -> int, or None to use a rough
        # character/4 estimate.
        self.embedder = embedder
        self.token_counter = token_counter

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors (fallback for near-dup check)."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def dedupe(self, documents: list[Document]) -> list[Document]:
        """TODO(Project 27): remove exact and near-duplicate chunks.

        - Exact: drop any document whose ``page_content`` was already seen.
        - Near (only when ``self.embedder`` is set): embed all unique chunks
          in one ``embed_documents`` call and drop any chunk whose cosine to
          an *earlier kept* chunk exceeds ``NEAR_DUP_THRESHOLD``. First
          occurrence wins; order is preserved.
        """
        raise NotImplementedError("TODO(Project 27): implement ContextAssembler.dedupe")

    def reorder(self, documents: list[Document], mode: str = "best_first") -> list[Document]:
        """TODO(Project 27): reorder by ``mode``.

        Modes:
        - "best_first": descending by ``metadata["score"]`` (fall back to the
          original order when scores are missing).
        - "retrieval_order": the input order, unchanged.
        - "answer_first": keep the highest-scoring chunk first, then the rest
          in original order (a simple answer-first heuristic).
        Unknown modes raise ValueError.
        """
        raise NotImplementedError("TODO(Project 27): implement ContextAssembler.reorder")

    def truncate(self, documents: list[Document], token_budget: int) -> list[Document]:
        """TODO(Project 27): greedily keep chunks up to ``token_budget`` tokens.

        Walk the list in order, counting tokens per chunk (use
        ``self.token_counter`` when set, else ``len(text) // 4``), and keep
        chunks while the running total stays under the budget. A chunk that
        would exceed the budget on its own is skipped. Return the kept list.
        """
        raise NotImplementedError("TODO(Project 27): implement ContextAssembler.truncate")


class LostInTheMiddleExperiment:
    """Measure answer-position bias: same answer, three context positions.

    Reproduces the Liu et al. 2023 finding that LLMs use the start and end
    of a long context better than the middle. For each question, builds three
    contexts — answer chunk at start / middle / end — with the same distractor
    chunks, asks the LLM, and records whether the answer was correct.
    """

    def __init__(self, llm, n_distractors: int = 7):
        # llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.gemini.GeminiLLM or a local Ollama wrapper).
        self.llm = llm
        self.n_distractors = n_distractors

    def _ask(self, context: list[Document], question: str) -> str:
        """TODO(Project 27): ask the LLM for an answer given ``context``.

        Build a prompt with the chunk texts (numbered) followed by the
        question, and return ``self.llm.invoke(prompt)`` stripped. Keep the
        instruction minimal: "Answer using only the context."
        """
        raise NotImplementedError("TODO(Project 27): implement LostInTheMiddleExperiment._ask")

    def _correct(self, answer: str, expected: str) -> bool:
        """TODO(Project 27): judge whether ``answer`` contains ``expected``.

        A simple, honest check: True when the expected answer text (or a
        normalized version of it) appears in the answer, case-insensitive.
        """
        raise NotImplementedError("TODO(Project 27): implement LostInTheMiddleExperiment._correct")

    def run(
        self,
        answer_chunk: Document,
        distractors: list[Document],
        question: str,
        expected: str,
    ) -> dict[str, bool]:
        """Return {"start": bool, "middle": bool, "end": bool} correctness.

        Builds the three contexts by splicing ``answer_chunk`` at each
        position among ``distractors[:self.n_distractors]`` (pad with extra
        distractors if short), asks the LLM, and checks each answer.
        """
        pool = distractors[: self.n_distractors]
        results: dict[str, bool] = {}
        for position in ("start", "middle", "end"):
            chunks = list(pool)
            if position == "start":
                context = [answer_chunk] + chunks
            elif position == "end":
                context = chunks + [answer_chunk]
            else:  # middle: split the pool in half, insert the answer between
                mid = len(chunks) // 2
                context = chunks[:mid] + [answer_chunk] + chunks[mid:]
            answer = self._ask(context, question)
            results[position] = self._correct(answer, expected)
        return results


if __name__ == "__main__":
    # No-network smoke tests with fakes — the Project-27 stubs are filled in by
    # tiny subclasses so the run()/dedupe()/truncate() scaffolds can be tested.
    class _FakeLLM:
        """'Answers' by returning the first chunk's text."""

        def __init__(self, answer_text: str):
            self.answer_text = answer_text

        def invoke(self, prompt: str) -> str:
            return self.answer_text

    class _Asker(LostInTheMiddleExperiment):
        """Minimal _ask/_correct so the three-position harness can run."""

        def _ask(self, context: list[Document], question: str) -> str:
            return next(d.page_content for d in context if "capital" in d.page_content)

        def _correct(self, answer: str, expected: str) -> bool:
            return expected.lower() in answer.lower()

    class _Assembler(ContextAssembler):
        """Minimal exact-dup dedupe + greedy truncate so the scaffold can run."""

        def dedupe(self, documents: list[Document]) -> list[Document]:
            seen: set[str] = set()
            out: list[Document] = []
            for doc in documents:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    out.append(doc)
            return out

        def truncate(self, documents: list[Document], token_budget: int) -> list[Document]:
            kept: list[Document] = []
            total = 0
            for doc in documents:
                tokens = len(doc.page_content) // 4
                if total + tokens <= token_budget:
                    kept.append(doc)
                    total += tokens
            return kept

    answer = Document(page_content="The capital of France is Paris.")
    distractors = [
        Document(page_content=f"Unrelated filler paragraph number {i} about nothing in particular.")
        for i in range(7)
    ]

    exp = _Asker(_FakeLLM("The capital of France is Paris."), n_distractors=7)
    out = exp.run(answer, distractors, "What is the capital of France?", "Paris")
    assert set(out) == {"start", "middle", "end"}
    assert all(out.values()), out  # the fake LLM always 'knows' — the bias shows with real LLMs

    asm = _Assembler()  # no embedder: exact-dup only
    dups = [
        Document(page_content="same text"),
        Document(page_content="same text"),
        Document(page_content="different"),
    ]
    assert len(asm.dedupe(dups)) == 2
    budget = asm.truncate(dups, token_budget=2)  # 2 tokens ("same text") fits, nothing else does
    assert len(budget) == 1

    print("OK: assembler dedupes/truncates and the lost-in-the-middle harness runs")
