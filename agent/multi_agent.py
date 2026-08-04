"""Multi-agent RAG.

Agent block: split the RAG job across specialist agents — a supervisor routes
the question, a planner decomposes it into search intents, retriever agents
search in parallel, and a writer synthesizes the evidence. Handoffs happen
through shared state, not chat text. See Topics/Project-31-Multi-Agent-RAG/README.md.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document


class SearchPlannerAgent:
    """Turn a question into concrete, independent search intents.

    Each intent is a string a retriever can run directly. Multi-hop questions
    produce 2-3 intents; simple questions produce one.
    """

    def __init__(self, llm):
        # llm: any object exposing invoke(prompt_text) -> str.
        self.llm = llm

    def plan(self, question: str) -> list[str]:
        """TODO(Project 31): decompose ``question`` into search intents.

        Prompt the LLM for 1-3 independent search queries (one per line),
        parse them, and fall back to ``[question]`` when nothing usable
        comes back. Simple questions naturally yield one intent.
        """
        raise NotImplementedError("TODO(Project 31): implement SearchPlannerAgent.plan")


class RetrieverAgent:
    """Execute one search intent against a retriever.

    Intentionally narrow: one intent in, chunks out. The narrowness is the
    point — a retriever agent never plans or writes, so its context stays
    clean and it can run in parallel with its siblings.
    """

    def __init__(self, retriever, top_k: int = 5):
        # retriever: any object exposing retrieve(question) -> list[Document].
        self.retriever = retriever
        self.top_k = top_k

    def search(self, intent: str) -> list[Document]:
        """TODO(Project 31): retrieve ``top_k`` chunks for ``intent``.

        Call ``self.retriever.retrieve(intent)[: self.top_k]`` and tag each
        returned chunk's metadata with ``{"intent": intent}`` so the writer
        knows which search produced which evidence.
        """
        raise NotImplementedError("TODO(Project 31): implement RetrieverAgent.search")


class WriterAgent:
    """Synthesize the final answer from the collected evidence chunks.

    Only ever sees the evidence — never the planning noise — so the answer
    is grounded in exactly the chunks the retriever agents returned.
    """

    def __init__(self, llm):
        # llm: any object exposing invoke(prompt_text) -> str.
        self.llm = llm

    def write(self, question: str, evidence: list[Document]) -> str:
        """TODO(Project 31): produce the final answer from ``evidence``.

        Build a prompt with the numbered chunk texts + the question, invoke
        the LLM, and return the stripped answer. Instruction should forbid
        inventing facts beyond the evidence.
        """
        raise NotImplementedError("TODO(Project 31): implement WriterAgent.write")


class SupervisorAgent:
    """Route a question to a path, then orchestrate the specialist agents.

    The router decides whether the question is simple (direct retrieval),
    multi-hop (plan + parallel retrieval), or needs summarization — and the
    orchestrator runs the chosen path, collecting evidence for the writer.
    """

    def __init__(self, router_llm, planner: SearchPlannerAgent, retriever_agent: RetrieverAgent, writer: WriterAgent):
        self.router_llm = router_llm
        self.planner = planner
        self.retriever_agent = retriever_agent
        self.writer = writer

    def _route(self, question: str) -> str:
        """TODO(Project 31): classify the question.

        Prompt the router LLM to answer with exactly one of
        "direct" | "multi-hop" | "summary". Fall back to "direct" when the
        response is not one of the three. Multi-hop questions should route
        to "multi-hop" — that is the case the parallel path exists for.
        """
        raise NotImplementedError("TODO(Project 31): implement SupervisorAgent._route")

    def _run_multi_hop(self, question: str) -> list[Document]:
        """Plan intents and search them in parallel.

        Uses ``ThreadPoolExecutor`` so independent intents run concurrently
        — the parallelism the README asks you to measure. Order of the
        merged evidence is preserved (one list per intent, concatenated).
        """
        intents = self.planner.plan(question)
        with ThreadPoolExecutor(max_workers=min(len(intents), 4)) as pool:
            results = list(pool.map(self.retriever_agent.search, intents))
        merged: list[Document] = []
        for chunks in results:
            merged.extend(chunks)
        return merged

    def run(self, question: str) -> dict:
        """TODO(Project 31): route, gather evidence, and write the answer.

        Return a dict:
            {"answer": str, "route": str, "intents": list[str], "evidence_count": int}

        - "direct": retrieve once, write from those chunks.
        - "multi-hop": ``_run_multi_hop`` then write.
        - "summary": same as multi-hop for now (the Stretch adds a dedicated
          summarizer path); route must still be reported accurately.
        """
        raise NotImplementedError("TODO(Project 31): implement SupervisorAgent.run")


if __name__ == "__main__":
    # No-network smoke test — the Project-31 stubs are filled in by tiny
    # subclasses so the parallel _run_multi_hop scaffold can be tested.
    class _FakeLLM:
        def __init__(self, text: str):
            self.text = text

        def invoke(self, prompt: str) -> str:
            return self.text

    class _FakeRetriever:
        def retrieve(self, question: str) -> list[Document]:
            return [Document(page_content=f"evidence about {question}")]

    class _Planner(SearchPlannerAgent):
        """Minimal plan() so the parallel orchestration scaffold can run."""

        def plan(self, question: str) -> list[str]:
            return ["Who wrote Waiting?", "Where was the author born?"]

    class _Retriever(RetrieverAgent):
        """Minimal search() tagging intent in metadata, as the stub promises."""

        def search(self, intent: str) -> list[Document]:
            docs = self.retriever.retrieve(intent)[: self.top_k]
            for doc in docs:
                doc.metadata["intent"] = intent
            return docs

    planner = _Planner(_FakeLLM("Who wrote Waiting?\nWhere was the author born?"))
    retriever_agent = _Retriever(_FakeRetriever(), top_k=2)
    writer = WriterAgent(_FakeLLM("synthesized answer"))
    supervisor = SupervisorAgent(
        router_llm=_FakeLLM("multi-hop"),
        planner=planner,
        retriever_agent=retriever_agent,
        writer=writer,
    )

    # _run_multi_hop executes both intents in parallel and merges evidence.
    evidence = supervisor._run_multi_hop("Who wrote Waiting, and where born?")
    assert len(evidence) == 2

    print("OK: supervisor, planner, retriever agent, and writer are wired")
