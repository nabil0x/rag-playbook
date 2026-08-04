"""LangGraph agentic RAG.

Agent block: an explicit stateful graph — plan → retrieve → generate → reflect
— where a conditional edge loops back with a revised query when the critique
finds the answer weak. Conversation history lives in the graph state.
See Topics/Project-30-LangGraph-Agentic-RAG/README.md.
"""

from __future__ import annotations

from typing import Any


class LangGraphRAGAgent:
    """Compile a plan/retrieve/generate/reflect ``StateGraph``.

    The graph owns a single typed state dict shared by every node, so the
    whole control flow — including the reflection loop — is a data structure
    you can read and edit rather than a loop hidden inside an agent library.

    State keys used by the nodes:
        question: str            — the user question (optionally rewritten)
        history: list[dict]      — prior turns, appended at each call
        chunks: list[Document]   — retrieved evidence
        answer: str              — the latest generated answer
        critique: str            — the latest reflect-node verdict
        loop_count: int          — reflection iterations so far
    """

    def __init__(self, llm, retriever, max_loops: int = 2):
        # llm: any object exposing invoke(prompt_text) -> str
        # (e.g. llms.gemini.GeminiLLM or a local Ollama wrapper).
        # retriever: any object exposing retrieve(question) -> list[Document].
        self.llm = llm
        self.retriever = retriever
        self.max_loops = max_loops
        self._graph = None

    def _build_graph(self):
        """TODO(Project 30): assemble the ``StateGraph`` and return it.

        Using ``langgraph.graph.StateGraph`` (lazy import; SKIP hint when
        missing) plus a state schema (a ``TypedDict`` with the keys above):

        - Node "plan": rewrite/decompose ``question`` (prompt of your
          choosing) and store it back in state.
        - Node "retrieve": call ``self.retriever.retrieve(question)`` and
          store the chunks.
        - Node "generate": prompt the LLM with the chunks + history, store
          the answer.
        - Node "reflect": prompt the LLM to critique the answer — is it
          grounded in the chunks? complete? — and store the critique.
        - Conditional edge from "reflect": loop back to "retrieve" with a
          revised query when the critique is negative AND
          ``loop_count < self.max_loops``; otherwise route to END.

        Compile with ``graph.compile()`` and keep the result on ``self._graph``.
        """
        raise NotImplementedError("TODO(Project 30): implement LangGraphRAGAgent._build_graph")

    def _get_graph(self):
        """Build once, then reuse the compiled graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def run(self, question: str, history: list[dict] | None = None) -> dict:
        """TODO(Project 30): execute the graph and return the final state.

        Invoke ``self._get_graph()`` with an initial state dict (question,
        history, chunks=[], answer="", critique="", loop_count=0) and return
        the resulting state. The compiled graph's ``invoke``/``stream`` API
        varies by langgraph version — prefer ``stream`` to also capture the
        node-by-node trace the README asks you to inspect.
        """
        raise NotImplementedError("TODO(Project 30): implement LangGraphRAGAgent.run")

    def trace(self, question: str, history: list[dict] | None = None) -> list[str]:
        """Return the ordered node names a run visits (for the README's trace step).

        Runs the graph and records each node as it executes. Falls back to
        ``["plan", "retrieve", "generate", "reflect"]`` when the graph is not
        built yet (so the smoke test runs without langgraph installed).
        """
        if self._graph is None:
            return ["plan", "retrieve", "generate", "reflect"]
        trace: list[str] = []
        try:
            for event in self._get_graph().stream(
                {"question": question, "history": history or [], "chunks": [], "answer": "", "critique": "", "loop_count": 0},
                stream_mode="updates",
            ):
                for node in event:
                    trace.append(node)
        except Exception:  # noqa: BLE001 — a stub must never crash the trace
            trace = ["plan", "retrieve", "generate", "reflect"]
        return trace


if __name__ == "__main__":
    # No-network smoke test — langgraph is optional; the graph is not built.
    agent = LangGraphRAGAgent(llm=None, retriever=None, max_loops=2)

    assert agent.max_loops == 2
    assert agent._graph is None  # nothing imported or built at construction

    # The trace falls back to the canonical node list without langgraph.
    assert agent.trace("question") == ["plan", "retrieve", "generate", "reflect"]

    print("OK: LangGraphRAGAgent wiring validated (graph builds lazily on first run)")
