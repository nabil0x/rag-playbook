"""Tool-calling RAG agent.

Agent block: instead of always retrieving, expose ``search_documents`` and
``read_document`` as tools and let the LLM decide — per turn — whether, what,
and how many times to query the index. The tool-call trace shows exactly what
the model chose to do. See Topics/Project-29-Tool-Calling-RAG/README.md.
"""

from __future__ import annotations

from langchain_core.documents import Document


class ToolCallingAgent:
    """Agent that retrieves via tool calls the LLM chooses itself.

    Wraps an LLM with tool-calling support (Ollama ``qwen2.5-coder:7b`` or
    Gemini) plus a retriever, and exposes two tools: ``search_documents``
    (top-k chunks) and ``read_document`` (full text by id). The tools are
    plain Python callables — the agent library binds them by schema.
    """

    #: Docstrings become the tool descriptions the model "reads".
    SEARCH_DESCRIPTION = (
        "Search the document store and return the top-k most relevant chunk texts "
        "for the query. Use this when you need evidence from the documents."
    )
    READ_DESCRIPTION = (
        "Return the full text of one document by its id. Use this after "
        "search_documents when a chunk alone is not enough context."
    )

    def __init__(self, llm, retriever, top_k: int = 5, max_tool_calls: int = 3):
        # llm: any object exposing bind_tools(tools) and an agent-ready
        # interface (e.g. a ChatOllama with tool calling, or Gemini).
        # retriever: any object exposing retrieve(question) -> list[Document].
        # Imported by the caller, never at module level.
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.max_tool_calls = max_tool_calls

    def search_documents(self, query: str) -> str:
        """Tool: return the top-k chunk texts for ``query`` as one string."""
        docs = self.retriever.retrieve(query)[: self.top_k]
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def read_document(self, doc_id: str) -> str:
        """Tool: return the full text of one document by ``doc_id``.

        TODO(Project 29): resolve ``doc_id`` against the store's metadata.
        The stub walks the retriever's candidates for a ``metadata["id"]``
        match and returns the page_content; a real implementation fetches
        from the source store.
        """
        raise NotImplementedError("TODO(Project 29): implement ToolCallingAgent.read_document")

    def _tools(self) -> list[dict]:
        """Tool schemas in the OpenAI function format the agent loop expects."""
        return [
            {
                "name": "search_documents",
                "description": self.SEARCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query."}},
                    "required": ["query"],
                },
            },
            {
                "name": "read_document",
                "description": self.READ_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {"doc_id": {"type": "string", "description": "The document id."}},
                    "required": ["doc_id"],
                },
            },
        ]

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        """TODO(Project 29): run the tool-calling loop and return the result.

        Expected shape of the return dict:
            {"answer": str, "tool_calls": [tool_name, ...]}

        Implement with the agent tooling available in your environment:
        - ``langchain_classic``: ``create_tool_calling_agent`` +
          ``AgentExecutor`` with ``self.llm.bind_tools(self._tools())``;
          extract the intermediate steps into ``tool_calls``.
        - or a manual loop: prompt the LLM, detect a tool invocation, call
          the bound tool, feed the observation back — up to
          ``self.max_tool_calls`` iterations, then ask for the final answer.

        ``history`` (optional list of {"role", "content"}) is prepended so
        follow-ups can reference the previous turn.
        """
        raise NotImplementedError("TODO(Project 29): implement ToolCallingAgent.answer")


if __name__ == "__main__":
    # No-network smoke test with a fake retriever — no agent loop is run.
    class _FakeRetriever:
        def retrieve(self, question: str) -> list[Document]:
            return [Document(page_content=f"chunk about {question}", metadata={"id": "doc-1"})]

    agent = ToolCallingAgent(llm=None, retriever=_FakeRetriever(), top_k=3, max_tool_calls=3)

    search_out = agent.search_documents("chunking")
    assert "chunk about chunking" in search_out

    # Tool schemas must be in the OpenAI function format.
    schemas = agent._tools()
    assert len(schemas) == 2
    assert schemas[0]["name"] == "search_documents" and "parameters" in schemas[0]
    assert schemas[1]["name"] == "read_document"

    print("OK: tools are wired and schemas are well-formed")
