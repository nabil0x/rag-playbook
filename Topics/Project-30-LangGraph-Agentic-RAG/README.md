# Project 30 — LangGraph Agentic RAG

> **Goal:** Rebuild the tool-calling agent as an explicit stateful graph —
> `plan → retrieve → generate → reflect`, with a conditional loop that re-
> retrieves when the first answer is weak, and memory carried in graph state.

## Why

The tool-calling agent in Project 29 hides its control flow inside an agent
library: tools, memory, and the stop condition are all implicit. When the
answer comes back wrong, you have no knob to turn. LangGraph makes the control
flow a *data structure*: nodes (plan, retrieve, generate, reflect) connected
by edges, sharing a typed state dict. That turns "the agent loop" into
something you can read, pause, and edit. It also makes reflection practical —
after generating, an LLM critique checks "is this answer grounded in the
chunks? is it complete?" and a conditional edge either loops back with a
revised query or ends. Multi-step, self-correcting retrieval stops being magic
and becomes four named functions.

## Learn

- Graph state: one typed dict shared by all nodes — the single source of truth for the run
- Nodes and edges: `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`, `END`
- Reflection loop: generate → critique (grounded? complete?) → re-retrieve with a revised query → re-generate, capped at N iterations
- Memory in state: conversation history appended to state, so follow-ups have context
- Visualizing the graph (mermaid/ASCII) and tracing one run node-by-node

## Execute

1. **Setup** — `pip install langgraph` (plus `langchain-classic` if using agent utilities) and Ollama or Gemini
2. **Read** — `src/agent/langgraph_agent.py` — the `LangGraphRAGAgent` stub with its node skeletons and state schema
3. **Implement** — the five nodes, the conditional reflect edge, and the loop counter in state
4. **Run** — `python agent/langgraph_agent.py` for the smoke test; then compile and invoke the graph on a hard question
5. **Measure** — print the node-by-node trace (which nodes ran, in what order) and the number of reflect→re-retrieve loops before the critique passes
6. **Acceptance criteria** — a deliberately vague first question triggers at least one reflection loop with a revised query; the graph reaches `END` (no infinite loop); state carries history so a follow-up references the previous turn

## Stretch

- Add a `grade_documents` node that filters chunks by relevance before generation
- Route on the critique's confidence: end early on high confidence, loop on low
- Add a web-search tool node as an alternative evidence path when retrieval misses

## Article

- [ ] `30-langgraph-agentic-rag.md`

## Code

- `src/agent/langgraph_agent.py` — `LangGraphRAGAgent(llm, retriever, max_loops)` — compiles a plan/retrieve/generate/reflect `StateGraph`

## Notebook

`NoteBooks/Project-30-LangGraph-Agentic-RAG/01-langgraph-rag-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Project-30-LangGraph-Agentic-RAG/01-langgraph-rag-spec.py NoteBooks/Project-30-LangGraph-Agentic-RAG/01-langgraph-rag.ipynb
```
