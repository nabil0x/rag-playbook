# Project 31 — Multi-Agent RAG

> **Goal:** Split the RAG job across specialist agents — a supervisor routes the
> question, a planner decomposes it, retriever agents search in parallel, and a
> writer synthesizes the evidence — and see when one brain is not enough.

## Why

A single agent does everything: it plans, retrieves, and writes in one context.
That coupling fails at scale — a long multi-hop question overwhelms the plan,
one agent's retrieval choices are another agent's hallucinated context, and
everything is serialized. Multi-agent RAG assigns each skill its own agent with
its own prompt and context: the supervisor decides the route (direct /
multi-hop / summary), the planner turns the route into concrete search intents,
retriever agents execute those intents *in parallel* (they are embarrassingly
parallel), and the writer only ever sees the collected evidence — never the
planning noise. You also get the failure isolation for free: when the answer is
wrong, you know which specialist produced the bad step.

## Learn

- The supervisor pattern: a router agent that classifies the question before any retrieval
- Agent specialization: each agent gets a narrow prompt and a narrow context, not the whole conversation
- Parallel execution: retriever agents running concurrently (`ThreadPoolExecutor`) on independent search intents
- Handoff mechanics: passing evidence between agents through shared state, not through chat text
- When multi-agent wins vs when it is overhead: multi-hop and mixed questions vs simple lookups

## Execute

1. **Setup** — `pip install langchain-classic` (for agent utilities) and Ollama or Gemini
2. **Read** — `agent/multi_agent.py` — the `SupervisorAgent`, `SearchPlannerAgent`, `RetrieverAgent`, `WriterAgent` stubs
3. **Implement** — the router decision, the intent decomposition, the parallel retrieval fan-out, and the writer synthesis
4. **Run** — `python agent/multi_agent.py` for the smoke test; then a multi-hop question through `run(question)`
5. **Measure** — the route chosen, the intents planned, wall-clock time with parallel vs serial retrieval, and the writer's evidence list
6. **Acceptance criteria** — a multi-hop question routes to the multi-hop path, produces 2+ intents, and the final answer cites chunks from *both* retrievals; parallel fan-out is measurably faster than serial on 3 intents

## Stretch

- Add a third specialist (e.g. `SummarizerAgent` for long documents) and route to it
- Log each agent's prompt and output to a trace file for debugging
- Compare one big-agent vs supervisor+specialists on the same 5 questions — quality and cost

## Article

- [ ] `31-multi-agent-rag.md`

## Code

- `agent/multi_agent.py` — `SupervisorAgent`, `SearchPlannerAgent`, `RetrieverAgent`, `WriterAgent` — routed, parallel multi-agent RAG

## Notebook

`NoteBooks/Project-31-Multi-Agent-RAG/01-multi-agent-spec.py` → generate with:

```bash
python scripts/gen_notebook.py NoteBooks/Project-31-Multi-Agent-RAG/01-multi-agent-spec.py NoteBooks/Project-31-Multi-Agent-RAG/01-multi-agent.ipynb
```
