# Project 29 — Tool-Calling RAG

> **Goal:** Stop always-retrieving. Give the LLM a `search_documents` tool and
> let it decide — per turn, per question — whether, what, and how many times to
> query the index.

## Why

Every pipeline so far retrieves unconditionally: one question in, top-k chunks
out. But real usage is full of questions that need *no* retrieval ("What does
RAG stand for?" — answer from memory), questions that need *two* retrievals
("compare the chunk sizes in project 4 and project 9"), and follow-ups that
should reuse context instead of re-querying. Tool calling turns retrieval from
a hardcoded step into a decision the model makes: you expose `search_documents`
and `read_document` as tools, describe them in a schema, and let the LLM emit a
tool call when it needs evidence. You also see exactly what it chose to call —
the tool-call trace — which is the first step toward agentic RAG (Projects
30–31).

## Learn

- Function/tool calling: the LLM outputs a structured tool invocation instead of plain text
- Tool schema design: name, description, and parameters — the description is what the model "reads"
- When retrieval becomes a decision: no-retrieval, single-retrieval, multi-retrieval turns
- Tool-call traces: inspecting which tools the model chose, in what order
- Ollama tool support (`qwen2.5-coder:7b` or a model with tool calling); Gemini function-calling as the cloud alternative

## Execute

1. **Setup** — `pip install langchain-classic` and have Ollama running (or `GOOGLE_API_KEY` in `.env`)
2. **Read** — `src/agent/tool_calling.py` — the `ToolCallingAgent` stub and its two tool schemas
3. **Implement** — `search_documents(query)` (top-k chunks from a wrapped retriever) and `read_document(doc_id)` (full document text); wire the agent loop and capture `tool_calls`
4. **Run** — `python agent/tool_calling.py` for the smoke test; then ask the three question types
5. **Measure** — the tool-call trace per question: how many retrievals, which queries, did it retrieve when it shouldn't have
6. **Acceptance criteria** — the no-retrieval question triggers zero tool calls; the compare question triggers two; the answer to each question cites only chunks the agent actually retrieved

## Stretch

- Add a `list_documents` tool and let the agent browse before searching
- Add conversation history to the prompt and verify a follow-up skips redundant retrieval
- Add a hard budget: max 3 tool calls per answer, enforced in the loop

## Article

- [ ] `29-tool-calling-rag.md`

## Code

- `src/agent/tool_calling.py` — `ToolCallingAgent(llm, retriever)` — tool-calling loop; `search_documents` + `read_document` tools

## Notebook

`NoteBooks/Projects/Project-29-Tool-Calling-RAG/01-tool-calling-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Projects/Project-29-Tool-Calling-RAG/01-tool-calling-spec.py NoteBooks/Projects/Project-29-Tool-Calling-RAG/01-tool-calling.ipynb
```
