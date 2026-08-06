# Layer-1 RAG Playbook — curriculum

Python-first labs for the 10 core RAG concepts. Each track is a folder of
standalone `.py` labs; every lab must run and pass its verification gate
before it is converted to a notebook under
`NoteBooks/Curriculum/Curriculum-<NN>-<concept>/` (see `AGENTS.md` for the workflow).

## Tracks

| Track | Data | Reuses | Custom tools |
|---|---|---|---|
| 01-chunking | `Data/corpus/gutenberg/` (public domain), `Data/local-docs/`, SD samples | `src/splitters/` | — |
| 02-embeddings | rag-mini-wikipedia | `src/embeddings/{bge,e5}.py` | — |
| 03-vector-databases | rag-mini-wikipedia | `src/vectordb/{faiss,chroma,qdrant}.py` | — |
| 04-retrieval | rag-mini + beir qrels | `src/retrieval/`, `src/evaluation/` | — |
| 05-query-transformation | rag-mini test, hotpotqa | `src/retrieval/{query_rewrite,hyde,decompose,step_back}.py` | `src/tools/prf.py` |
| 06-re-ranking | beir qrels, lost-in-the-middle | `src/retrieval/{rerank,rerank_advanced}.py` | `src/tools/reranker.py` |
| 07-evaluation | qrels + gold answers | `src/evaluation/` | — |
| 08-graphrag | hotpotqa, rag-mini | `src/llms/ollama.py` | `src/tools/{graph,graphrag}.py` |
| 09-raptor | rag-mini-wikipedia | `src/llms/ollama.py`, `src/embeddings/bge.py` | `src/tools/raptor.py` |
| 10-agentic-rag | hotpotqa, scifact | `src/llms/ollama.py`, `src/embeddings/bge.py`, LangGraph | `src/tools/verifier.py` |

Full plan: `.omo/plans/layer1-rag-playbook.md`.

## Scratch notebooks — LangChain-native mirrors of every lab

Alongside the component-reusing notebooks under `NoteBooks/Curriculum/Curriculum-*`, each
track also ships a **Scratch** series — `NoteBooks/Scratch/Scratch-<NN>-<concept>/`,
one self-contained notebook per lab. A Scratch notebook is the *same
experiment* as its lab (same constants, dataset paths, demo output, and
`[PASS]` verification gate) but implemented **inline**: it imports only
LangChain and its ecosystem (sentence-transformers, faiss, Chroma, Qdrant,
Groq, Ollama) plus raw libraries, and never imports from `src/`. Anything the
lab gets from a shared component is rebuilt in the notebook as plain code
behind the same contract.

| What it shows | Detail |
|---|---|
| One notebook per lab | 47 notebooks across all 10 tracks (01: 6, 02: 4, 03: 4, 04: 6, 05: 6, 06: 5, 07: 5, 08: 4, 09: 3, 10: 4) |
| Self-contained | Local embeddings (BGE/E5), in-memory FAISS, inline prompts, hand-rolled metrics — no `src/` imports |
| Fidelity | Same constants, corpus paths, measured quantities, and verification gates as the source lab |
| Verification | Each notebook passes a static no-`src`-import audit and an execution gate (`[PASS]` checks, identical to the lab's CI-style `--verify` run) |
| Local-first | Embeddings stay local (AGENTS.md #1); LLMs are Groq or local Ollama depending on the lab |

The Scratch series exists to prove the labs translate to a dependency-minimal,
framework-native implementation — the same decisions built from raw LangChain
blocks rather than the repo's shared components.

