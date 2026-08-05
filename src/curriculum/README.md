# Layer-1 RAG Playbook — curriculum

Python-first labs for the 10 core RAG concepts. Each track is a folder of
standalone `.py` labs; every lab must run and pass its verification gate
before it is converted to a notebook under
`NoteBooks/Curriculum-<NN>-<concept>/` (see `AGENTS.md` for the workflow).

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
