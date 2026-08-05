# Custom RAG tools

Where LangChain and its associated frameworks do not support an operation, the
needed class is built here — shared, reusable, and implementing the same
contract style as the component library (`src/loaders/`, `src/retrieval/`, …).

## Inventory

| File | What it does | Used by |
|---|---|---|
| `reranker.py` | Cross-encoder re-ranking via sentence-transformers `CrossEncoder` (LangChain has no native cross-encoder reranker) | 06-re-ranking |
| `prf.py` | Pseudo-relevance feedback — expand the query with terms from top-k retrieved docs | 05-query-transformation |
| `graph.py` | Entity/relation graph construction (networkx) from documents | 08-graphrag |
| `graphrag.py` | GraphRAG query flow: global vs local search over the graph | 08-graphrag |
| `raptor.py` | RAPTOR: embedding clustering + recursive summarization tree build + tree traversal | 09-raptor |
| `verifier.py` | Evidence verification: retrieve candidate evidence + LLM claim verdict (SUPPORTED / REFUTED / NOT_ENOUGH_INFO) | 10-agentic-rag |

Rule of thumb: if LangChain already ships it (or a class in this repo does),
reuse it — do not reimplement. `src/tools/` is only for the gaps.
