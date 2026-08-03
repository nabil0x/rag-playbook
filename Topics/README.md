# Topics — Content Map (Basic → Advanced RAG)

Publishable content organized along the README's **17-project progression + capstone**.
Each project folder contains a card (README) with the article checklist, the component
modules it exercises, and the matching notebook. Tick boxes as articles are published.

| # | Project | Core idea | Notebook | Status |
|---|---------|-----------|----------|--------|
| 01 | [Baseline RAG](Project-01-Baseline-RAG/README.md) | Full pipeline, simplest components | [4 notebooks](../NoteBooks/Project-01-Baseline-RAG/) | 🔲 Draft |
| 02 | [PDF Knowledge Base](Project-02-PDF-Knowledge-Base/README.md) | PDF parsing | [01-pdf-knowledge-base.ipynb](../NoteBooks/Project-02-PDF-Knowledge-Base/01-pdf-knowledge-base.ipynb) | 🔲 Draft |
| 03 | [Local Documentation Search](Project-03-Local-Documentation-Search/README.md) | Offline embeddings + vector search | [01-local-documentation-search.ipynb](../NoteBooks/Project-03-Local-Documentation-Search/01-local-documentation-search.ipynb) | 🔲 Draft |
| 04 | [Markdown Documentation RAG](Project-04-Markdown-Documentation-RAG/README.md) | Structure-preserving splitting | [01-markdown-documentation-rag.ipynb](../NoteBooks/Project-04-Markdown-Documentation-RAG/01-markdown-documentation-rag.ipynb) | 🔲 Draft |
| 05 | [HTML Documentation](Project-05-HTML-Documentation/README.md) | HTML hierarchy | [01-html-documentation.ipynb](../NoteBooks/Project-05-HTML-Documentation/01-html-documentation.ipynb) | 🔲 Draft |
| 06 | [Better Embeddings](Project-06-Better-Embeddings/README.md) | Embedding model swap + eval | [01-comparing-embedding-models.ipynb](../NoteBooks/Project-06-Better-Embeddings/01-comparing-embedding-models.ipynb) | 🔲 Draft |
| 07 | [Compare Vector Databases](Project-07-Compare-Vector-Databases/README.md) | Store swap + measurements | [01-comparing-vector-databases.ipynb](../NoteBooks/Project-07-Compare-Vector-Databases/01-comparing-vector-databases.ipynb) | 🔲 Draft |
| 08 | [MMR Retrieval](Project-08-MMR-Retrieval/README.md) | Diversity vs relevance | [01-mmr-retrieval.ipynb](../NoteBooks/Project-08-MMR-Retrieval/01-mmr-retrieval.ipynb) | 🔲 Draft |
| 09 | [Parent-Child Retrieval](Project-09-Parent-Child-Retrieval/README.md) | Small chunks, large context | [01-parent-child-retrieval.ipynb](../NoteBooks/Project-09-Parent-Child-Retrieval/01-parent-child-retrieval.ipynb) | 🔲 Draft |
| 10 | [MultiQuery Retrieval](Project-10-MultiQuery-Retrieval/README.md) | Query expansion | [01-multiquery-retrieval.ipynb](../NoteBooks/Project-10-MultiQuery-Retrieval/01-multiquery-retrieval.ipynb) | 🔲 Draft |
| 11 | [Context Compression](Project-11-Context-Compression/README.md) | Token reduction | [01-context-compression.ipynb](../NoteBooks/Project-11-Context-Compression/01-context-compression.ipynb) | 🔲 Draft |
| 12 | [Prompt Engineering](Project-12-Prompt-Engineering/README.md) | Basic → JSON → few-shot → citation → reasoning | [01-prompt-engineering.ipynb](../NoteBooks/Project-12-Prompt-Engineering/01-prompt-engineering.ipynb) | 🔲 Draft |
| 13 | [Multi-format RAG](Project-13-Multi-format-RAG/README.md) | PDF + Markdown + CSV + Web + JSON | [01-multi-format-rag.ipynb](../NoteBooks/Project-13-Multi-format-RAG/01-multi-format-rag.ipynb) | 🔲 Draft |
| 14 | [Metadata Filtering](Project-14-Metadata-Filtering/README.md) | Scoped retrieval | [01-metadata-filtering.ipynb](../NoteBooks/Project-14-Metadata-Filtering/01-metadata-filtering.ipynb) | 🔲 Draft |
| 15 | [Hybrid Search](Project-15-Hybrid-Search/README.md) | BM25 + dense fusion | [01-hybrid-search.ipynb](../NoteBooks/Project-15-Hybrid-Search/01-hybrid-search.ipynb) | 🔲 Draft |
| 16 | [Build Without LangChain](Project-16-Build-Without-LangChain/README.md) | Pure Python pipeline | [01-rag-without-langchain.ipynb](../NoteBooks/Project-16-Build-Without-LangChain/01-rag-without-langchain.ipynb) | 🔲 Draft |
| 17 | [Modular RAG Framework](Project-17-Modular-RAG-Framework/README.md) | Your own pluggable architecture | [01-modular-rag-framework.ipynb](../NoteBooks/Project-17-Modular-RAG-Framework/01-modular-rag-framework.ipynb) | 🔲 Draft |
| 18 | [RAG Benchmark Suite](Project-18-RAG-Benchmark-Suite/README.md) | Capstone: design, compare, evaluate | — (capstone) | 🔲 Draft |

## How content maps to code

- Component modules (repo root): `loaders/`, `splitters/`, `embeddings/`, `vectordb/`,
  `retrieval/`, `prompts/`, `llms/` — one class per component, named in each project card.
- Hands-on notebooks: `NoteBooks/Project-NN-*` (see each project card).
- Sample data: `Data/` (e.g. `Waiting.txt`, a public-domain Project Gutenberg book).

## Publishing workflow

1. Write the article under the project folder (one `.md` per project).
2. Fill in the matching component module in the repo root.
3. Add/run the matching notebook in `NoteBooks/Project-NN-*`.
4. Tick the checkbox in the project card and update this table.
