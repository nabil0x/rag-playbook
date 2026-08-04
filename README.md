# RAG Playbook

> A hands-on Retrieval-Augmented Generation portfolio: 34 project cards covering
> Projects 01-36, 10 python-first concept tracks, runnable notebooks, reusable
> pipeline components, benchmark datasets, and special-document experiments.

I built this repository to study RAG the way it is actually engineered: not as
one demo notebook, but as a system of decisions. Every lab changes one part of
the pipeline, measures what changed, and turns the lesson into reusable code.

For recruiters, this repo is evidence that I can reason across the RAG stack:
loading, chunking, embeddings, vector search, retrieval strategy, reranking,
prompting, evaluation, agentic flows, observability, and service design.

For learners, this repo is a guided path: start with a baseline RAG app, swap
one component at a time, then study advanced retrieval and production patterns
without losing sight of the full system.

---

## What This Project Demonstrates

This is not only a collection of notebooks. It is a structured learning and
implementation system for RAG.

| Signal | What it shows |
|---|---|
| End-to-end RAG architecture | Loader -> splitter -> embedder -> vector DB -> retriever -> prompt -> LLM |
| Component-level understanding | Each block lives in its own module and can be swapped independently |
| Local-first embedding discipline | BGE/E5 sentence-transformer embeddings are the default study path |
| LangChain ecosystem fluency | Uses LangChain, LangChain Community, HuggingFace, Chroma, FAISS, Qdrant, and LangGraph-style patterns |
| Evaluation mindset | Retrieval metrics, qrels, generation metrics, attribution checks, drift, regression, and judge cross-checking |
| Advanced RAG coverage | Query rewrite, HyDE, decomposition, MMR, hybrid search, parent-child retrieval, compression, reranking, GraphRAG, RAPTOR, agentic RAG |
| Production awareness | API service, async ingestion, observability, online evaluation, regression gates, and Docker-oriented structure |
| Teaching ability | Topics, notebooks, and curriculum tracks explain concepts step by step |

---

## The Core Idea

Most RAG tutorials hide the real engineering question:

> Which component should change, and how do I know it improved the system?

This repo answers that by treating RAG as a replaceable pipeline.

```text
Documents
  -> Loaders
  -> Splitters
  -> Embeddings
  -> Vector DB
  -> Retriever
  -> Prompt
  -> LLM
  -> Answer
```

Each folder maps to one part of that pipeline:

| Pipeline block | Folder | Examples |
|---|---|---|
| Load documents | `loaders/` | Web, PDF, CSV, Word, Gutenberg |
| Split text | `splitters/` | Recursive, semantic, token-aware |
| Embed chunks | `embeddings/` | Local BGE, local E5, Gemini |
| Store vectors | `vectordb/` | Chroma, FAISS, Qdrant, FAISS index internals |
| Retrieve context | `retrieval/` | Similarity, MMR, hybrid, rewrite, HyDE, decomposition, rerank, context assembly |
| Build prompts | `prompts/` | Basic prompt, citation prompt |
| Generate answers | `llms/` | Gemini, OpenAI, Groq |
| Measure behavior | `evaluation/` | Metrics, golden tests, judge, attribution, drift, regression, online eval |

The endgame is [`main.py`](main.py), where these pieces become a pluggable
`RAGPipeline`.

```python
pipeline = RAGPipeline(
    loader=CSVLoader("Data/sample.csv"),
    splitter=DocumentProcessor(),
    embedder=GeminiEmbedding(),
    db=ChromaVectorStore(embedding=embedder),
    retriever=SimilarityRetriever(db),
    llm=GeminiLLM(),
)

pipeline.ingest()
answer = pipeline.ask("What is the sample about?")
```

---

## Recruiter View: Skills Proven Here

If you are evaluating this repo quickly, these are the strongest signals.

| Skill area | Evidence in the repo |
|---|---|
| RAG systems | 34 project cards in [`Topics/`](Topics/) covering Projects 01-36, plus runnable notebooks in [`NoteBooks/`](NoteBooks/) |
| Retrieval engineering | Similarity, MMR, hybrid, parent-child, query transformation, context compression, reranking |
| Embedding and vector search | Local BGE/E5, Gemini embeddings, FAISS, Chroma, Qdrant, index tradeoffs |
| Evaluation | BEIR qrels, rag-mini-wikipedia, HotpotQA, SciFact, generation metrics, attribution, drift, regression |
| Production patterns | [`api/`](api/), [`agent/`](agent/), [`observability/`](observability/), [`docker/`](docker/) |
| Learning-to-teaching | Topic cards, curriculum tracks, notebooks, and special-document studies |
| Engineering discipline | Python-first labs, verification before notebooks, shared components, local embeddings by default |

Suggested review path:

1. Start with [`main.py`](main.py) to see the final architecture.
2. Open [`curriculum/README.md`](curriculum/README.md) to see the concept map.
3. Open [`Topics/README.md`](Topics/README.md) to see the full project roadmap.
4. Inspect [`retrieval/`](retrieval/) and [`evaluation/`](evaluation/) for advanced RAG work.
5. Run one notebook from [`NoteBooks/Project-01-Baseline-RAG/`](NoteBooks/Project-01-Baseline-RAG/) or one curriculum lab from [`curriculum/`](curriculum/).

---

## Learner View: How To Use This Repo

The repo has three learning layers.

### 1. Project Track: Build A RAG System Piece By Piece

The project track starts from baseline RAG and then swaps one component at a
time.

| Range | Focus |
|---|---|
| Projects 01-05 | Baseline RAG, PDF, local docs, markdown, HTML |
| Projects 06-11 | Embeddings, vector DBs, MMR, parent-child retrieval, MultiQuery, compression |
| Projects 12-18 | Prompting, multi-format loading, metadata filtering, hybrid search, no-framework RAG, modular framework, benchmarking |
| Projects 20-22 | Deep evaluation, query rewrite, HyDE, decomposition |
| Projects 24-36 | Index internals, reranking, context assembly, citation scoring, tool-calling RAG, LangGraph, multi-agent RAG, API service, async ingestion, observability, online eval, drift and regression |

Project cards live in [`Topics/`](Topics/). Notebooks live in
[`NoteBooks/`](NoteBooks/).

### 2. Curriculum Track: Study Core Concepts Directly

The curriculum is python-first. Each lab starts as a standalone `.py` file under
[`curriculum/`](curriculum/) and is converted to a notebook only after it runs
and passes verification.

| Track | Concept |
|---|---|
| 01 | Chunking |
| 02 | Embeddings |
| 03 | Vector databases |
| 04 | Retrieval |
| 05 | Query transformation |
| 06 | Re-ranking |
| 07 | Evaluation |
| 08 | GraphRAG |
| 09 | RAPTOR |
| 10 | Agentic RAG |

See [`curriculum/README.md`](curriculum/README.md) for the detailed map.

### 3. Special Documents Track: RAG On Messy Real Formats

The special-document series studies formats that simple RAG demos usually skip:

| Series | Format |
|---|---|
| SD-01 | Word documents |
| SD-02 | PowerPoint |
| SD-03 | Excel |
| SD-04 | Email threads |
| SD-05 | Scanned/OCR documents |
| SD-06 | Tables and forms |
| SD-07 | Chat transcripts |
| SD-08 | Multilingual invoices |

The goal is to compare naive parsing against structure-preserving parsing and
measure the impact on retrieval and answer quality.

---

## RAG Concepts Covered

This repo is designed to make each concept concrete.

| Concept | What you can study here |
|---|---|
| Chunking | Fixed, recursive, token-aware, markdown-aware, HTML-aware, semantic, chunk-size sweeps |
| Embeddings | Local BGE, local E5, embedding similarity, model comparison, dimensionality visualization |
| Vector DBs | FAISS, Chroma persistence, Qdrant, vector index benchmarking |
| Retrieval | Top-k, MMR, hybrid search, parent-child retrieval, compression, self-query-style filtering |
| Query transformation | Rewrite, MultiQuery, decomposition, HyDE, step-back prompting, pseudo-relevance feedback |
| Reranking | Cross-encoder reranking and advanced reranker patterns |
| Context engineering | Deduplication, ordering, citation-aware context, lost-in-the-middle mitigation |
| Evaluation | Recall, MRR, nDCG, answer metrics, judge agreement, attribution, golden sets |
| Advanced RAG | GraphRAG, RAPTOR, tool-calling RAG, LangGraph-style agentic RAG, multi-agent RAG |
| Production RAG | FastAPI service, async ingestion, observability, online eval, drift, prompt regression |

---

## Data And Evaluation

The repo uses local sample data plus benchmark corpora under [`Data/`](Data/).

| Dataset area | Used for |
|---|---|
| `rag-mini-wikipedia` | General RAG generation and retrieval experiments |
| `beir-fiqa` and `beir-nfcorpus` | Retrieval metrics with qrels |
| `lost-in-the-middle` | Context ordering and position-bias experiments |
| `scifact` | Attribution and verification |
| `hotpotqa` | Multi-hop, GraphRAG, and agentic RAG |
| `gutenberg` | Long-document chunking experiments |
| `SD-*` samples | Structure-preserving parsing benchmarks |

Corpus provenance is tracked through manifest files in [`Data/`](Data/). See
[`Data/README.md`](Data/README.md) for dataset notes.

---

## Local-First Embeddings

By default, labs and notebooks embed locally with sentence-transformer models:

| Model family | Module |
|---|---|
| BGE | [`embeddings/bge.py`](embeddings/bge.py) |
| E5 | [`embeddings/e5.py`](embeddings/e5.py) |

API models such as Gemini, OpenAI, or Groq are used as LLM backends when a lab
needs generation. They are not the default workaround for embeddings.

This distinction matters because it makes retrieval experiments reproducible,
cheap to run, and easier to compare.

---

## Repository Map

```text
.
|-- README.md                  # portfolio and learning entrypoint
|-- AGENTS.md                  # working conventions for this repo
|-- main.py                    # pluggable RAGPipeline assembly
|-- requirements.txt           # Python dependencies
|-- curriculum/                # python-first concept labs
|-- Topics/                    # project cards and article plans
|-- NoteBooks/                 # runnable notebooks
|-- Data/                      # samples, benchmark corpora, manifests
|-- loaders/                   # document loaders
|-- splitters/                 # chunking strategies
|-- embeddings/                # local and API embedding wrappers
|-- vectordb/                  # vector store wrappers and index internals
|-- retrieval/                 # retrieval, query transformation, reranking
|-- prompts/                   # prompt templates
|-- llms/                      # LLM adapters
|-- evaluation/                # metrics, judge, drift, regression
|-- tools/                     # custom tools where LangChain has gaps
|-- agent/                     # tool-calling, LangGraph-style, multi-agent RAG
|-- api/                       # service layer experiments
|-- observability/             # tracing and metrics
|-- docker/                    # deployment-oriented files
`-- scripts/                   # corpus fetchers and notebook tooling
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/nabil0x/rag-playbook.git
cd rag-playbook

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Optional LLM Keys

Some notebooks use API LLMs for generation.

```bash
cp .env.example .env
```

Then add keys as needed:

```text
GOOGLE_API_KEY=...
GROQ_API_KEY=...
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
```

Local embedding labs do not require embedding API keys.

### 3. Run A Python Lab

```bash
python curriculum/01-chunking/01-fixed-vs-recursive.py
```

### 4. Run The End-To-End Pipeline

`main.py` uses Gemini for embeddings and generation, so it requires
`GOOGLE_API_KEY`.

```bash
python main.py
```

### 5. Open A Notebook

Start with:

```text
NoteBooks/Project-01-Baseline-RAG/04-baseline-rag.ipynb
```

Run notebooks from their own folder because paths are relative to the notebook
directory.

---

## My Learning Philosophy In This Repo

RAG quality is not created by one model call. It is created by the interaction
between data preparation, chunking, embedding choice, vector indexing, retrieval
strategy, context assembly, prompting, answer generation, and evaluation.

That is why this repo is organized as a playbook:

1. Build the simplest working system.
2. Change exactly one component.
3. Measure the effect.
4. Turn the lesson into reusable code.
5. Convert verified code into notebooks and teaching material.

The result is both a personal study record and a teaching resource for anyone
who wants to understand RAG beyond copy-paste demos.

---

## Current Notes

- Some advanced modules and project cards are still evolving.
- P20-P36 module stubs, topic cards, fresh corpora, and related scripts may be
  uncommitted depending on the current working tree.
- Vector-store folders and regenerated artifacts are intentionally excluded
  from git.
- See [`AGENTS.md`](AGENTS.md) for the standing conventions used while working
  in this repository.
