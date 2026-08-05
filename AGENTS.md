# AGENTS.md — RAG Playbook working conventions

Standing rules for anyone (human or AI) working in this repo. These are the
defaults; override them only when a task explicitly says otherwise.

## 1. Local embeddings only (never API embeddings by default)

- Every pipeline, lab, and notebook embeds with a **local model**:
  sentence-transformers, BGE `BAAI/bge-base-en-v1.5` (default,
  `src/embeddings/bge.py`) or E5 (`src/embeddings/e5.py`).
- If a suitable local model/package is **not installed, install it**
  (e.g. `pip install sentence-transformers faiss-cpu …`). Do not switch to an
  API embedding as a workaround.
- API models (Gemini/OpenAI) are used as the **LLM** when a lab needs
  generation — never as the default embedding path.

## 2. Python code first, notebooks after verification

- Every concept starts as a standalone `.py` in
  `src/curriculum/<NN>-<concept>/`.
- The `.py` must **run and pass its verification gate** (concrete output +
  `py_compile` clean) **before** it is converted to a notebook under
  `NoteBooks/Curriculum-<NN>-<concept>/`.
- Never write the notebook first.

## 3. LangChain ecosystem first; build your own only for the gaps

- Use pure LangChain and its associated frameworks: `langchain`,
  `langchain-community`, `langchain-huggingface`, `langchain-chroma`,
  `langchain-classic` (retriever classes), `langchain-text-splitters`,
  LangGraph.
- If LangChain does not support an operation, build a custom class
  implementing the same contract and put it in `src/tools/` (shared, reusable —
  not notebook-embedded). See `src/tools/README.md` for the inventory.
- Reuse the repo's component library before writing anything:
  `src/loaders/`, `src/splitters/`, `src/embeddings/`, `src/vectordb/`, `src/retrieval/`,
  `src/prompts/`, `src/llms/`, `src/evaluation/`.

## 4. Data

- Fresh corpora live in `Data/corpus/`; provenance is
  `Data/.corpus-manifest.txt` (`url|sha256|bytes|relpath` per line). Never
  re-download what the manifest verifies; use `src/scripts/fetch_fresh_corpus.py`
  for new/forced fetches (`--mind` gates the research-licensed MIND set).
- SD samples: `Data/.samples-manifest.txt` + `src/scripts/fetch_sd_samples.py`.
- Corpus map: rag-mini-wikipedia (core), beir-fiqa/nfcorpus (qrels for
  retrieval+rerank eval), lost-in-the-middle (position bias), scifact
  (attribution/verification), hotpotqa (multi-hop/GraphRAG/agentic),
  gutenberg (public-domain long prose — chunking labs; fetched via
  `src/scripts/fetch_gutenberg.py`, manifest-verified like the rest).

## 5. Repo layout

```
src/                   all Python (components, labs, scripts, service, entrypoints)
  curriculum/          Layer-1 playbook: python-first labs (10 concept tracks)
  tools/               custom classes where LangChain has gaps
  loaders/…evaluation/ shared swappable component library (the pipeline blocks)
  scripts/             fetchers and tooling
NoteBooks/             runnable notebooks (Project-0N, SD-0N, Curriculum-NN)
Topics/                publishable project cards
Data/                  samples + corpus + manifests
.omo/plans/            approved plans (this repo plans by plan-doc)
```

## 6. Evaluation conventions

- Retrieval metrics (Recall@k, MRR, nDCG) evaluate against **qrels**:
  `Data/corpus/beir-fiqa/fiqa/qrels/*.tsv`,
  `Data/corpus/beir-nfcorpus/nfcorpus/qrels/*.tsv`.
- Generation metrics use gold answers: `rag-mini-wikipedia/test.parquet`,
  `hotpotqa` dev set, `scifact` claims — via `src/evaluation/`.
- LLM-as-judge runs are cross-checked against reference-based metrics
  (kappa) before they are trusted.

## 7. Repo state

- P20–P36 module stubs, Topics cards, `Data/corpus/`, and
  `src/scripts/fetch_fresh_corpus.py` are currently **uncommitted**. Nothing is
  committed or moved without explicit request.
- Commit only when asked; inspect `git status`/`git diff` first.
