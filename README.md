# RAG Playbook

> **RAG from Zero to Advanced** — a hands-on, component-swappable curriculum for
> Retrieval-Augmented Generation, with a benchmark-driven research series on
> special document formats.

Three things live in this repo, all built on the same swappable pipeline:

1. **A curriculum** — 18 projects + a benchmark capstone. You build **one RAG
   application** and swap **one component at a time**: loader, splitter,
   embedding model, vector database, retriever, prompt, or LLM. Each swap is a
   project, so you can isolate the effect of every design decision and answer
   the questions that matter:

   - Why this chunk size?
   - Why this embedding model?
   - Why this retriever?
   - How do I measure the difference?

2. **A component library** — every block of the pipeline as a swappable class
   (19 implemented modules). Swap any block without touching the rest.

3. **A Special Documents research series** (`NoteBooks/SD-0N-*`) — real
   benchmark experiments on structured document formats most RAG tutorials
   skip: Word (.docx), PowerPoint (.pptx), Excel (.xlsx), Email (.eml), and
   more. Each one ships with sample data under `Data/SD-0N-<slug>/` and a
   notebook that measures naive vs structure-preserving parsing.

---

## The pipeline

Every project below is a RAG pipeline. Each block is replaceable — and each
block maps to a folder in this repo.

```
            Documents                        ← your data (web pages, PDFs, markdown…)
                │
                ▼
         Document Loader   loaders/          ← text in → LangChain Documents out
                │
                ▼
          Text Splitter    splitters/        ← big documents → small chunks
                │
                ▼
        Embedding Model    embeddings/       ← chunks → vectors (numbers)
                │
                ▼
        Vector Database    vectordb/         ← vectors stored, searched by similarity
                │
                ▼
           Retriever       retrieval/        ← query → top-k most relevant chunks
                │
                ▼
             Prompt         prompts/         ← package the chunks + the question
                │
                ▼
               LLM           llms/           ← reads the prompt → answers
                │
                ▼
              Answer
```

---

## Curriculum

Work through the projects in order. Each row changes one block of the pipeline.

| # | Project | What changes | Core concept |
|---|---------|--------------|--------------|
| 01 | [Baseline RAG](Topics/Project-01-Baseline-RAG/README.md) | — | Full pipeline with the simplest components |
| 02 | [PDF Knowledge Base](Topics/Project-02-PDF-Knowledge-Base/README.md) | Loader | PDF parsing |
| 03 | [Local Documentation Search](Topics/Project-03-Local-Documentation-Search/README.md) | Loader + Embedding + Vector DB | Offline embeddings, offline vector search |
| 04 | [Markdown Documentation RAG](Topics/Project-04-Markdown-Documentation-RAG/README.md) | Splitter + Prompt | Structure-preserving splitting |
| 05 | [HTML Documentation](Topics/Project-05-HTML-Documentation/README.md) | Loader + Splitter | HTML hierarchy |
| 06 | [Better Embeddings](Topics/Project-06-Better-Embeddings/README.md) | Embedding | Embedding model comparison |
| 07 | [Compare Vector Databases](Topics/Project-07-Compare-Vector-Databases/README.md) | Vector DB | Store comparison (index, query, memory) |
| 08 | [MMR Retrieval](Topics/Project-08-MMR-Retrieval/README.md) | Retriever | Diversity vs relevance |
| 09 | [Parent-Child Retrieval](Topics/Project-09-Parent-Child-Retrieval/README.md) | Splitter + Retriever | Small chunks in, large context out |
| 10 | [MultiQuery Retrieval](Topics/Project-10-MultiQuery-Retrieval/README.md) | Retriever | Query expansion |
| 11 | [Context Compression](Topics/Project-11-Context-Compression/README.md) | Retriever | Token reduction |
| 12 | [Prompt Engineering](Topics/Project-12-Prompt-Engineering/README.md) | Prompt | Basic → JSON → few-shot → citation → reasoning |
| 13 | [Multi-format RAG](Topics/Project-13-Multi-format-RAG/README.md) | Loader | PDF + Markdown + CSV + Web + JSON |
| 14 | [Metadata Filtering](Topics/Project-14-Metadata-Filtering/README.md) | Retriever | Scoped retrieval |
| 15 | [Hybrid Search](Topics/Project-15-Hybrid-Search/README.md) | Retriever | BM25 + dense fusion |
| 16 | [Build Without LangChain](Topics/Project-16-Build-Without-LangChain/README.md) | Everything | Pure-Python pipeline, no framework |
| 17 | [Modular RAG Framework](Topics/Project-17-Modular-RAG-Framework/README.md) | Architecture | Your own pluggable framework |
| 18 | [RAG Benchmark Suite](Topics/Project-18-RAG-Benchmark-Suite/README.md) | Capstone | Design, compare, and evaluate RAG systems |

Each project card (linked above) lists its stack, the component modules it
exercises, and the matching notebook. Drafted articles live inside the project
folders as the curriculum evolves.

---

## Special Documents research series

The `SD-*` notebooks are a parallel track: what happens when your documents
aren't clean PDFs and markdown? Every entry uses the same benchmark methodology
(Project 18's question set) to compare **naive** parsing against
**structure-preserving** parsing — and reports real numbers.

| # | Format | Sample data (`Data/SD-0N-<slug>/`) | Notebooks | Status |
|---|--------|------------------------------------|-----------|--------|
| 01 | Word (`.docx`) | 4 real documents — FCC notice, EPA TSD, UNDP evaluation, docx4j guide | `SD-01-Word-Documents/01…03` | ✅ Shipped |
| 02 | PowerPoint (`.pptx`) | `prs-notes.pptx` | `SD-02-PowerPoint/01` | ✅ Shipped |
| 03 | Excel (`.xlsx`) | `SampleSS.xlsx` | `SD-03-Excel/01` | ✅ Shipped |
| 04 | Email (`.eml`) | `raw_email_with_nested_attachment.eml` | `SD-04-Email-Threads/01` | ✅ Shipped |
| 05 | Scanned/OCR | `gilman1892.pdf`, `gitanjali1914_jp2.zip` | — | 🟡 Planned |
| 06 | Tables & forms | SEC filing (`.htm`), bank statements (`bkash`, `nagad`), IRS `f1040.pdf` | — | 🟡 Planned |
| 07 | Chat transcripts | `sample-chat.txt`, `bangla_chat.txt` | — | 🟡 Planned |
| 08 | Invoices (multilingual) | restaurant bill, `Invoice_1.pdf`, `mushak63_invoice.pdf` | — | 🟡 Planned |

**Headline findings so far (SD-01, Word):**

- **Naive character splitting destroys document structure.** FCC EAS
  questionnaire accuracy dropped from `0.75` (structured) to `0.50` (naive)
  because table cells were torn apart mid-question.
- **Structure-preserving loaders fix title pollution.** EPA TSD rose from
  `0.33` → `1.00`; UNDP evaluation from `0.00` → `0.33+` once page-footers and
  header boilerplate stopped contaminating chunks.
- **Benchmark totals:** naive `5–6/14` vs structured `8–9/14` across four
  documents — structure wins even when it produces *more* chunks.
- See `NoteBooks/SD-01-Word-Documents/03-word-rag-pure-langchain.ipynb` for the
  full comparison table and methodology.

---

## Repository map — everything in this repo

```
rag-playbook/
├── README.md                     ← you are here
├── requirements.txt              ← Python dependencies
├── .env.example                  ← copy to .env, add your API keys
├── .gitignore                    ← keeps secrets & regenerable artifacts out of git
├── main.py                       ← runnable RAGPipeline wiring (Project 17)
│
├── Topics/                       ← publishable content — one folder per project
│   ├── README.md                 ← content map + per-project status
│   └── Project-01-…/Project-18-… ← 18 project cards (stack, modules, notebooks)
│
├── NoteBooks/                    ← runnable notebooks (run each from its own folder)
│   ├── Project-01-…/Project-17-… ← one notebook per curriculum project
│   └── SD-01-Word-Documents/     ← Special Documents series (SD-01…SD-08)
│       └── 01…03-*.ipynb         ← naive vs structured benchmark experiments
│
├── Data/                         ← sample documents (public domain), at repo root
│   ├── sample.csv, sample.json, local-docs/   ← Project 01 fixtures
│   └── SD-01-word/ … SD-08-invoices/          ← Special Documents samples
│
├── loaders/                      ← pipeline block #1: text in → Documents out
├── splitters/                    ← pipeline block #2: documents → chunks
├── embeddings/                   ← pipeline block #3: chunks → vectors
├── vectordb/                     ← pipeline block #4: store & search vectors
├── retrieval/                    ← pipeline block #5: query → top-k chunks
├── prompts/                      ← pipeline block #6: chunks + question → prompt
├── llms/                         ← pipeline block #7: prompt → answer
└── scripts/                      ← tooling (fetch SD samples, generate fixtures)
```

### Component library (the pluggable blocks)

Every file is one class implementing the same contract as its siblings, so you
can swap any block without touching the rest of the pipeline.

| Module | Class | What it does | Status |
|--------|-------|--------------|--------|
| `loaders/web.py` | `WebLoader` | Scrapes one URL (requests + BeautifulSoup) → `list[Document]` | ✅ Implemented |
| `loaders/pdf.py` | `PDFLoader` | Parses a PDF (PyPDFLoader) → documents | ✅ Implemented |
| `loaders/csv_loader.py` | `CSVLoader` | Reads CSV rows → documents | ✅ Implemented |
| `splitters/recursive.py` | `DocumentProcessor` | Splits documents into overlapping chunks (RecursiveCharacterTextSplitter) | ✅ Implemented |
| `splitters/semantic.py` | `SemanticSplitter` | Splits on semantic similarity (embedding-aware) | ✅ Implemented |
| `splitters/token_splitter.py` | `TokenSplitter` | Splits by token budget | ✅ Implemented |
| `embeddings/gemini.py` | `GeminiEmbedding` | Gemini embeddings (Google AI Studio) — the default model | ✅ Implemented |
| `embeddings/bge.py` | `BGEEmbedding` | Local BGE embeddings (sentence-transformers) | ✅ Implemented |
| `embeddings/e5.py` | `E5Embedding` | Local E5 embeddings | ✅ Implemented |
| `vectordb/chroma.py` | `ChromaVectorStore` | Chroma — persistent, the default store | ✅ Implemented |
| `vectordb/faiss.py` | `FAISSVectorStore` | FAISS — fast in-memory index | ✅ Implemented |
| `vectordb/qdrant.py` | `QdrantVectorStore` | Qdrant — server-based store | ✅ Implemented |
| `retrieval/similarity.py` | `SimilarityRetriever` | Top-k by vector similarity | ✅ Implemented |
| `retrieval/mmr.py` | `MMRRetriever` | Maximum Marginal Relevance — diverse results | ✅ Implemented |
| `retrieval/hybrid.py` | `HybridRetriever` | BM25 keyword + dense vector fusion | ✅ Implemented |
| `prompts/basic.py` | `BasicPrompt` | "Answer from context" template + `format()` | ✅ Implemented |
| `prompts/citation.py` | `CitationPrompt` | "Answer with source citations" template + `format()` | ✅ Implemented |
| `llms/gemini.py` | `GeminiLLM` | Gemini chat completions — the default LLM | ✅ Implemented |
| `llms/openai.py` | `OpenAILLM` | OpenAI chat completions | ✅ Implemented |

**All 19 blocks are implemented.** Run any module directly
(`python loaders/csv_loader.py`, `python splitters/token_splitter.py`, …) for
a self-check, or the whole pipeline with `python main.py` (needs a
`GOOGLE_API_KEY` in `.env`). Every component that depends on an optional
package (`pypdf`, `sentence-transformers`, `faiss-cpu`, `langchain-qdrant`,
`langchain-openai`) imports it lazily and prints a `SKIP: pip install …` hint
when it's missing.

### Notebooks — one per project, under `NoteBooks/Project-NN-*`

Every project ships a structured, runnable notebook: small section-wise cells,
setup → load → split → embed → store → retrieve → prompt → answer, plus
"What you should notice" and exercises. Project 01 has four (basics + the
baseline); Projects 02–17 have one each. Run each notebook from **its own
folder** — paths are relative to the notebook directory.

| Project | Notebook | What it teaches |
|---------|----------|-----------------|
| 01 | `Project-01-Baseline-RAG/01-langchain-intro.ipynb` | LangChain basics — loaders and splitters (`WebBaseLoader`, `RecursiveCharacterTextSplitter`) |
| 01 | `Project-01-Baseline-RAG/02-document-loader.ipynb` | Rolling your own loader — requests + BeautifulSoup → `Document` → splitter |
| 01 | `Project-01-Baseline-RAG/03-ingestion-pipeline.ipynb` | Ingestion template — build your own end-to-end ingestion step |
| 01 | `Project-01-Baseline-RAG/04-baseline-rag.ipynb` | **The baseline** — load → chunk → embed (Gemini) → store (Chroma) → answer |
| 02 | `Project-02-PDF-Knowledge-Base/01-pdf-knowledge-base.ipynb` | PDF parsing into a searchable knowledge base |
| 03 | `Project-03-Local-Documentation-Search/01-local-documentation-search.ipynb` | Offline BGE embeddings + FAISS, no API keys |
| 04 | `Project-04-Markdown-Documentation-RAG/01-markdown-documentation-rag.ipynb` | Structure-preserving markdown splitting |
| 05 | `Project-05-HTML-Documentation/01-html-documentation.ipynb` | HTML hierarchy — load & split by document structure |
| 06 | `Project-06-Better-Embeddings/01-comparing-embedding-models.ipynb` | Embedding model comparison (Gemini vs local) |
| 07 | `Project-07-Compare-Vector-Databases/01-comparing-vector-databases.ipynb` | Chroma vs FAISS vs Qdrant — index, query, memory |
| 08 | `Project-08-MMR-Retrieval/01-mmr-retrieval.ipynb` | MMR — diversity vs relevance |
| 09 | `Project-09-Parent-Child-Retrieval/01-parent-child-retrieval.ipynb` | Small chunks in, large context out |
| 10 | `Project-10-MultiQuery-Retrieval/01-multiquery-retrieval.ipynb` | Query expansion via `MultiQueryRetriever` |
| 11 | `Project-11-Context-Compression/01-context-compression.ipynb` | Token reduction via contextual compression |
| 12 | `Project-12-Prompt-Engineering/01-prompt-engineering.ipynb` | Basic → JSON → few-shot → citation → reasoning |
| 13 | `Project-13-Multi-format-RAG/01-multi-format-rag.ipynb` | PDF + Markdown + CSV + Web + JSON loaders |
| 14 | `Project-14-Metadata-Filtering/01-metadata-filtering.ipynb` | Scoped retrieval with metadata filters |
| 15 | `Project-15-Hybrid-Search/01-hybrid-search.ipynb` | BM25 keyword + dense vector fusion (`EnsembleRetriever`) |
| 16 | `Project-16-Build-Without-LangChain/01-rag-without-langchain.ipynb` | Pure-Python pipeline, no framework |
| 17 | `Project-17-Modular-RAG-Framework/01-modular-rag-framework.ipynb` | Your own pluggable `RAGPipeline` framework |

> `langchain` 1.3.14 note: retriever classes (Project 09–11, 15) live in the
> `langchain-classic` package (`langchain_classic.retrievers`). The notebooks
> import them with a `try/except` fallback, and `requirements.txt` now includes
> `langchain-classic`.

All sample documents live under `Data/` at the **repo root** — `sample.csv`,
`sample.json`, a local markdown doc set under `local-docs/`, and one folder per
Special Document (`SD-01-word/` … `SD-08-invoices/`). The
`chroma_langchain_db/` folders inside the notebooks directory are regenerable
vector-store artifacts.

### `main.py` — the endgame

```python
pipeline = RAGPipeline(
    loader=CSVLoader("Data/sample.csv"),
    splitter=DocumentProcessor(),     # any splitter — swap freely
    embedder=GeminiEmbedding(),      # any embedding model
    db=ChromaVectorStore(embedding=embedder),  # any vector store
    retriever=SimilarityRetriever(db),         # any retriever
    llm=GeminiLLM(),                 # any LLM
)
pipeline.ingest()                    # load → split → embed → store
pipeline.ask("What is the sample about?")
```

This is the **endgame**: `RAGPipeline` in `main.py` wires the six blocks together
and every block is a swappable class from the component library above. Run
`python main.py` from the repo root to see the whole pipeline work end to end
(needs a `GOOGLE_API_KEY` in `.env`). By the end of Project 17 you build exactly
this from the components you implemented along the way.

### Configuration

- `.env.example` — template with the keys used by the projects:
  `GOOGLE_API_KEY` (Gemini — most projects), `GROQ_API_KEY` (fast local-hosted
  LLM option), `LANGSMITH_API_KEY` (optional tracing).
- `requirements.txt` — `langchain` stack plus `pypdf` and `python-dotenv`;
  the optional vector-DB drivers (`faiss-cpu`, `langchain-qdrant`, …) are listed
  as comments, activated when you reach Project 07.
- `.gitignore` — keeps `.env`, `__pycache__/`, vector-store artifacts
  (`chroma_langchain_db/`, `*.sqlite3`, `lancedb/`, `qdrant_storage/`), the
  large Enron corpus (`Data/SD-04-email/enron/`), and tooling state out of
  version control.

---

## Try it now

Only the implemented components, no API keys needed. Run this from the repo
root to see loading → chunking → prompting work end to end:

```bash
python -c "
from loaders.web import WebLoader
from splitters.recursive import DocumentProcessor
from prompts.basic import BasicPrompt

docs = WebLoader('https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i').load()
chunks = DocumentProcessor(chunk_size=1000, chunk_overlap=200).split_docs(docs)

print(f'Loaded {len(docs)} document(s), split into {len(chunks)} chunks')
print(BasicPrompt().format(context=chunks[0].page_content, question='What is RAG?')[:600])
"
```

---

## Getting started

**Prerequisites**

- Python 3.10+
- A free [Google AI Studio](https://aistudio.google.com/) API key for Gemini
  (most projects use Gemini embeddings + LLM by default)

**Install**

```bash
git clone https://github.com/nabil0x/rag-playbook.git && cd rag-playbook
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your API keys
```

**Run the first project**

Open `NoteBooks/Project-01-Baseline-RAG/04-baseline-rag.ipynb` and run it end
to end — that is Project 01, the full baseline pipeline. The companion article
is `Topics/Project-01-Baseline-RAG/01-baseline-rag.md`.

**Follow along**

1. Work through the projects in order. Each one changes one block.
2. When you hit an advanced project (06–15), read its card in `Topics/` first,
   then implement the component in the matching module.
3. Finish with Project 17 (build the framework) and Project 18 (benchmark it).
4. Curious about hard document formats? Try the Special Documents series —
   start with `NoteBooks/SD-01-Word-Documents/01-word-documents-rag.ipynb`.

---

## Roadmap — build order

All components are implemented; this is the dependency order the projects
unlock them in — and the order you'd rebuild them from scratch in Project 16
(pure Python) and Project 17 (your own framework):

1. `embeddings/gemini.py` → `vectordb/chroma.py` → `llms/gemini.py`
   — completes the **Project 01** pipeline as plain code.
2. `loaders/pdf.py` — unlocks **Project 02**.
3. `embeddings/bge.py` + `vectordb/faiss.py` — unlocks **Project 03** (offline).
4. `vectordb/qdrant.py` — **Project 07** store comparison.
5. `retrieval/mmr.py` → `retrieval/hybrid.py` — **Projects 08 & 15**.
6. `splitters/semantic.py` + `splitters/token_splitter.py` — **Projects 09 & 11**.
7. `llms/openai.py` — optional alternative backend.
8. **Project 16** — rewrite each block in pure Python (no LangChain).
9. **Project 17** — assemble `RAGPipeline` in `main.py` from your classes.

**Special Documents series** — SD-05…SD-08 notebooks are next: scanned/OCR,
tables & forms, chat transcripts, and multilingual invoices. Sample data
already ships in `Data/`.

---

## Notes

- Sample data under `Data/` is public-domain text from
  [Project Gutenberg](https://www.gutenberg.org/) plus small original fixtures.
- Vector database folders (e.g. `chroma_langchain_db/`) are regenerable
  artifacts and gitignored.
- This repo grows with you: each project card, article, component, and Special
  Documents experiment lands as the curriculum evolves.
