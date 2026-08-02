# Canonical Notebook Spec — RAG Curriculum (Projects 01–17)

Every notebook in this repo follows the same teaching philosophy and structure.
Read this file BEFORE writing any notebook. Also read
`NoteBooks/Project-01-Baseline-RAG/04-baseline-rag.ipynb` as a reference for the
real library imports and patterns used in this repo.

## Teaching philosophy

- **Small section-wise cells.** One concept per cell. Never put two logical steps
  in one code cell. A section = one markdown cell (the teacher) + 1–3 short code
  cells (the demo). This is the core requirement: "structured cell management,
  small section-wise cells".
- **Teacher-first markdown.** Every markdown cell explains WHAT the step does,
  WHY it matters for RAG, and WHAT to expect in the output. No bare headers.
- **Runnable, real code.** Notebooks use the real libraries already in
  `requirements.txt` — never the repo stub modules (they raise
  `NotImplementedError`). Match the imports of `04-baseline-rag.ipynb`.
- **Honest about optional deps.** If a step needs an optional package
  (sentence-transformers, faiss-cpu, langchain-qdrant), include an install
  cell in Setup and note it's needed for this project only.
- **Learning-phase documentation.** Each notebook ends with
  "What you should notice" (takeaways specific to this project's changed block)
  and "Exercises" (stretch tasks).

## Canonical structure — full-pipeline projects

Follow exactly for projects that run the whole pipeline (02, 03, 04, 05, 09, 13, 14):

```
[md]  # Project NN — <Title>
[md]  > Goal: <one-line goal from the project card>
[md]  ## 0 · Setup — environment & keys
[md]  Small explanation: .env file, GOOGLE_API_KEY, optional installs for THIS project.
[code] load_dotenv() + masked key check (print key[:4]+"…" if set, else friendly error)
[code] (optional) pip install cell for this project's optional deps
[code] imports (grouped, one cell)
[md]  ## 1 · Load
[md]  Explain this project's loader, why it's used, what Document objects look like.
[code] loader init + load
[code] peek: document count + first doc's metadata/snippet
[md]  ## 2 · Split
[md]  Explain chunk_size / chunk_overlap for THIS project's splitter.
[code] splitter init + split
[code] peek: chunk count + one sample chunk
[md]  ## 3 · Embed
[md]  Explain embeddings in plain words (text → vectors, similar text → near vectors).
[code] embed a sample string, show vector dimensionality
[md]  ## 4 · Store
[md]  Explain the vector DB for THIS project + persistence notes.
[code] create vector store from chunks
[md]  ## 5 · Retrieve
[md]  Explain the retriever for THIS project + top-k concept.
[code] query + similarity search (top-k)
[code] show retrieved chunks with scores/snippets
[md]  ## 6 · Prompt
[md]  Explain how the prompt packages context + question. Show the template.
[code] build prompt template (ChatPromptTemplate) + render an example
[md]  ## 7 · Answer
[md]  Explain the LLM call for THIS project.
[code] llm.invoke + print answer
[md]  ## 8 · Try it yourself
[md]  Short note: change the query, change k, etc.
[code] 1–2 extra query cells
[md]  ## What you should notice
[md]  3–6 bullets: takeaways specific to the block this project changes.
[md]  ## Exercises
[md]  3 stretch tasks (change a parameter, add a feature, compare with previous project).
```

## Canonical structure — comparison/rotation projects

For projects that rotate one block and compare (06 embeddings, 07 vector DBs, 12 prompts):

```
[md]  # Project NN — <Title> + goal
[md]  ## 0 · Setup (same as above)
[md]  ## 1 · Load + 2 · Split (same as above — keep them short, the point is the comparison)
[md]  ## 3 · The rotation
[md]  Explain what is being rotated (model/store/prompt list from the project card).
[code] define a reusable function: run(query) -> retrieved chunks / answer, parameterized by the rotated block
[code] loop over each variant, collect results
[md]  Show a comparison table (markdown): model | dims | retrieval quality / latency / notes
[md]  ## 4 · Deep-dive on the winner (or each)
[md]  ## What you should notice (comparison takeaways)
[md]  ## Exercises
```

## Real library imports cheat-sheet (verified against this repo's requirements.txt)

```python
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    WebBaseLoader, PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.retrievers import (
    ParentDocumentRetriever, MultiQueryRetriever, ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
```

Model names used in this repo (match the reference notebook):
- Gemini embeddings: `GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")`
- Gemini LLM: `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`

Optional deps (install cell in Setup, commented note):
- BGE embeddings: `pip install sentence-transformers` → `HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")`
- FAISS: `pip install faiss-cpu`
- Qdrant: `pip install langchain-qdrant` → `from langchain_qdrant import Qdrant`

Sample data: `NoteBooks/Data/` (public domain, Project Gutenberg). For PDF
projects, instruct the user to drop a PDF into `NoteBooks/Data/` and set a
`PDF_PATH` variable; offer a fallback download from Gutenberg if the file is
missing.

## Generator usage (mandatory)

1. Write a Python spec file (anywhere under `/tmp/opencode/`) defining:
   ```python
   CELLS = [
       {"type": "md", "source": "# Title\n\nbody"},
       {"type": "code", "source": "import os\n"},
   ]
   ```
2. Generate the notebook:
   ```bash
   python3 scripts/gen_notebook.py /tmp/opencode/<spec>.py NoteBooks/Project-NN-<Name>/01-<slug>.ipynb
   ```
3. Validate your output:
   ```bash
   python3 -c "
   import json
   nb = json.load(open('<out>'))
   assert nb['nbformat'] == 4
   cells = nb['cells']
   assert cells and cells[0]['cell_type'] == 'markdown'
   for c in cells: assert c['source'] and isinstance(c['source'], str)
   md = sum(1 for c in cells if c['cell_type'] == 'markdown')
   code = sum(1 for c in cells if c['cell_type'] == 'code')
   print(f'OK: {len(cells)} cells ({md} md / {code} code)')"
   ```
   The final output must print `OK: N cells (M md / C code)`.

## MUST DO

- Small section-wise cells: split logic into many tiny code cells; md cells teach each step.
- Every markdown section explains WHAT + WHY (learning documentation).
- Use real libraries only (cheat-sheet above); the repo stub modules are off-limits.
- Match the kernelspec/format produced by the generator (never hand-write ipynb JSON).
- End with "What you should notice" + "Exercises" markdown sections.
- Keep code cells ≤ ~15 lines; if a step is long, split it into more cells.
- Sanitize keys: only print masked key presence, never the full key.

## MUST NOT DO

- Do NOT edit anything outside the notebook(s) you are assigned.
- Do NOT modify `Topics/`, `README.md`, `requirements.txt`, repo stub modules, or `scripts/`.
- Do NOT delete or rename the existing Project-01 notebooks.
- Do NOT execute notebook cells (no API calls, no model downloads, no training).
- Do NOT use `nbformat` (not installed); always go through `scripts/gen_notebook.py`.
- Do NOT invent libraries that are not in `requirements.txt` without an install cell.
- No bare markdown headers — every section header must have explanation text under it.
