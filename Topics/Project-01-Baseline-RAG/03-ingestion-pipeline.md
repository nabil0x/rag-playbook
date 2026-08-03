> Source notebook: `NoteBooks/Project-01-Baseline-RAG/03-ingestion-pipeline.ipynb`


---

# Project 01 · Notebook 3 — The Ingestion Pipeline Template

> **Goal:** Build your own reusable **ingestion** step — the half of RAG that
> happens *before* any question is asked: load → split → embed → store.

This notebook is a **template with four empty cells**, one per stage. It is your
job to fill each cell using what you learned in notebooks 1 and 2. When you are
done you will have a documented, re-runnable ingestion pipeline that feeds a
vector store — exactly what the baseline RAG notebook expects to find.

## What you will build, step by step

| Cell | Stage        | What to put in it                                            |
|------|--------------|--------------------------------------------------------------|
| 1    | Setup        | imports (`dotenv`, loader, splitter, embeddings, Chroma)     |
| 2    | Load         | loader init + `.load()` → `docs`                             |
| 3    | Split        | splitter init + `split_documents(docs)` → `chunks`           |
| 4    | Embed & Store| embeddings + `Chroma.from_documents(chunks)`                 |

---

## 0 · Setup — imports & environment

**WHAT:** An empty cell waiting for your imports and `load_dotenv()`.

**WHY:** Keeping environment loading and imports in one dedicated cell means the
rest of the pipeline reads like a recipe: load, then transform, then store.

**WHAT TO EXPECT:** No output (the cell is empty) — but once you fill it, the
cell should run without errors, proving every library is installed and your
`.env` is set.

---

## 1 · Load — get your documents

**WHAT:** Your turn to instantiate a loader — `WebBaseLoader` from notebook 1, or
the custom `requests` + BeautifulSoup loader you built in notebook 2 — and call
`.load()` into a `docs` variable.

**WHY (for RAG):** Ingestion starts with documents. Whatever data source you
choose, the contract is the same: a list of LangChain `Document`s.

**WHAT TO EXPECT:** Once filled, `len(docs)` should print a small number — how
many documents were loaded.

---

## 2 · Split — chunk the documents

**WHAT:** Create a `RecursiveCharacterTextSplitter` (try
`chunk_size=1000, chunk_overlap=200`) and produce `chunks` with
`split_documents(docs)`.

**WHY (for RAG):** Chunks are what gets embedded and retrieved. The splitter
determines the granularity of everything the retriever can later find.

**WHAT TO EXPECT:** `len(chunks)` should be noticeably larger than `len(docs)`.

---

## 3 · Embed & Store — index the chunks

**WHAT:** Create `GoogleGenerativeAIEmbeddings` and persist the chunks with
`Chroma.from_documents(documents=chunks, embedding=embeddings)`.

**WHY (for RAG):** This is the last ingestion stage — text becomes vectors and
the vectors are indexed so a retriever can search them. After this step your
data is "RAG-ready".

**WHAT TO EXPECT:** A `Chroma` vector store object, and a
`chroma_langchain_db/` folder appearing on disk.

---

## What you should notice

- **Ingestion is a one-time cost.** Loading, splitting, embedding and storing
  happen once; every question you ask later reuses the same index.
- **The stages are independent.** You can swap any cell's implementation (a
  different loader, splitter, or store) without touching the other three — that
  is the design philosophy of this whole curriculum.
- **The contract is `Document` in, vector store out.** As long as each stage
  respects that shape, the pipeline stays runnable.

---

## Exercises

1. **Persist the store.** Give your `Chroma` a `persist_directory` and verify
   the `chroma_langchain_db/` folder appears on disk.
2. **Inspect what you indexed.** Re-open the store in a later cell and run a
   `similarity_search` on one chunk to confirm it is retrievable.
3. **Refactor into a function.** Turn the four cells into a single
   `ingest(docs)` function you can call from any notebook.
