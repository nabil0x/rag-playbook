"""Lab 04 — Parent-child retrieval: small chunks in, large context out.

Plain chunked retrieval has a tension: small chunks match a query precisely
but carry too little surrounding context for an answer; large chunks carry
context but match a query coarsely. Parent-child retrieval splits the
difference — embed SMALL child chunks for precise matching, but store the
LARGER parent document each child came from, and return the PARENT whenever a
child hits.

This lab builds a ``ParentDocumentRetriever`` (from ``langchain-classic``,
the retriever home of the LangChain 1.x era) over a deterministic subset of
``Data/corpus/rag-mini-wikipedia``:

* PARENTS  — first 20 passages of the corpus, re-split at ``PARENT_CHUNK_SIZE``
  (500 chars): these are the large contexts that get returned.
* CHILDREN — each parent split again at ``CHILD_CHUNK_SIZE`` (120 chars):
  these are the small pieces that get embedded and matched.
* The retriever keeps a ``docstore`` mapping every child's ``doc_id`` back to
  its parent, so a query that matches a 120-char child returns the whole
  ~500-char parent.

Local embeddings only (BGE via sentence-transformers); no LLM, no API keys.
The vector store is FAISS in-memory — nothing is written to disk.

Run from the repo root:
    python src/curriculum/04-retrieval/04-parent-child.py
    python src/curriculum/04-retrieval/04-parent-child.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/04-retrieval/04-parent-child.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langchain_classic.retrievers import ParentDocumentRetriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.stores import InMemoryStore  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PARENTS = 20  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1604]  # real questions from test.parquet, answers inside the subset
CHILD_CHUNK_SIZE = 120  # small chunks: embedded and matched against the query
CHILD_OVERLAP = 20
PARENT_CHUNK_SIZE = 500  # large contexts: returned to the caller
PARENT_OVERLAP = 50
K = 3  # search_kwargs: how many parents each query returns
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
PREVIEW = 62  # max characters of chunk text shown next to each hit


# --------------------------------------------------------------------------
# 2. Load — corpus + questions from the fresh rag-mini-wikipedia parquet files
# --------------------------------------------------------------------------
def load_parents(path: Path, n: int) -> list[Document]:
    """Return the first ``n`` passages as parent Documents (id -> doc_id)."""
    df = pd.read_parquet(path)
    subset = df.head(n)
    return [
        Document(page_content=text, metadata={"doc_id": str(i)})
        for i, text in enumerate(subset["passage"].tolist())
    ]


def load_questions(path: Path, ids: list[int]) -> list[tuple[int, str]]:
    """Return [(question_id, question_text)] for the requested test rows."""
    df = pd.read_parquet(path)
    rows = df.loc[ids]
    return [(int(idx), row["question"]) for idx, row in rows.iterrows()]


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a chunk for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --------------------------------------------------------------------------
# 3. Experiment — split into children, embed, index, query; returns every
#    artifact the demo and the verification gate need (no re-computation
#    between the two paths)
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    parents = load_parents(PASSAGES_PATH, N_PARENTS)
    questions = load_questions(TEST_PATH, QUESTION_IDS)

    # --- Splitters: one for parents (big), one for children (small) ---------
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_OVERLAP
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_OVERLAP
    )

    # --- Local BGE embeddings (never an API model) --------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- In-memory stores ---------------------------------------------------
    # FAISS cannot infer the embedding dimension from an empty list, so seed
    # it with one real parent and delete the seed again (index stays empty).
    seed = FAISS.from_documents([parents[0]], embedding=embeddings)
    seed_id = next(iter(seed.index_to_docstore_id.values()))
    seed.delete([seed_id])
    vs = seed  # holds the CHILDREN
    docstore = InMemoryStore()  # holds the PARENTS, keyed by doc_id

    retriever = ParentDocumentRetriever(
        vectorstore=vs,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": K},
    )

    # --- Split + embed + index in one call ----------------------------------
    t0 = time.perf_counter()
    retriever.add_documents(parents, add_to_docstore=True)
    ingest_s = time.perf_counter() - t0

    # Children live inside the FAISS vector store, keyed by index position in
    # index_to_docstore_id; every child knows its parent via metadata.doc_id.
    child_ids = list(vs.index_to_docstore_id.values())
    children = vs.get_by_ids(child_ids)
    parent_ids = list(docstore.yield_keys())
    id_to_parent = dict(zip(parent_ids, docstore.mget(parent_ids)))

    # --- Query: invoke() returns PARENTS; a direct vector search shows the
    #     CHILD that matched ------------------------------------------------
    results = []
    for qid, qtext in questions:
        retrieved = retriever.invoke(qtext)  # K parents
        matched_child, score = vs.similarity_search_with_score(qtext, k=1)[0]
        results.append(
            {
                "qid": qid,
                "qtext": qtext,
                "parents": retrieved,
                "child": matched_child,
                "child_score": score,
                "child_parent": id_to_parent.get(matched_child.metadata.get("doc_id")),
            }
        )

    parent_lens = [len(d.page_content) for d in id_to_parent.values()]
    child_lens = [len(c.page_content) for c in children]

    return {
        "n_parents": len(parent_ids),
        "n_children": len(children),
        "children": children,
        "parent_lens": parent_lens,
        "child_lens": child_lens,
        "results": results,
        "ingest_s": ingest_s,
        "n_queries": len(questions),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    n_parents, n_children = exp["n_parents"], exp["n_children"]
    mean_p = sum(exp["parent_lens"]) / len(exp["parent_lens"])
    mean_c = sum(exp["child_lens"]) / len(exp["child_lens"])

    print("=" * 66)
    print("Lab 04 — Parent-child retrieval: small chunks in, large context out")
    print(f"{BGE_MODEL_NAME} | FAISS (in-memory) | ParentDocumentRetriever")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {n_parents} parents (first {N_PARENTS} passages of 3200, split at {PARENT_CHUNK_SIZE} chars)")
    print(f"    {n_children} children (embedded at {CHILD_CHUNK_SIZE} chars)")
    print(f"    {exp['n_queries']} questions from test.parquet:")
    for r in exp["results"]:
        print(f"      [{r['qid']}] {r['qtext']}")

    print(f"\n[2] Split + embed + index:")
    print(f"    {n_parents} parents -> {n_children} children in {exp['ingest_s']:.2f}s")
    print(f"    mean parent length {mean_p:.0f} chars vs mean child length {mean_c:.0f} chars"
          f" ({mean_p / mean_c:.1f}x)")
    print(f"    children >> parents: {n_children} child vectors indexed, "
          f"{n_parents} parent docs in the docstore")

    print(f"\n[3] Top-{K} per question (each hit is a PARENT, matched via its children):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["qtext"]}"')
        for rank, doc in enumerate(r["parents"], 1):
            print(f"      {rank}. PARENT ({len(doc.page_content)} chars) "
                  f"[{preview(doc.page_content)}]")
        child = r["child"]
        print(f"      matched CHILD ({len(child.page_content)} chars, "
              f"score {r['child_score']:.4f}):")
        print(f"        [{preview(child.page_content)}]")
        cp = r["child_parent"]
        if cp is not None:
            print(f"      child -> parent ({len(cp.page_content)} chars): the returned "
                  f"context is {len(cp.page_content) / max(len(child.page_content), 1):.1f}x "
                  f"the chunk that matched")

    print("\n[4] Takeaway")
    print("    The retriever embeds 120-char children for precise matching, then")
    print("    maps each hit back through metadata.doc_id to its ~500-char parent.")
    print("    Precision comes from the small chunk, context from the large one —")
    print("    the 'small chunks in, large context out' trick of Project 09.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # The child splitter actually split: many small chunks from few parents.
    checks.append((f"children ({exp['n_children']}) outnumber parents ({exp['n_parents']})",
                   exp["n_children"] > exp["n_parents"]))
    checks.append(("no empty child chunks",
                   all(len(c.page_content) > 0 for c in exp["children"])))

    # Every retrieved hit is a PARENT (longer than any child chunk)...
    all_parents = all(
        len(d.page_content) > CHILD_CHUNK_SIZE
        for r in exp["results"] for d in r["parents"]
    )
    checks.append(("every retrieved doc is a parent (len > CHILD_CHUNK_SIZE)",
                   all_parents))

    # ...and each question returns between 1 and K parents. ParentDocumentRetriever
    # dedupes: several top children can belong to the same parent, so the count
    # is at most K but not always exactly K.
    checks.append((f"each question returns 1..{K} parents (deduped)",
                   all(1 <= len(r["parents"]) <= K for r in exp["results"])))

    # Content: Q1610 "Who founded Montevideo?" must retrieve the parent that
    # says the Spanish founded Montevideo.
    q1610_top = exp["results"][1]["parents"][0].page_content.lower()
    checks.append(("Q1610 top-1 parent names the Spanish founder of Montevideo",
                   "spanish" in q1610_top))

    # Q1606 "Is Uruguay's capital Montevideo?" must retrieve a Uruguay parent
    # that mentions Montevideo.
    q1606_top = exp["results"][0]["parents"][0].page_content.lower()
    checks.append(("Q1606 top-1 parent mentions Montevideo", "montevideo" in q1606_top))

    # Linkage: the matched child is literally a fragment of the parent the
    # docstore says it came from (deterministic — the splitter concatenates
    # substrings, it never rewrites text).
    linked = all(
        r["child_parent"] is not None
        and r["child"].page_content.strip() in r["child_parent"].page_content
        for r in exp["results"]
    )
    checks.append(("matched child text is contained in its docstore parent",
                   linked))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
