"""Lab 02 — Chroma: the persistent vector store.

Lab 01's FAISS index is pure RAM: build it, query it, and when the process
exits the index is gone. Chroma takes the opposite trade: everything is
written to a directory on disk (``persist_dir``), so the store survives the
process that created it.

This lab builds a Chroma store over the same deterministic
``Data/corpus/rag-mini-wikipedia`` subset as lab 01 and proves persistence the
hard way — the store is closed, reopened **in a brand-new Python process**, and
answered the same question with the same passage and the same score.

Three things to notice:

* ON-DISK LAYOUT — a persistent Chroma collection is a directory containing
  ``chroma.sqlite3`` (metadata + collection registry) plus a per-collection
  folder (the HNSW index). The lab prints the files and their sizes so
  "persistent" is concrete, not a claim.
* SCORE CONVENTION — Chroma's default collection uses the l2 space, and this
  version of langchain-chroma returns the raw (squared) L2 distance — the
  same numbers FAISS reports (lab 01). The lab queries FAISS and Chroma with
  the *same precomputed vectors* and prints the scores side by side: they
  match to the last few digits. Qdrant (lab 03) is the odd one out with its
  cosine score.
* REOPENING — the store is re-opened with a ``load()`` call, not by re-adding
  documents. Re-adding would UPSERT into the existing collection and quietly
  duplicate every passage — the classic persistent-store footgun the
  component docstring warns about.

Persistence is a spectrum, not a checkbox: FAISS trades it away entirely,
Chroma pays a little disk I/O for it, and Qdrant (lab 03) offers both a
persistent mode and an in-memory one on the same API.

Run from the repo root:
    python src/curriculum/03-vector-databases/02-chroma-persistent.py
    python src/curriculum/03-vector-databases/02-chroma-persistent.py --verify
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/03-vector-databases/02-chroma-persistent.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from vectordb.chroma import ChromaVectorStore  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus
QUESTION_IDS = [1606, 1610]  # real questions; answers live inside the subset
TOP_K = 3
PREVIEW = 62
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768
SCORE_TOL = 1e-3  # FAISS vs Chroma are separate float32 code paths; allow last-digit jitter
COLLECTION = "lab02"


# --------------------------------------------------------------------------
# 2. Load — corpus + questions (same helpers as lab 01)
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> tuple[list[str], list[int]]:
    """Return (passage_texts, passage_ids) for the first ``n`` passages."""
    df = pd.read_parquet(path)
    subset = df.head(n)
    return subset["passage"].tolist(), subset.index.tolist()


def load_questions(path: Path, ids: list[int]) -> list[tuple[int, str]]:
    """Return [(question_id, question_text)] for the requested test rows."""
    df = pd.read_parquet(path)
    rows = df.loc[ids]
    return [(int(idx), row["question"]) for idx, row in rows.iterrows()]


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --------------------------------------------------------------------------
# 3. Experiment — embed once, feed the same vectors to FAISS and Chroma,
#    then prove Chroma persists across a process boundary
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)
    chunks = [
        Document(page_content=t, metadata={"id": pid})
        for t, pid in zip(passage_texts, passage_ids)
    ]

    # --- Embed the subset once; BOTH stores index the same vectors ----------
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passage_texts)
    embed_s = time.perf_counter() - t0
    query_vecs = [embedder.embed_query(q) for _, q in questions]

    # --- FAISS (in-memory, lab 01) — score cross-check baseline ------------
    faiss_store = FAISSVectorStore()
    faiss_store.add(chunks, embeddings=passage_vecs)
    faiss_scored = [faiss_store.query_with_scores(q, top_k=TOP_K) for q in query_vecs]

    # --- Chroma (persistent) — added once, then reopened twice --------------
    tmp = tempfile.TemporaryDirectory(prefix="lab02_chroma_")
    persist_dir = tmp.name

    chroma_store = ChromaVectorStore(collection_name=COLLECTION, persist_dir=persist_dir)
    t0 = time.perf_counter()
    chroma_store.add(chunks, embeddings=passage_vecs)
    add_s = time.perf_counter() - t0
    chroma_scored = [chroma_store.query_with_scores(q, top_k=TOP_K) for q in query_vecs]

    # On-disk artifact listing (proves the data lives outside RAM).
    disk_files = sorted(os.listdir(persist_dir))
    disk_bytes = {
        name: os.path.getsize(os.path.join(persist_dir, name)) for name in disk_files
    }

    # In-process reopen: drop the first instance, open a fresh one, no re-add.
    del chroma_store
    gc.collect()
    reopened = ChromaVectorStore(collection_name=COLLECTION, persist_dir=persist_dir)
    reopened.load()
    reopened_scored = [reopened.query_with_scores(q, top_k=TOP_K) for q in query_vecs]

    # Cross-process reopen: a brand-new Python interpreter reads the same dir.
    qvec_file = os.path.join(persist_dir, "query_vector.json")
    with open(qvec_file, "w") as fh:
        json.dump(query_vecs[1], fh)  # Q1610 "Who founded Montevideo?"
    sub_script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})\n"
        "from vectordb.chroma import ChromaVectorStore\n"
        f"q = json.load(open({qvec_file!r}))\n"
        f"s = ChromaVectorStore(collection_name={COLLECTION!r}, persist_dir={persist_dir!r})\n"
        "s.load()\n"
        "doc, score = s.query_with_scores(q, 1)[0]\n"
        'print(f"{doc.metadata.get(chr(105)+chr(100), chr(63))}|{score:.4f}")\n'
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", sub_script], capture_output=True, text=True, timeout=120
    )
    reopen_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"reopen subprocess failed: {proc.stderr[-500:]}")
    sub_pid_str, sub_score_str = proc.stdout.strip().split("|")

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "embed_s": embed_s,
        "add_s": add_s,
        "reopen_s": reopen_s,
        "faiss_scored": faiss_scored,
        "chroma_scored": chroma_scored,
        "reopened_scored": reopened_scored,
        "disk_files": disk_files,
        "disk_bytes": disk_bytes,
        "sub_pid": int(sub_pid_str),
        "sub_score": float(sub_score_str),
        "dim": len(passage_vecs[0]),
        "indexed": len(passage_vecs),
        "persist_dir": persist_dir,
        "tmp": tmp,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 02 — Chroma: the persistent vector store")
    print(f"{BGE_MODEL_NAME} | l2 space | persist_dir on disk")
    print("=" * 66)

    print(f"\n[1] Corpus + embedding:")
    print(f"    {exp['indexed']} passages, dim {exp['dim']}, embedded in {exp['embed_s']:.2f}s")
    print(f"    same vectors indexed into FAISS (lab 01 baseline) AND Chroma")

    print(f"\n[2] Chroma index build:")
    print(f"    {exp['indexed']} passages added in {exp['add_s']:.3f}s")
    print("    on-disk layout of persist_dir:")
    for name, size in exp["disk_bytes"].items():
        print(f"      {name:<45} {size:>9,} B")
    print(f"      {'total':<45} {sum(exp['disk_bytes'].values()):>9,} B")

    print(f"\n[3] Top-{TOP_K} per question — FAISS vs Chroma (squared L2, LOWER = better):")
    for i, (qid, qtext) in enumerate(exp["questions"]):
        print(f'\n    Q[{qid}] "{qtext}"')
        for (fdoc, fscore), (cdoc, cscore) in zip(
            exp["faiss_scored"][i], exp["chroma_scored"][i]
        ):
            match = "SAME" if fdoc.metadata["id"] == cdoc.metadata["id"] else "DIFF"
            print(f"      faiss {fscore:8.4f}  chroma {cscore:8.4f}  "
                  f"[passage {cdoc.metadata['id']}] {preview(cdoc.page_content)}  {match}")

    print(f"\n[4] Persistence — close, reopen, ask again:")
    print("    same top-1 after in-process reopen:")
    for (qid, _), hits in zip(exp["questions"], exp["reopened_scored"]):
        doc, score = hits[0]
        print(f"      Q[{qid}] -> [passage {doc.metadata['id']}] score {score:.4f}")
    print(f"    brand-new Python process reopened the same dir in {exp['reopen_s']:.2f}s:")
    print(f"      Q[{exp['questions'][1][0]}] -> [passage {exp['sub_pid']}] "
          f"score {exp['sub_score']:.4f}")
    print("    the first Chroma instance was deleted; the data survived because")
    print("    it lives in chroma.sqlite3 + the HNSW files, not in RAM.")

    print("\n[5] Takeaway")
    print("    Chroma swaps lab 01's 'index vanishes at exit' for a directory")
    print("    on disk. The price is visible in [2]: building the index touches")
    print("    sqlite and the HNSW files, so add() is slower than FAISS's.")
    print("    And reopening must call load(), never add() again — re-adding")
    print("    upserts and duplicates every passage in the existing collection.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append(("embedding dimension is 768 (BGE base)", exp["dim"] == BGE_DIM))
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))

    # Chroma's score convention matches FAISS's (both squared L2, lower = better).
    all_same_rank = True
    all_scores_close = True
    for fhits, chits in zip(exp["faiss_scored"], exp["chroma_scored"]):
        for (fdoc, fscore), (cdoc, cscore) in zip(fhits, chits):
            all_same_rank &= fdoc.metadata["id"] == cdoc.metadata["id"]
            all_scores_close &= abs(fscore - cscore) < SCORE_TOL
    checks.append(("FAISS and Chroma return the same passages in the same order", all_same_rank))
    checks.append(("FAISS and Chroma scores agree within tolerance", all_scores_close))

    # The store actually wrote to disk.
    checks.append(("persist_dir contains chroma.sqlite3", "chroma.sqlite3" in exp["disk_files"]))

    # Content checks (same as lab 01 — the top-1 answer passages).
    q1610_top = exp["chroma_scored"][1][0][0].page_content.lower()
    checks.append(("Q1610 top-1 names the Spanish founder of Montevideo", "spanish" in q1610_top))
    q1606_top = exp["chroma_scored"][0][0][0].page_content.lower()
    checks.append(("Q1606 top-1 mentions Montevideo", "montevideo" in q1606_top))

    # In-process reopen reproduces the original ranking exactly.
    reopen_matches = all(
        o[0][0].metadata["id"] == r[0][0].metadata["id"]
        for o, r in zip(exp["chroma_scored"], exp["reopened_scored"])
    )
    checks.append(("in-process reopen reproduces every top-1 passage id", reopen_matches))

    # Cross-process reopen answers Q1610 with the same passage and score.
    orig = exp["chroma_scored"][1][0]
    checks.append(("new-process reopen matches the original Q1610 top-1 id",
                   exp["sub_pid"] == orig[0].metadata["id"]))
    checks.append(("new-process reopen matches the original Q1610 score",
                   abs(exp["sub_score"] - orig[1]) < SCORE_TOL))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    try:
        if "--verify" in sys.argv:
            sys.exit(verify_gate(exp))
        print_demo(exp)
    finally:
        exp["tmp"].cleanup()  # remove the persistent dir after the demo/gate
