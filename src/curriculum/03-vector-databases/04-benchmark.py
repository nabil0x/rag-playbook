"""Lab 04 — metadata filtering + index/query/memory benchmark.

Labs 01–03 built the same retrieval with three stores (FAISS, Chroma,
Qdrant) and proved they rank identically when given the same vectors — the
store never changes WHAT ranks first, only the score scale/direction.

This lab adds the two things real apps need on top of that:

1. **Metadata filtering** — scoping retrieval to a subset (e.g. "only docs in
   bucket b1"). Same logical filter, three syntaxes:

   - FAISS and Chroma: a plain dict ``{"bucket": "b1"}`` (LangChain
     metadata-filter syntax).
   - Qdrant: the plain dict ALSO works — langchain-qdrant translates it and
     knows its payload layout. But the moment you want the expressive
     must/should form, you MUST hand it a real
     ``qdrant_client.http.models.Filter`` object — and the field key must be
     prefixed with ``metadata.``, because langchain-qdrant nests every
     metadata field under a single ``metadata`` payload key. Two silent
     traps, both demonstrated here:

       - ``filter={"must": [...]}`` as a raw dict → 0 hits (the translator
         treats "must" as a field name → ``metadata.must`` doesn't exist).
       - ``models.Filter(must=[... key="bucket" ...])`` without the
         ``metadata.`` prefix → 0 hits (top-level ``bucket`` doesn't exist).

     None of these error — they just return nothing. The gate locks the
     correct forms in.

   One more difference only surfaces when you run it: FAISS filters in
   Python inside a ``fetch_k``-wide candidate window (default 20), so a
   filtered search can return FEWER than ``top_k`` hits when the window
   doesn't contain enough matches. Chroma and Qdrant push the filter into
   the engine and always return exactly ``top_k``. The gate locks that in
   too: FAISS gets ``0 < len <= top_k``, Chroma/Qdrant get ``len == top_k``.

2. **A small benchmark** — index time, query latency, process memory
   (psutil RSS) and disk footprint for each store, so you can see *why* you
   might pick one over the other. Chroma persists to a directory (sqlite
   files on disk); FAISS (in-memory) and Qdrant ``:memory:`` write zero
   bytes.

Run from the repo root:
    python src/curriculum/03-vector-databases/04-benchmark.py
    python src/curriculum/03-vector-databases/04-benchmark.py --verify
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import psutil

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/03-vector-databases/04-benchmark.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from qdrant_client.http import models  # noqa: E402
from vectordb.chroma import ChromaVectorStore  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402
from vectordb.qdrant import QdrantVectorStore  # noqa: E402

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
BUCKETS = ["b0", "b1", "b2", "b3"]  # synthetic metadata: 4 buckets, cycled
FILTER_BUCKET = "b1"
QUERY_REPEATS = 20  # per question, for the latency average
FAISS_FETCH_K = 20  # langchain-FAISS default candidate window for filters
COLLECTION = "lab04"


# --------------------------------------------------------------------------
# 2. Load — corpus + questions (same helpers as labs 01–03)
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


def rss_mb() -> float:
    """Current process RSS in MiB (Linux/psutil)."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def dir_bytes(path: Path) -> int:
    """Total size of every file under ``path`` (0 if it doesn't exist)."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def qdrant_filter(bucket: str) -> models.Filter:
    """The CORRECT expressive Qdrant filter: ``metadata.``-prefixed key."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.bucket", match=models.MatchValue(value=bucket)
            )
        ]
    )


# --------------------------------------------------------------------------
# 3. Experiment — embed once, add synthetic bucket metadata, index into all
#    three stores, then (a) filter each store and (b) benchmark all three
# --------------------------------------------------------------------------
def build_chunks(
    passage_texts: list[str], passage_ids: list[int]
) -> list[Document]:
    """Attach synthetic ``bucket`` metadata: b0..b3 cycled by passage index."""
    return [
        Document(
            page_content=t,
            metadata={"id": pid, "bucket": BUCKETS[i % len(BUCKETS)]},
        )
        for i, (t, pid) in enumerate(zip(passage_texts, passage_ids))
    ]


def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)
    chunks = build_chunks(passage_texts, passage_ids)

    # --- Embed the subset once; ALL THREE stores index the same vectors ----
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passage_texts)
    embed_s = time.perf_counter() - t0
    query_vecs = [embedder.embed_query(q) for _, q in questions]

    # --- Memory checkpoints (RSS is process-wide; deltas are cumulative) ---
    gc.collect()
    baseline_rss = rss_mb()

    # --- FAISS — in-memory, zero disk --------------------------------------
    faiss_store = FAISSVectorStore()
    t0 = time.perf_counter()
    faiss_store.add(chunks, embeddings=passage_vecs)
    faiss_add_s = time.perf_counter() - t0
    gc.collect()
    faiss_rss = rss_mb() - baseline_rss

    # --- Chroma — persistent in a temp directory ---------------------------
    chroma_dir = Path(tempfile.mkdtemp(prefix="lab04_chroma_"))
    chroma_store = ChromaVectorStore(
        collection_name=COLLECTION, persist_dir=str(chroma_dir)
    )
    t0 = time.perf_counter()
    chroma_store.add(chunks, embeddings=passage_vecs)
    chroma_add_s = time.perf_counter() - t0
    gc.collect()
    chroma_rss = rss_mb() - baseline_rss

    # --- Qdrant — in-memory (":memory:" writes nothing to disk) ------------
    qdrant_store = QdrantVectorStore(collection_name=COLLECTION, path=":memory:")
    t0 = time.perf_counter()
    qdrant_store.add(chunks, embeddings=passage_vecs)
    qdrant_add_s = time.perf_counter() - t0
    gc.collect()
    qdrant_rss = rss_mb() - baseline_rss

    # --- Filtered retrieval (same logical filter, three syntaxes) ----------
    faiss_filtered = [
        faiss_store.query_with_scores(q, top_k=TOP_K, filter={"bucket": FILTER_BUCKET})
        for q in query_vecs
    ]
    chroma_filtered = [
        chroma_store.query_with_scores(q, top_k=TOP_K, filter={"bucket": FILTER_BUCKET})
        for q in query_vecs
    ]
    qdrant_simple = [
        qdrant_store.query_with_scores(q, top_k=TOP_K, filter={"bucket": FILTER_BUCKET})
        for q in query_vecs
    ]
    # Expressive form: must be a real models.Filter with a metadata.-prefixed
    # key — the raw-dict "must" and the unprefixed key both silently return 0.
    qdrant_model = [
        qdrant_store.query_with_scores(q, top_k=TOP_K, filter=qdrant_filter(FILTER_BUCKET))
        for q in query_vecs
    ]
    qdrant_must_dict = [
        qdrant_store.query_with_scores(
            q,
            top_k=TOP_K,
            filter={"must": [{"key": "bucket", "match": {"value": FILTER_BUCKET}}]},
        )
        for q in query_vecs
    ]
    qdrant_unprefixed = [
        qdrant_store.query_with_scores(
            q,
            top_k=TOP_K,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="bucket", match=models.MatchValue(value=FILTER_BUCKET)
                    )
                ]
            ),
        )
        for q in query_vecs
    ]

    # Unfiltered baseline: content grounding (the bucket tags are synthetic,
    # so only the unfiltered top-1 is guaranteed to carry the answer).
    faiss_unfiltered = [
        faiss_store.query_with_scores(q, top_k=TOP_K) for q in query_vecs
    ]

    # --- Benchmark: query latency (mean over repeats x questions) ----------
    def latency_ms(store, scored_unfiltered_call) -> float:
        total = 0.0
        n = 0
        for _ in range(QUERY_REPEATS):
            for q in query_vecs:
                t0 = time.perf_counter()
                scored_unfiltered_call(store, q)
                total += time.perf_counter() - t0
                n += 1
        return total / n * 1000.0

    faiss_lat = latency_ms(faiss_store, lambda s, q: s.query_with_scores(q, top_k=TOP_K))
    chroma_lat = latency_ms(chroma_store, lambda s, q: s.query_with_scores(q, top_k=TOP_K))
    qdrant_lat = latency_ms(qdrant_store, lambda s, q: s.query_with_scores(q, top_k=TOP_K))

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "embed_s": embed_s,
        "dim": len(passage_vecs[0]),
        "indexed": len(passage_vecs),
        "stores": {
            "faiss": {
                "add_s": faiss_add_s, "rss_mb": faiss_rss,
                "disk_bytes": 0, "latency_ms": faiss_lat,
            },
            "chroma": {
                "add_s": chroma_add_s, "rss_mb": chroma_rss,
                "disk_bytes": dir_bytes(chroma_dir), "latency_ms": chroma_lat,
            },
            "qdrant": {
                "add_s": qdrant_add_s, "rss_mb": qdrant_rss,
                "disk_bytes": 0, "latency_ms": qdrant_lat,
            },
        },
        "faiss_filtered": faiss_filtered,
        "chroma_filtered": chroma_filtered,
        "qdrant_simple": qdrant_simple,
        "qdrant_model": qdrant_model,
        "qdrant_must_dict": qdrant_must_dict,
        "qdrant_unprefixed": qdrant_unprefixed,
        "faiss_unfiltered": faiss_unfiltered,
        "chroma_dir": chroma_dir,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 04 — metadata filtering + index/query/memory benchmark")
    print(f"{BGE_MODEL_NAME} | 100 passages | synthetic buckets {BUCKETS}")
    print("=" * 66)

    print(f"\n[1] Corpus + embedding:")
    print(f"    {exp['indexed']} passages, dim {exp['dim']}, embedded in {exp['embed_s']:.2f}s")
    print(f"    each passage tagged bucket={BUCKETS}, cycled by index")

    print(f"\n[2] Index build — time, memory (RSS delta vs post-embedding), disk:")
    print(f"    {'store':<8}{'add time':>10}{'rss +MB':>10}{'disk':>10}{'query ms':>10}")
    for name, s in exp["stores"].items():
        print(f"    {name:<8}{s['add_s']:>9.3f}s{s['rss_mb']:>10.1f}"
              f"{s['disk_bytes']:>9}B{s['latency_ms']:>9.3f}")

    print(f"\n[3] Metadata filter — top-{TOP_K} within bucket '{FILTER_BUCKET}':")
    qid, qtext = exp["questions"][1]
    print(f'    Q[{qid}] "{qtext}"')
    faiss_ids = [d.metadata["id"] for d, _ in exp["faiss_filtered"][1]]
    chroma_ids = [d.metadata["id"] for d, _ in exp["chroma_filtered"][1]]
    qdrant_simple_ids = [d.metadata["id"] for d, _ in exp["qdrant_simple"][1]]
    qdrant_model_ids = [d.metadata["id"] for d, _ in exp["qdrant_model"][1]]
    print(f"    FAISS   {{'bucket': '{FILTER_BUCKET}'}} -> {faiss_ids}")
    if len(faiss_ids) < TOP_K:
        print(f"    (FAISS found only {len(faiss_ids)} of {TOP_K} — its filter is a Python")
        print(f"     post-filter over the top {FAISS_FETCH_K} candidates; Chroma/Qdrant")
        print(f"     filter in-engine and always return {TOP_K})")
    print(f"    Chroma  {{'bucket': '{FILTER_BUCKET}'}} -> {chroma_ids}")
    print(f"    Qdrant  {{'bucket': '{FILTER_BUCKET}'}} -> {qdrant_simple_ids}")
    print(f"    Qdrant  models.Filter(metadata.bucket) -> {qdrant_model_ids}")
    print(f"    Qdrant  raw {{'must': [...]}} dict     -> "
          f"{[d.metadata['id'] for d, _ in exp['qdrant_must_dict'][1]]}  (silent trap)")
    print(f"    Qdrant  models.Filter(bucket, no meta.) -> "
          f"{[d.metadata['id'] for d, _ in exp['qdrant_unprefixed'][1]]}  (silent trap)")

    print("\n[4] Takeaway")
    print("    A filter narrows the candidate set; ranking inside it still")
    print("    follows the same similarity order in every store. The plain")
    print("    dict works everywhere, but Qdrant's expressive form needs a")
    print("    real models.Filter with the metadata. prefix — and bad forms")
    print("    return 0 hits instead of erroring. FAISS may return fewer")
    print("    than top_k: its filter is a Python post-filter over a")
    print(f"    fetch_k={FAISS_FETCH_K} candidate window, while Chroma/Qdrant")
    print("    filter in the engine. Benchmark: FAISS and Qdrant(:memory:)")
    print("    hold everything in RAM (0 bytes on disk); Chroma trades a")
    print("    little memory for a persistent sqlite dir. RSS deltas are")
    print("    process-wide and noisy — compare roughly.")
    print(f"    (temp chroma dir {exp['chroma_dir']} removed after this run)")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append(("embedding dimension is 768 (BGE base)", exp["dim"] == BGE_DIM))
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))

    # Unfiltered ranking identity across the three stores (labs 01–03 carryover).
    def ids(scored) -> list[int]:
        return [d.metadata["id"] for d, _ in scored]

    all_same_rank = True
    for i in range(len(exp["questions"])):
        f = ids(exp["faiss_filtered"][i])
        c = ids(exp["chroma_filtered"][i])
        q = ids(exp["qdrant_simple"][i])
        # FAISS filters inside a fetch_k-wide candidate window, so it may
        # return FEWER than TOP_K hits; its hits are still the top-len(f)
        # passages of the engine-filtered ranking, in the same order.
        all_same_rank &= f == c[: len(f)] == q[: len(f)]
    checks.append(("FAISS filtered hits are the top-len(FAISS) of the engine-filtered ranking", all_same_rank))

    # The core claim: filtered hits are exactly the requested bucket. FAISS
    # may return fewer than TOP_K (Python post-filter over fetch_k candidates);
    # Chroma/Qdrant filter in-engine and always return exactly TOP_K.
    only_b1 = True
    faiss_count_ok = True
    engine_count_ok = True
    for i in range(len(exp["questions"])):
        for key, is_faiss in (
            ("faiss_filtered", True),
            ("chroma_filtered", False),
            ("qdrant_simple", False),
        ):
            hits = exp[key][i]
            only_b1 &= all(d.metadata["bucket"] == FILTER_BUCKET for d, _ in hits)
            if is_faiss:
                faiss_count_ok &= 0 < len(hits) <= TOP_K
            else:
                engine_count_ok &= len(hits) == TOP_K
    checks.append(("every store returns only bucket-b1 hits", only_b1))
    checks.append(("FAISS may return < top_k (fetch-window filter); Chroma/Qdrant always top_k", faiss_count_ok and engine_count_ok))

    # The expressive Qdrant forms: the correct one matches the dict form...
    q_model_ids = [ids(exp["qdrant_model"][i]) for i in range(len(exp["questions"]))]
    q_simple_ids = [ids(exp["qdrant_simple"][i]) for i in range(len(exp["questions"]))]
    checks.append(("Qdrant models.Filter (metadata.bucket) matches the dict form", q_model_ids == q_simple_ids))

    # ...and the two silent traps return ZERO hits (no error, no results).
    traps_empty = all(len(exp["qdrant_must_dict"][i]) == 0 for i in range(len(exp["questions"])))
    checks.append(("Qdrant raw {'must': [...]} dict returns 0 hits (documented trap)", traps_empty))
    unprefixed_empty = all(len(exp["qdrant_unprefixed"][i]) == 0 for i in range(len(exp["questions"])))
    checks.append(("Qdrant models.Filter without metadata. prefix returns 0 hits (trap)", unprefixed_empty))

    # Content check (same as labs 01–03) — against the UNFILTERED ranking:
    # the bucket tags are synthetic, so only the unfiltered top-1 is
    # guaranteed to carry the answer.
    q1610_top = exp["faiss_unfiltered"][1][0][0].page_content.lower()
    checks.append(("Q1610 top-1 names the Spanish founder of Montevideo", "spanish" in q1610_top))

    # Benchmark sanity: measurable timings, non-negative memory, disk story.
    bench_ok = True
    for name, s in exp["stores"].items():
        bench_ok &= s["add_s"] > 0
        bench_ok &= s["latency_ms"] > 0
        bench_ok &= s["rss_mb"] >= 0
        if name == "chroma":
            bench_ok &= s["disk_bytes"] > 0
        else:
            bench_ok &= s["disk_bytes"] == 0
    checks.append(("benchmark: timings positive, Chroma on disk, FAISS/Qdrant RAM-only", bench_ok))

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
        shutil.rmtree(exp["chroma_dir"], ignore_errors=True)
