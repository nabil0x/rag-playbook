# Project 24 — Vector Index Internals

> **Goal:** Stop treating the vector database as a magic black box — build and
> benchmark the four index types FAISS actually ships: flat (exact), IVF, HNSW,
> and PQ — and learn what "approximate" buys you in speed and memory.

## Why

By Project 07 you could compare Chroma vs FAISS vs Qdrant by swapping one line.
But "FAISS" is not one algorithm — it is a toolbox of approximate-nearest-
neighbor (ANN) indexes with wildly different tradeoffs. Plain flat search
compares your query against *every* vector: exact, but linear, and unusable at
a million rows. IVF clusters the space and only searches the nearest clusters.
HNSW builds a navigable graph of neighbors. PQ compresses every vector into
short codes so the whole corpus fits in RAM. Each one trades a little recall
for a lot of latency and memory — and the right choice depends on how many
vectors you have, how fast you need answers, and how much RAM you can afford.
This project makes those tradeoffs measurable in your own hands.

## Learn

- Exact search (flat L2) as the correctness baseline — everything else is an approximation of it
- IVF: `nlist` clusters at index time, `nprobe` clusters visited at query time; recall rises with `nprobe`, latency does too
- HNSW: navigable small-world graph; `M` (connections per node) and `efSearch` (candidate list) control recall vs speed
- PQ: split vectors into `M` sub-vectors, quantize each to `nbits` — 8x-64x memory compression at a recall cost; the `train` step happens once, search uses the codes
- The recall@k vs latency vs memory triangle — measure all three, never just one
- OPQ (rotation before PQ) recovers much of the recall PQ loses

## Execute

1. **Setup** — `pip install faiss-cpu sentence-transformers` (or `fastembed`)
2. **Read** — `src/vectordb/faiss_index.py` — the `FaissIndex` stub: index types, `build`, `search`, `benchmark`
3. **Implement** — fill in the four index constructors and the `benchmark` loop that returns `recall@k` and latency per index type
4. **Run** — `python vectordb/faiss_index.py` for the smoke test; then embed ~300 chunks from `Data/local-docs/` and benchmark
5. **Measure** — recall@10 vs mean latency for flat / IVF / HNSW / PQ on the same query set; report the memory footprint of each index
6. **Acceptance criteria** — IVF `nprobe=10` gets >= 90% of flat recall; HNSW `efSearch=64` beats IVF latency at >= 95% recall; PQ at `M=8, nbits=8` uses a fraction of flat memory; you can explain *why* each number moved

## Stretch

- Add OPQ preprocessing (`faiss.OPQMatrix` + PQ) and show the recall recovery
- Sweep `nprobe` / `efSearch` and plot the recall-latency curve
- Scale to 1M random 128-d vectors and watch flat search fall over

## Article

- [ ] `24-vector-index-internals.md`

## Code

- `src/vectordb/faiss_index.py` — `FaissIndex(index_type, dim)` — build/search/benchmark for flat, IVF, HNSW, PQ

## Notebook

`NoteBooks/Project-24-Vector-Index-Internals/01-index-internals-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Project-24-Vector-Index-Internals/01-index-internals-spec.py NoteBooks/Project-24-Vector-Index-Internals/01-index-internals.ipynb
```
