"""Lab 03 — Position bias: does a reranker care where the gold sits?

The lost-in-the-middle dataset was built to show LLMs ignoring passages in
the middle of long contexts. The same failure appears one level earlier, in
retrieval: a candidate list arrives in SOME order, the pipeline truncates it
to top-k, and a relevant passage buried at position 9 of 10 is silently
dropped — the pipeline has position bias.

This lab measures that bias directly, using the NQ-open 10-passage variant:
every question comes with 10 passages, exactly one is gold (``isgold``), and
the gold is placed at a controlled position — 0, 4, or 9 — in three matched
files (2,655 questions each). We take the first 25 questions per bucket and
compare two ways of producing a top-3:

* KEEP-FIRST — naive truncation: take the 3 passages in the given order.
  Recovery should track the gold's position: ~100% at position 0, 0% at
  positions 4 and 9.
* RERANK — score all 10 passages with the cross-encoder, keep the top-3.
  The reranker reads only CONTENT, never position, so recovery should be
  identical no matter where the gold sat.

No embeddings are needed here: the cross-encoder scores the 10 given
passages directly. That is the point — reranking makes retrieval
order-agnostic.

Run from the repo root:
    python src/curriculum/06-re-ranking/03-position-bias.py
    python src/curriculum/06-re-ranking/03-position-bias.py --verify
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/06-re-ranking/03-position-bias.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langchain_core.documents import Document  # noqa: E402
from tools.reranker import CrossEncoderReranker  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
LITM_DIR = Path("Data/corpus/lost-in-the-middle/10_total_documents")
GOLD_POSITIONS = (0, 4, 9)  # the three matched files differ only here
N_PER_BUCKET = 25  # questions per position bucket (deterministic head)
TOP_K = 3  # depth both systems are evaluated at


def bucket_path(pos: int) -> Path:
    return LITM_DIR / f"nq-open-10_total_documents_gold_at_{pos}.jsonl.gz"


# --------------------------------------------------------------------------
# 2. Load — questions with 10 passages, gold at a controlled position
# --------------------------------------------------------------------------
def load_bucket(path: Path, n: int) -> list[dict]:
    """Return the first ``n`` questions from one position bucket.

    Each item is ``{"question", "gold_idx", "passages": [text, ...]}`` where
    ``passages[gold_idx]`` is the gold passage and ``gold_idx == pos`` is
    asserted against the bucket's file name.
    """
    out: list[dict] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if len(out) >= n:
                break
            d = json.loads(line)
            gold_idx = next(i for i, c in enumerate(d["ctxs"]) if c["isgold"])
            out.append(
                {
                    "question": d["question"],
                    "gold_idx": gold_idx,
                    "passages": [c["text"] for c in d["ctxs"]],
                }
            )
    return out


# --------------------------------------------------------------------------
# 3. Experiment — keep-first vs rerank, per position bucket
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    reranker = CrossEncoderReranker()
    t0 = time.perf_counter()
    buckets = []
    for pos in GOLD_POSITIONS:
        questions = load_bucket(bucket_path(pos), N_PER_BUCKET)
        for q in questions:
            assert q["gold_idx"] == pos  # the file says where gold sits

        keep_first_hits = 0
        rerank_hits = 0
        rank_sum = 0
        for q in questions:
            gold = q["gold_idx"]
            keep_first_hits += 1 if gold < TOP_K else 0

            docs = [
                Document(page_content=text, metadata={"idx": i})
                for i, text in enumerate(q["passages"])
            ]
            ranked = reranker.rerank(q["question"], docs, top_k=10)
            gold_rank = next(
                i for i, d in enumerate(ranked, 1) if d.metadata["idx"] == gold
            )
            rerank_hits += 1 if gold_rank <= TOP_K else 0
            rank_sum += gold_rank

        buckets.append(
            {
                "pos": pos,
                "n": len(questions),
                "keep_first_rate": keep_first_hits / len(questions),
                "rerank_rate": rerank_hits / len(questions),
                "mean_gold_rank": rank_sum / len(questions),
            }
        )
    score_s = time.perf_counter() - t0
    return {"buckets": buckets, "score_s": score_s}


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 03 — Position bias: does a reranker care where the gold sits?")
    print(f"lost-in-the-middle NQ-open 10-passage, top-{TOP_K} "
          f"(cross-encoder/ms-marco-MiniLM-L-6-v2)")
    print("=" * 66)

    print(f"\n[1] Data (deterministic head, {N_PER_BUCKET} questions per bucket):")
    print("    gold passage placed at position 0 / 4 / 9 of a 10-passage list;")
    print("    every question has exactly one gold (isgold) passage.")

    print(f"\n[2] Gold-in-top-{TOP_K} recovery rate, by gold position:")
    print(f"    {'gold at':<12}{'keep-first':>14}{'rerank':>12}{'mean rank':>14}")
    for b in exp["buckets"]:
        print(f"    position {b['pos']:<5}"
              f"{b['keep_first_rate']:>14.2f}"
              f"{b['rerank_rate']:>12.2f}"
              f"{b['mean_gold_rank']:>14.2f}")

    print(f"\n[3] Takeaway")
    print("    KEEP-FIRST tracks the position: gold at 0 is always in the")
    print("    top-3, gold at 4 or 9 is always dropped — a hard position")
    print("    bias. RERANK is flat across buckets: the cross-encoder scores")
    print("    (query, passage) pairs and never sees the order, so a buried")
    print("    gold is recovered exactly as often as a leading one.")
    print("    That is why production pipelines rerank the candidate list")
    print("    before truncating it: truncation is where relevant passages")
    print("    get lost, and reranking makes truncation order-agnostic.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    rates = {b["pos"]: b for b in exp["buckets"]}
    checks.append((f"one bucket per position {GOLD_POSITIONS}",
                   set(rates) == set(GOLD_POSITIONS)))
    for pos in GOLD_POSITIONS:
        b = rates[pos]
        checks.append((f"bucket {pos}: {N_PER_BUCKET} questions",
                       b["n"] == N_PER_BUCKET))

    checks.append(("keep-first is position-biased (1.0 / 0.0 / 0.0)",
                   rates[0]["keep_first_rate"] == 1.0
                   and rates[4]["keep_first_rate"] == 0.0
                   and rates[9]["keep_first_rate"] == 0.0))
    checks.append(("rerank recovers buried gold (rate > 0.5 at pos 4 AND 9)",
                   rates[4]["rerank_rate"] > 0.5
                   and rates[9]["rerank_rate"] > 0.5))
    rerank_rates = [b["rerank_rate"] for b in exp["buckets"]]
    checks.append(("rerank is position-agnostic (rates agree within 0.01)",
                   max(rerank_rates) - min(rerank_rates) < 0.01))
    gold_ranks = [b["mean_gold_rank"] for b in exp["buckets"]]
    checks.append(("mean gold rank after rerank is consistent across buckets",
                   max(gold_ranks) - min(gold_ranks) < 0.01))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
