"""Lab 03 — Tree-vs-flat retrieval eval (RAPTOR step 3).

Labs 01-02 built the RAPTOR tree. This lab asks the question that decides
whether the tree is worth its LLM calls: does *tree retrieval* find the
right chunks, and how does it compare with the flat baseline it replaces?

* **Flat baseline** — embed the question and cosine-rank every one of the
  N=24 passages directly; take the top 4. This is standard vector search:
  O(N) similarity computations, no summaries.
* **Tree retrieval** — ``tools/raptor.retrieve`` walks the tree instead:
  at each level it embeds the question, keeps the ``top_k`` most similar
  children, descends into them, and finally returns the best *leaf* chunks
  (``collapse=True``) or the chosen summary nodes (``collapse=False``).
  Each step only compares against a handful of summaries, so a query that
  matches a topic broad enough to be summarized can jump straight to the
  right cluster without scanning every chunk.

We evaluate both methods on 3 yes/no questions from the corpus test set and
report, per question, the top passage each method surfaces and whether that
passage contains the gold answer's words (an ``answer_contains``-style
normalized substring check). The hit rates are the lesson — they are
printed but not hard-required by the verification gate; the gate checks
only that both methods return well-formed retrievals.

Run from the repo root:
    python curriculum/09-raptor/03-tree-vs-flat.py
    python curriculum/09-raptor/03-tree-vs-flat.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/09-raptor/03-tree-vs-flat.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402
from tools.raptor import build_tree, retrieve  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 24  # corpus both methods search; tree build costs ~1 LLM call/cluster
N_QUESTIONS = 3  # deterministic head of the test set
TOP_K = 4  # retrieved chunks per method per question
MAX_CLUSTER_SIZE = 8  # max chunks per summary node in the tree
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DEVICE = "cpu"  # shared GPU: Ollama holds most of VRAM
PREVIEW = 100  # characters of each top passage to print


# --------------------------------------------------------------------------
# 2. Load — first N passages + first N test questions of rag-mini-wikipedia
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> list[str]:
    """Return the first ``n`` passage texts (deterministic head)."""
    df = pd.read_parquet(path)
    return [str(text).strip() for text in df.head(n)["passage"].tolist()]


def load_questions(path: Path, n: int) -> list[dict]:
    """Return the first ``n`` ``{"question", "answer"}`` pairs (deterministic)."""
    df = pd.read_parquet(path)
    return [
        {
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip(),
        }
        for _, row in df.head(n).iterrows()
    ]


# --------------------------------------------------------------------------
# 3. Experiment — build the tree once, then compare flat vs tree retrieval
# --------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def flat_top_k(
    passages: list[str],
    passage_vecs: list[list[float]],
    question_vec: list[float],
    top_k: int,
) -> dict:
    """Rank all passages against the question; return the ``top_k``."""
    ranked = sorted(
        range(len(passages)),
        key=lambda i: _cosine(question_vec, passage_vecs[i]),
        reverse=True,
    )[:top_k]
    return {
        "texts": [passages[i] for i in ranked],
        "ids": list(ranked),
        "scores": [round(_cosine(question_vec, passage_vecs[i]), 3)
                   for i in ranked],
    }


def answer_contains(gold: str, text: str) -> bool:
    """Normalized substring check: are the gold answer's words in ``text``?"""
    return gold.strip().lower() in text.strip().lower()


def run_experiment() -> dict:
    passages = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, N_QUESTIONS)
    llm = OllamaLLM()  # local qwen2.5-coder:7b; only the tree build uses it
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device=BGE_DEVICE)

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passages)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = build_tree(passages, embedder, llm, max_cluster_size=MAX_CLUSTER_SIZE)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows: list[dict] = []
    for q in questions:
        question_vec = embedder.embed_documents([q["question"]])[0]
        flat = flat_top_k(passages, passage_vecs, question_vec, TOP_K)
        tree_ret = retrieve(
            tree["tree"], q["question"], embedder,
            top_k=TOP_K, collapse=True,
        )
        rows.append(
            {
                "question": q["question"],
                "gold": q["answer"],
                "flat": flat,
                "flat_gold_words": answer_contains(q["answer"], flat["texts"][0]),
                "tree": tree_ret,
                "tree_gold_words": answer_contains(q["answer"], tree_ret["texts"][0]),
            }
        )
    query_s = time.perf_counter() - t0
    total_s = time.perf_counter() - t_start

    return {
        "passages": passages,
        "questions": questions,
        "rows": rows,
        "tree": tree,
        "embed_s": embed_s,
        "build_s": build_s,
        "query_s": query_s,
        "total_s": total_s,
        "agg": {
            "questions": len(rows),
            "flat_gold_words": sum(r["flat_gold_words"] for r in rows),
            "tree_gold_words": sum(r["tree_gold_words"] for r in rows),
        },
    }


# --------------------------------------------------------------------------
# 4. Demo — print the comparison
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 09-03 — Tree-vs-flat retrieval eval")
    t = exp["tree"]
    print(f"{len(exp['passages'])} passages; tree {t['levels']} levels / "
          f"{t['leaves']} leaves / {t['llm_calls']} LLM calls in "
          f"{exp['build_s']:.1f}s (total {exp['total_s']:.1f}s)")
    print("=" * 66)

    for i, row in enumerate(exp["rows"], start=1):
        flat, tree = row["flat"], row["tree"]
        print(f"\nQ{i}: {row['question']}")
        print(f"    gold answer    : {row['gold']}")
        print(f"    flat  top [id {flat['ids'][0]}] "
              f"{flat['texts'][0][:PREVIEW]}")
        print(f"    flat  gold words : {row['flat_gold_words']}")
        print(f"    tree  top [id {tree['ids'][0]}] "
              f"{tree['texts'][0][:PREVIEW]}")
        print(f"    tree  gold words : {row['tree_gold_words']}")

    a = exp["agg"]
    print(f"\n[5] Gold-answer words surfaced in the top-1 passage "
          f"({a['questions']} questions)")
    print(f"    flat : {a['flat_gold_words']}/{a['questions']}")
    print(f"    tree : {a['tree_gold_words']}/{a['questions']}")

    print(f"\n[6] Takeaway")
    print("    The tree trades a few LLM calls at index time for query-time")
    print("    retrieval that only compares against handfuls of summaries.")
    print("    It shines when a question maps to a whole cluster; the flat")
    print("    baseline stays competitive on small corpora like this one, so")
    print("    the right choice depends on corpus size and query breadth.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    a = exp["agg"]
    t = exp["tree"]

    for i, row in enumerate(exp["rows"], start=1):
        flat, tree = row["flat"], row["tree"]
        checks.append((f"Q{i} flat returns {TOP_K} non-empty texts + "
                       f"{TOP_K} ids",
                       len(flat["texts"]) == TOP_K
                       and len(flat["ids"]) == TOP_K
                       and all(text.strip() for text in flat["texts"])))
        checks.append((f"Q{i} tree returns {TOP_K} non-empty texts + "
                       f"{TOP_K} ids",
                       len(tree["texts"]) == TOP_K
                       and len(tree["ids"]) == TOP_K
                       and all(text.strip() for text in tree["texts"])))
        checks.append((f"Q{i} flat ids are valid chunk indices (0.."
                       f"{len(exp['passages'])})",
                       all(0 <= idx < len(exp["passages"])
                           for idx in flat["ids"])))
        checks.append((f"Q{i} tree ids are valid chunk indices (0.."
                       f"{len(exp['passages'])})",
                       all(0 <= idx < len(exp["passages"])
                           for idx in tree["ids"])))

    checks.append(("scores are sane (0.0..1.0) for both methods",
                   all(0.0 <= score <= 1.0
                       for row in exp["rows"]
                       for score in row["flat"]["scores"] + row["tree"]["scores"])))
    checks.append((f"tree is well-formed (levels >= 1, leaves == "
                   f"{len(exp['passages'])}, llm_calls <= 40)",
                   t["levels"] >= 1
                   and t["leaves"] == len(exp["passages"])
                   and t["llm_calls"] <= 40))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"  (info) gold-answer words in top-1: flat "
          f"{a['flat_gold_words']}/{a['questions']}, tree "
          f"{a['tree_gold_words']}/{a['questions']} — reported, not required")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
