"""Lab 04 — Local vs global search over the GraphRAG index (HotpotQA).

The index from labs 01-03 can be queried two ways, and the choice is a
real engineering tradeoff:

* **Local search** starts from the question's own entities, links them
  into the graph, expands one hop to their neighbors, and answers from the
  passages those entities live in. Precise — it only sees passages that
  touch the question — but it fails when the question's entities are not
  in the graph or the answer needs a *distant* part of the corpus.
* **Global search** ignores the graph topology for retrieval: it embeds
  the question, ranks the precomputed community summaries by similarity,
  and answers from the top summaries. It sees the whole corpus (compressed
  into summaries), so it can answer corpus-level questions, but it is
  lossier — the answer must survive the map/reduce compression.

The lab runs both strategies on the same HotpotQA questions. Each question
carries 10 candidate paragraphs; we build a graph over them, detect
communities, summarize the top ones, then evaluate both searches against
the dataset's gold answer and gold paragraphs (``supporting_facts``).

Run from the repo root:
    python src/curriculum/08-graphrag/04-local-global.py
    python src/curriculum/08-graphrag/04-local-global.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/08-graphrag/04-local-global.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402
from tools.graph import build_entity_graph  # noqa: E402
from tools.graphrag import (  # noqa: E402
    community_summaries,
    detect_communities,
    global_search,
    local_search,
)

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
HOTPOT_PATH = Path("Data/corpus/hotpotqa/hotpot_dev_distractor_v1.json")
N_QUESTIONS = 3  # each question costs ~16 LLM calls; keep the lab fast
MAX_COMMUNITIES = 4  # per-question cap on community summaries
LINK_TOP_N = 6  # max passages local search may surface
LINK_THRESHOLD = 0.55  # minimum cosine for entity linking
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DEVICE = "cpu"  # shared GPU: Ollama holds most of VRAM


# --------------------------------------------------------------------------
# 2. Load — first N HotpotQA questions (each has 10 context paragraphs)
# --------------------------------------------------------------------------
def load_questions(path: Path, n: int) -> list[dict]:
    """Return the first ``n`` questions from the dev set."""
    with open(path) as f:
        records = json.load(f)
    return records[:n]


def question_passages(rec: dict) -> tuple[list[str], list[str]]:
    """Return (passage_texts, paragraph_titles) for the 10 context paragraphs."""
    texts: list[str] = []
    titles: list[str] = []
    for title, sentences in rec["context"]:
        titles.append(title)
        texts.append(" ".join(sentences))
    return texts, titles


def gold_paragraphs(rec: dict) -> set[str]:
    """The paragraph titles HotpotQA marks as required evidence."""
    return {title for title, _ in rec["supporting_facts"]}


def answer_contains(gold: str, answer: str) -> bool:
    """Normalized substring check: is the gold answer inside the answer?"""
    return gold.strip().lower() in answer.strip().lower()


# --------------------------------------------------------------------------
# 3. Experiment — build a per-question index, run both searches, evaluate
# --------------------------------------------------------------------------
def run_one_question(rec: dict, llm, embedder) -> dict:
    passages, titles = question_passages(rec)
    gold = rec["answer"]
    gold_titles = gold_paragraphs(rec)

    graph = build_entity_graph(passages, llm)
    communities = detect_communities(graph, seed=42)
    summaries = community_summaries(
        llm, graph, communities, max_communities=MAX_COMMUNITIES
    )

    local = local_search(
        rec["question"], llm, embedder, graph, passages,
        top_n=LINK_TOP_N, threshold=LINK_THRESHOLD,
    )
    glob = global_search(rec["question"], llm, embedder, summaries)

    local_gold_ids = [
        titles[i] for i in local["retrieved_ids"]
        if titles[i] in gold_titles
    ]
    return {
        "question": rec["question"],
        "gold": gold,
        "gold_titles": sorted(gold_titles),
        "local": {
            "answer": local["answer"],
            "linked": local["linked"],
            "gold_titles_retrieved": local_gold_ids,
        },
        "global": {
            "answer": glob["answer"],
        },
        "scores": {
            "local_answer_contains_gold": answer_contains(gold, local["answer"]),
            "global_answer_contains_gold": answer_contains(gold, glob["answer"]),
            "local_gold_para_recall": len(local_gold_ids) > 0,
        },
    }


def run_experiment() -> dict:
    questions = load_questions(HOTPOT_PATH, N_QUESTIONS)
    llm = OllamaLLM()
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device=BGE_DEVICE)

    t0 = time.perf_counter()
    rows = [run_one_question(rec, llm, embedder) for rec in questions]
    total_s = time.perf_counter() - t0

    return {
        "rows": rows,
        "total_s": total_s,
        "agg": {
            "local_answer_contains_gold": sum(
                r["scores"]["local_answer_contains_gold"] for r in rows
            ),
            "global_answer_contains_gold": sum(
                r["scores"]["global_answer_contains_gold"] for r in rows
            ),
            "local_gold_para_recall": sum(
                r["scores"]["local_gold_para_recall"] for r in rows
            ),
            "questions": len(rows),
        },
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 08-04 — Local vs global search over the GraphRAG index")
    print(f"{exp['agg']['questions']} questions in {exp['total_s']:.1f}s")
    print("=" * 66)

    for i, row in enumerate(exp["rows"], start=1):
        print(f"\nQ{i}: {row['question'][:90]}")
        print(f"    gold answer : {row['gold']}")
        print(f"    gold paras  : {', '.join(row['gold_titles'])[:80]}")
        print(f"    linked      : {row['local']['linked']}")
        print(f"    local  hit  : {row['scores']['local_answer_contains_gold']} "
              f"(gold paras {row['local']['gold_titles_retrieved']})")
        print(f"    global hit  : {row['scores']['global_answer_contains_gold']}")
        print(f"    local  ans  : {row['local']['answer'][:110]}")
        print(f"    global ans  : {row['global']['answer'][:110]}")

    a = exp["agg"]
    print(f"\n[5] Aggregates over {a['questions']} questions")
    print(f"    local  answer contains gold : {a['local_answer_contains_gold']}/{a['questions']}")
    print(f"    global answer contains gold : {a['global_answer_contains_gold']}/{a['questions']}")
    print(f"    local  surfaced a gold para : {a['local_gold_para_recall']}/{a['questions']}")

    print(f"\n[6] Takeaway")
    print("    Local search is entity-precise but narrow: it only sees")
    print("    passages that touch the question's entities, so it shines on")
    print("    multi-hop questions whose bridge entities are in the graph.")
    print("    Global search trades that precision for recall: the question")
    print("    is answered from compressed community summaries, so it can")
    print("    generalize but may lose exact facts during map/reduce.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    a = exp["agg"]

    checks.append((f"exactly {N_QUESTIONS} questions processed",
                   a["questions"] == N_QUESTIONS))
    checks.append(("local search linked entities for >= 1 question",
                   any(r["local"]["linked"] for r in exp["rows"])))
    checks.append(("every answer is non-empty",
                   all(r["local"]["answer"].strip()
                       for r in exp["rows"])
                   and all(r["global"]["answer"].strip()
                           for r in exp["rows"])))
    checks.append(("local search surfaced a gold paragraph for >= 1 question",
                   a["local_gold_para_recall"] >= 1))
    checks.append(("per-question scores are 0/1 flags (reportable)",
                   all(v in (0, 1)
                       for r in exp["rows"]
                       for v in r["scores"].values())))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
