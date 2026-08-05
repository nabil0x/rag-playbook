"""Lab 02 — Iterative multi-hop retrieval loop (HotpotQA).

Single-shot retrieval fails on questions whose answer lives in two separate
passages. A multi-hop loop turns retrieval into an iterative process: retrieve
with the raw question, let the LLM decide what fact is still missing and
rewrite the query for the next hop, retrieve again, and only then answer.

Each HotpotQA question carries 10 candidate paragraphs (2 gold + 8
distractors). We index those 10 paragraphs per question with local BGE
embeddings and run a small loop:

* hop 0 retrieves top-2 with the raw question (guarantees a first hit),
* each later hop asks the LLM for the next search need (a rewritten
  sub-query) given the question + evidence so far, then retrieves top-2
  unseen paragraphs by cosine similarity,
* when the model reports enough evidence (``done``) or the hop budget runs
  out, the LLM generates the final answer from all accumulated evidence.

The gate is deliberately structural and tolerant: a local 7B model may
retrieve the wrong paragraphs and still pass — what the lab verifies is that
the loop terminates, accumulates evidence, and produces a non-empty answer.

Run from the repo root:
    python src/curriculum/10-agentic-rag/02-multi-hop-loop.py
    python src/curriculum/10-agentic-rag/02-multi-hop-loop.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/10-agentic-rag/02-multi-hop-loop.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
HOTPOT_PATH = Path("Data/corpus/hotpotqa/hotpot_dev_distractor_v1.json")
N_QUESTIONS = 3  # each question costs ~3-4 LLM calls; keep the lab fast
MAX_HOPS = 3  # retrieval hops per question (hop 0 uses the raw question)
RETRIEVE_K = 2  # paragraphs pulled per hop
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
# 3. Experiment — iterative retrieve-then-refine loop per question
# --------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _top_unseen(query: str, vecs: list[list[float]], embedder,
                seen: set[int], k: int) -> list[int]:
    """Cosine top-k indices over ``vecs``, skipping indices already seen."""
    query_vec = embedder.embed_documents([query])[0]
    ranked = sorted(
        range(len(vecs)),
        key=lambda i: _cosine(query_vec, vecs[i]),
        reverse=True,
    )
    return [i for i in ranked if i not in seen][:k]


def _next_search_need(llm, question: str, evidence: list[dict]) -> str | None:
    """Ask the LLM for the next sub-query; None means 'enough evidence'."""
    evidence_block = "\n".join(
        f"- {item['title']}: {item['text'][:180]}..." for item in evidence
    ) or "- (none yet)"
    prompt = (
        "You are doing multi-hop retrieval for a question.\n"
        "Given the question and the evidence gathered so far, decide the "
        "next search need.\n"
        "Rules:\n"
        '- If the evidence already suffices to answer, respond '
        '{"done": true, "search_need": ""}.\n'
        '- Otherwise respond {"done": false, "search_need": "a short, '
        'specific query for the missing fact"}.\n'
        "- Output ONLY JSON.\n\n"
        f"Question: {question}\n\n"
        f"Evidence so far:\n{evidence_block}"
    )
    result = llm.json_object(prompt)
    if not isinstance(result, dict) or "error" in result:
        return None  # cannot refine -> stop hopping
    done = result.get("done", False)
    if done is True or str(done).strip().lower() in ("true", "1", "yes"):
        return None
    need = str(result.get("search_need", result.get("sub_query", ""))).strip()
    return need or None


def run_one_question(rec: dict, llm, embedder) -> dict:
    passages, titles = question_passages(rec)
    vecs = embedder.embed_documents(passages)  # embed the 10 paragraphs once
    gold = rec["answer"]
    gold_titles = gold_paragraphs(rec)

    evidence: list[dict] = []
    hops: list[tuple[str, list[str]]] = []
    seen: set[int] = set()
    llm_calls = 0
    sub_query = rec["question"]
    for hop in range(MAX_HOPS):
        hits = _top_unseen(sub_query, vecs, embedder, seen, RETRIEVE_K)
        if not hits:
            break
        seen.update(hits)
        evidence.extend(
            {"title": titles[i], "text": passages[i]} for i in hits
        )
        hops.append((sub_query, [titles[i] for i in hits]))
        if hop == MAX_HOPS - 1:
            break
        llm_calls += 1
        next_query = _next_search_need(llm, rec["question"], evidence)
        if next_query is None:
            break
        sub_query = next_query

    evidence_block = "\n\n".join(
        f"[{item['title']}] {item['text']}" for item in evidence
    )
    llm_calls += 1
    answer = llm.invoke(
        "Answer the multi-hop question using ONLY the evidence passages "
        "below. If the evidence is insufficient, say what is missing.\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {rec['question']}\n\n"
        "Answer in one or two sentences."
    ).strip()

    return {
        "question": rec["question"],
        "hops": hops,
        "evidence_titles": [titles[i] for i in seen],
        "answer": answer,
        "gold": gold,
        "gold_titles": sorted(gold_titles),
        "answer_contains_gold": answer_contains(gold, answer),
        "llm_calls": llm_calls,
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
            "questions": len(rows),
            "total_llm_calls": sum(row["llm_calls"] for row in rows),
            "hits_gold": sum(
                bool(set(row["evidence_titles"]) & set(row["gold_titles"]))
                for row in rows
            ),
        },
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 10-02 — Iterative multi-hop retrieval loop (HotpotQA)")
    print(f"{exp['agg']['questions']} questions in {exp['total_s']:.1f}s, "
          f"{exp['agg']['total_llm_calls']} LLM calls")
    print("=" * 66)

    for i, row in enumerate(exp["rows"], start=1):
        print(f"\nQ{i}: {row['question'][:90]}")
        for hop, (sub_query, hit_titles) in enumerate(row["hops"], start=1):
            print(f"    hop {hop}: query {sub_query!r} -> {hit_titles}")
        print(f"    evidence    : {row['evidence_titles'][:4]}")
        print(f"    final answer: {row['answer'][:120]}")
        print(f"    gold answer : {row['gold']}  "
              f"(gold paras: {', '.join(row['gold_titles'])[:70]})")
        print(f"    gold in answer: {row['answer_contains_gold']}")

    a = exp["agg"]
    print(f"\n[5] Aggregates over {a['questions']} questions")
    print(f"    total LLM calls        : {a['total_llm_calls']}")
    print(f"    questions whose evidence touched a gold paragraph: "
          f"{a['hits_gold']}/{a['questions']}")

    print(f"\n[6] Takeaway")
    print("    Multi-hop retrieval is a loop, not a single query: each hop")
    print("    re-embeds a model-refined sub-query and grows the evidence.")
    print("    The LLM decides when to stop (done) — turning retrieval into")
    print("    an agentic, budget-bounded process. Answer quality is not")
    print("    gated here; termination, evidence accumulation, and a non-")
    print("    empty answer are.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    rows = exp["rows"]

    checks.append((f"exactly {N_QUESTIONS} questions processed",
                   exp["agg"]["questions"] == N_QUESTIONS))
    checks.append(("every question has a non-empty hop trace",
                   all(row["hops"] for row in rows)))
    checks.append((f"every question terminates within {MAX_HOPS} hops",
                   all(len(row["hops"]) <= MAX_HOPS for row in rows)))
    checks.append(("every question accumulated >= 1 unique retrieved paragraph",
                   all(row["evidence_titles"] for row in rows)))
    checks.append(("every final answer is non-empty",
                   all(row["answer"].strip() for row in rows)))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
