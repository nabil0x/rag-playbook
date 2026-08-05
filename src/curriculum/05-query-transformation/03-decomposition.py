"""Lab 03 — Decomposition: split multi-hop questions into single-hop chunks.

A multi-hop question needs facts from 2+ chunks ("Who wrote the song, and
when did that writer die?"). Plain top-k embeds the WHOLE question and
returns the single most-similar chunk — the other fact's chunk is out of
reach, so the candidate set is incomplete before any LLM reads it.

Decomposition fixes retrieval, not the LLM: the decomposer splits the
question into independent sub-questions (one fact each), the retriever is
run with the original question AND every sub-question, and the results are
merged, deduplicated, and cut to ``top_k``. Every fact's chunk now gets a
retrieval pass of its own.

Data: HotpotQA (``hotpot_dev_distractor_v1.json``) — a multi-hop benchmark.
Each question ships its own 10-paragraph "context" plus gold
``supporting_facts`` (which paragraphs the answer needs). This lab indexes
each question's 10 paragraphs into its own small FAISS store, then compares:

* PLAIN top-3 — the raw question through ``SimilarityRetriever``;
* DECOMPOSED — ``DecomposeRetriever`` (``src/retrieval/decompose.py``) with the
  same inner retriever.

The three questions are pre-selected: plain top-3 MISSES at least one gold
supporting paragraph for every one of them (verified against the data), and
decomposition recovers the full supporting set — that is the whole point of
the technique, measured on gold labels instead of vibes.

The Groq LLM is only the decomposer (one call per question); embeddings stay
local BGE.

Run from the repo root:
    python src/curriculum/05-query-transformation/03-decomposition.py
    python src/curriculum/05-query-transformation/03-decomposition.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/05-query-transformation/03-decomposition.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives in the repo-root .env

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llms.groq import GroqLLM  # noqa: E402
from retrieval.decompose import DecomposeRetriever  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
HOTPOTQA_PATH = Path("Data/corpus/hotpotqa/hotpot_dev_distractor_v1.json")
# Three multi-hop questions, pre-selected from hotpotqa: for each one, plain
# top-3 misses a gold supporting paragraph while decomposition recovers it
# (verified against the data + the Groq decomposer before shipping).
QUESTION_IDS = [
    "5a722b8655429971e9dc9329",  # "Who was the writer of These Boots… and who died in 2007?"
    "5a8a3e745542996c9b8d5e70",  # "What is the name for the adventure in Tunnels and Trolls…?"
    "5adf37a95542995ec70e8f97",  # "The 2011–12 VCU Rams… represented VCU which was founded in what year?"
]
TOP_K = 3  # plain retrieval depth (deliberately small: the question can only fit one fact)
DECOMPOSE_TOP_K = 6  # merged depth after original + sub-question retrievals
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq is the *decomposer* LLM, never the embedder
# (Gemini alternative: LLM_MODEL = "gemini-2.5-flash" — needs GOOGLE_API_KEY in .env)
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
N_PARAGRAPHS = 10  # hotpotqa provides 10 context paragraphs per question


# --------------------------------------------------------------------------
# 2. Load — hotpotqa questions with their self-contained 10-paragraph contexts
# --------------------------------------------------------------------------
def load_hotpotqa(path: Path, ids: list[str]) -> dict[str, dict]:
    """Return {qid: question_dict} for the requested hotpotqa rows.

    Each dict keeps ``question``, ``answer``, ``supporting_facts`` (list of
    [title, sentence_index] pairs) and ``context`` (10 [title, [sentences…]]
    paragraphs). The gold ``supporting_facts`` titles are the ground truth
    the lab measures both retrievers against.
    """
    by_id: dict[str, dict] = {}
    with open(path) as f:
        for q in json.load(f):  # hotpotqa is one JSON array, not JSON-lines
            if q["_id"] in ids:
                by_id[q["_id"]] = q
    return by_id


def distinct_supporting_titles(q: dict) -> list[str]:
    """Gold supporting paragraph titles, deduplicated, in first-seen order."""
    return list(dict.fromkeys(s[0] for s in q["supporting_facts"]))


def preview(text: str, limit: int = 62) -> str:
    """Flatten a paragraph for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --------------------------------------------------------------------------
# 3. Experiment — per question: plain top-3 vs decomposed retrieval
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    questions = load_hotpotqa(HOTPOTQA_PATH, QUESTION_IDS)
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)

    results = []
    for qid in QUESTION_IDS:
        q = questions[qid]
        distinct = distinct_supporting_titles(q)

        # Each question gets its own small store over its 10 paragraphs.
        chunks = [
            Document(page_content=" ".join(sents), metadata={"title": title})
            for title, sents in q["context"]
        ]
        store = FAISSVectorStore(embedding=embedder)
        t0 = time.perf_counter()
        store.add(chunks, embeddings=embedder.embed_documents(
            [c.page_content for c in chunks]
        ))
        index_s = time.perf_counter() - t0

        inner = SimilarityRetriever(store, top_k=TOP_K)
        plain_docs = inner.retrieve(q["question"])

        decomposer = GroqLLM(model=LLM_MODEL)
        decomposed = DecomposeRetriever(decomposer, inner, top_k=DECOMPOSE_TOP_K)
        t0 = time.perf_counter()
        sub_questions = decomposed._decompose(q["question"])
        decompose_s = time.perf_counter() - t0
        decomposed_docs = decomposed.retrieve(q["question"])

        results.append(
            {
                "qid": qid,
                "question": q["question"],
                "answer": q["answer"],
                "supporting": distinct,
                "plain_titles": [d.metadata["title"] for d in plain_docs],
                "sub_questions": sub_questions,
                "decomposed_titles": [d.metadata["title"] for d in decomposed_docs],
                "decomposed_first": preview(decomposed_docs[0].page_content),
                "index_s": index_s,
                "decompose_s": decompose_s,
                "n_paragraphs": len(chunks),
            }
        )

    return {"questions": questions, "results": results}


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 03 — Decomposition: split multi-hop questions into single-hop chunks")
    print(f"{BGE_MODEL_NAME} (local) | HotpotQA | {LLM_MODEL} decomposer")
    print("=" * 66)

    print(f"\n[1] Questions ({len(exp['results'])} from hotpotqa, each with its own")
    print("    10-paragraph context + gold supporting facts):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"][:8]}] "{r["question"]}"')
        print(f"      answer: {r['answer']!r}")
        print(f"      gold supporting paragraphs: {r['supporting']}")

    print(f"\n[2] Plain top-{TOP_K} (raw question, one retrieval pass):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"][:8]}] "{r["question"]}"')
        for t in r["plain_titles"]:
            mark = " [SUPPORTING]" if t in r["supporting"] else ""
            print(f"      - {t}{mark}")
        missing = [t for t in r["supporting"] if t not in r["plain_titles"]]
        print(f"      MISSING gold paragraphs: {missing if missing else 'none'}")

    print(f"\n[3] Decomposed (original + sub-questions, merged to top-{DECOMPOSE_TOP_K}):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"][:8]}] "{r["question"]}"')
        print(f"      sub-questions ({r['decompose_s']:.1f}s):")
        for s in r["sub_questions"]:
            print(f"        - {s}")
        for t in r["decomposed_titles"]:
            mark = " [SUPPORTING]" if t in r["supporting"] else ""
            print(f"      - {t}{mark}")
        recovered = all(t in r["decomposed_titles"] for t in r["supporting"])
        print(f"      all gold paragraphs recovered: {recovered}")

    print("\n[4] Takeaway")
    print("    Decomposition fixes retrieval for multi-hop questions: one")
    print("    retrieval pass per fact (plus the original), merged. The gold")
    print("    supporting_facts labels show the win — every question here is")
    print("    one plain top-3 retrieval could NOT answer, and decomposition")
    print("    brings both paragraphs into the candidate set before the LLM")
    print("    ever reads them.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural: all requested questions loaded, with their full contexts.
    checks.append(("all 3 hotpotqa questions loaded",
                   len(exp["results"]) == len(QUESTION_IDS)))
    checks.append(("every question indexes its full 10-paragraph context",
                   all(r["n_paragraphs"] == N_PARAGRAPHS for r in exp["results"])))
    checks.append(("every question has >= 2 distinct gold supporting paragraphs",
                   all(len(r["supporting"]) >= 2 for r in exp["results"])))

    # The problem: plain top-3 must MISS at least one gold paragraph — this
    # is what makes the multi-hop question worth decomposing (deterministic:
    # pure retrieval, no LLM).
    for r in exp["results"]:
        tag = f"Q{r['qid'][:8]}"
        missing = [t for t in r["supporting"] if t not in r["plain_titles"]]
        checks.append((f"{tag} plain top-{TOP_K} misses >= 1 gold paragraph",
                       len(missing) >= 1))

    # The fix: decomposition must generate sub-questions…
    for r in exp["results"]:
        tag = f"Q{r['qid'][:8]}"
        checks.append((f"{tag} generated >= 1 sub-question",
                       len(r["sub_questions"]) >= 1))
        checks.append((f"{tag} no sub-question repeats the original verbatim",
                       all(s.strip().lower() != r["question"].strip().lower()
                           for s in r["sub_questions"])))

    # …and the merged retrieval must recover EVERY gold paragraph. This is
    # the teaching gate: it only passes when decomposition actually widened
    # the candidate set to the facts plain retrieval could not reach.
    for r in exp["results"]:
        tag = f"Q{r['qid'][:8]}"
        recovered = all(t in r["decomposed_titles"] for t in r["supporting"])
        checks.append((f"{tag} decomposition recovers ALL gold paragraphs",
                       recovered))
        checks.append((f"{tag} decomposed titles are deduplicated",
                       len(r["decomposed_titles"]) == len(set(r["decomposed_titles"]))))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
