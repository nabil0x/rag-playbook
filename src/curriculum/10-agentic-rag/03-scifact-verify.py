"""Lab 03 — Evidence verification loop (SciFact claims).

Generation metrics grade answers; attribution metrics grade *whether the
answer is actually backed by the retrieved evidence*. This lab implements the
evidence-verification half of that idea with ``src/tools/verifier.py`` over the
SciFact corpus.

For each claim we (1) embed the claim and rank the whole 5183-doc corpus by
cosine similarity to retrieve top-5 candidate evidence passages, then (2) ask
the local LLM — via ``json_object`` — for a three-way verdict:

* ``SUPPORTED``       — the evidence backs the claim,
* ``REFUTED``         — the evidence contradicts the claim,
* ``NOT_ENOUGH_INFO`` — the evidence neither supports nor refutes it.

The corpus is embedded exactly ONCE (CPU, ~5-6 minutes for 5183 docs) and
cached in the experiment dict; per-claim retrieval is then a pure cosine rank
over the cached vectors. The gate is structural and tolerant: it does NOT
require agreement with SciFact's gold labels — a local 7B model verifying 4
claims is not a benchmark run. It checks that every claim got a valid verdict,
a non-empty reason (or the explicit "parse failure" fallback), and that the
loop terminated.

Run from the repo root:
    python src/curriculum/10-agentic-rag/03-scifact-verify.py
    python src/curriculum/10-agentic-rag/03-scifact-verify.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/10-agentic-rag/03-scifact-verify.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from collections import Counter  # noqa: E402

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402
from tools.verifier import VALID_VERDICTS, verify_loop  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
CORPUS_PATH = Path("Data/corpus/scifact/data/corpus.jsonl")
CLAIMS_PATH = Path("Data/corpus/scifact/data/claims_dev.jsonl")
N_CLAIMS = 4  # each claim = 1 LLM verdict call; ~5-8 LLM calls total
TOP_K = 5  # candidate evidence passages retrieved per claim
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DEVICE = "cpu"  # shared GPU: Ollama holds most of VRAM


# --------------------------------------------------------------------------
# 2. Load — claims (deterministic head) and the full SciFact corpus
# --------------------------------------------------------------------------
def load_corpus(path: Path) -> list[dict]:
    """All corpus records: ``{"doc_id", "title", "abstract"}``."""
    docs: list[dict] = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def corpus_texts(docs: list[dict]) -> list[str]:
    """One searchable string per doc: ``title. <abstract sentences>``."""
    return [f"{doc['title']}. {' '.join(doc['abstract'])}" for doc in docs]


def load_claims(path: Path, n: int) -> list[dict]:
    """First ``n`` claims from the dev set (each carries its ``id``)."""
    claims: list[dict] = []
    with open(path) as f:
        for line in f:
            claims.append(json.loads(line))
            if len(claims) >= n:
                break
    return claims


# --------------------------------------------------------------------------
# 3. Experiment — embed the corpus once, then verify each claim
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    corpus = load_corpus(CORPUS_PATH)
    texts = corpus_texts(corpus)
    claims = load_claims(CLAIMS_PATH, N_CLAIMS)

    llm = OllamaLLM()  # local qwen2.5-coder:7b; json_object returns the verdict
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device=BGE_DEVICE)

    # Embed the 5183-doc corpus ONCE (~5-6 min on CPU) and keep the vectors in
    # the experiment dict so every claim reuses them.
    t0 = time.perf_counter()
    corpus_embeddings = embedder.embed_documents(texts)
    embed_s = time.perf_counter() - t0

    results = verify_loop(
        llm,
        embedder,
        [claim["claim"] for claim in claims],
        texts,
        top_k=TOP_K,
        progress=lambda done, total: print(
            f"  verified {done}/{total} claims", flush=True
        ),
        corpus_embeddings=corpus_embeddings,
    )
    total_s = time.perf_counter() - t0

    rows = []
    for claim, result in zip(claims, results):
        rows.append(
            {
                "id": claim["id"],
                "claim": result["claim"],
                "verdict": result["verdict"],
                "reason": result["reason"],
                "evidence_indices": result["evidence_indices"],
                "evidence_titles": [
                    corpus[i]["title"] for i in result["evidence_indices"]
                ],
            }
        )
    return {
        "rows": rows,
        "corpus_size": len(corpus),
        "embed_s": embed_s,
        "total_s": total_s,
        "corpus_embeddings": corpus_embeddings,  # cached for reuse/inspection
        "distribution": dict(Counter(row["verdict"] for row in rows)),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 10-03 — Evidence verification loop (SciFact claims)")
    print(f"{len(exp['rows'])} claims over {exp['corpus_size']} docs; "
          f"embed {exp['embed_s']:.0f}s, total {exp['total_s']:.0f}s")
    print("=" * 66)

    for i, row in enumerate(exp["rows"], start=1):
        print(f"\nC{i} ({row['id']}): {row['claim'][:100]}")
        for title in row["evidence_titles"][:3]:
            print(f"    evidence: {title[:95]}")
        print(f"    verdict : {row['verdict']}")
        print(f"    reason  : {row['reason'][:140]}")

    print(f"\n[4] Verdict distribution over {len(exp['rows'])} claims")
    for verdict in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"):
        print(f"    {verdict:<16} {exp['distribution'].get(verdict, 0)}")
    print(f"    {'TOTAL':<16} {len(exp['rows'])}")

    print(f"\n[5] Takeaway")
    print("    Verification is RAG's answer-quality gate: retrieve the most")
    print("    plausible evidence, then ask the model whether it actually")
    print("    supports, refutes, or ignores the claim. The verdict is a")
    print("    machine-checkable signal (not free text), and the 'parse")
    print("    failure' fallback guarantees the pipeline never crashes on a")
    print("    stubborn local model. Agreement with SciFact gold labels is")
    print("    intentionally not enforced in the gate.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    rows = exp["rows"]

    checks.append((f"all {N_CLAIMS} claims were verified (loop terminated)",
                   len(rows) == N_CLAIMS))
    checks.append(("every verdict is one of SUPPORTED/REFUTED/NOT_ENOUGH_INFO",
                   all(row["verdict"] in VALID_VERDICTS for row in rows)))
    checks.append(("every verdict has a non-empty reason (or a parse-failure note)",
                   all(row["reason"].strip() for row in rows)))
    checks.append((f"every claim retrieved {TOP_K} evidence passages",
                   all(len(row["evidence_indices"]) == TOP_K for row in rows)))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
