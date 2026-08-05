"""Lab 04 — ReAct + reflection over a LangGraph state machine.

Lab 02 hard-coded the "retrieve, then refine" loop in Python. This lab
promotes the same control flow into an explicit, inspectable state machine
with LangGraph: the nodes (``plan`` -> ``retrieve`` -> ``generate`` ->
``reflect``) share one typed state dict, and a conditional edge decides at
runtime whether to loop back with a revised query or stop. Conversation
history lives in the state, and the node-by-node trace is read straight off
``graph.stream(..., stream_mode="updates")``.

The reflection step is the agentic part: after generating an answer, the LLM
is asked to critique it — is it grounded in the retrieved evidence? is it
complete? — and returns a ``revised_query``. If the critique is negative
(and the loop budget allows), the graph routes back to ``plan`` with the
revised query and tries again; otherwise it ends.

Per HotpotQA question the 10 candidate paragraphs are embedded once and
indexed by cosine similarity (same setup as lab 02). The gate is structural:
graph compiles, every run terminates with ``loop_count`` bounded, the final
answer is non-empty, and history is carried in the state. Answer correctness
is deliberately not gated — a local 7B model driving this graph is a
demonstration, not a benchmark.

Run from the repo root:
    python curriculum/10-agentic-rag/04-react-reflection.py
    python curriculum/10-agentic-rag/04-react-reflection.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/10-agentic-rag/04-react-reflection.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
HOTPOT_PATH = Path("Data/corpus/hotpotqa/hotpot_dev_distractor_v1.json")
N_QUESTIONS = 3  # each question costs ~3-6 LLM calls (plan+generate+reflect)
MAX_LOOPS = 2  # reflection iterations before the graph must stop
RETRIEVE_K = 2  # paragraphs pulled per retrieve node
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


# --------------------------------------------------------------------------
# 3. Experiment — a plan/retrieve/generate/reflect LangGraph state machine
# --------------------------------------------------------------------------
def _extend(left: list, right: list) -> list:
    """Reducer: append rather than replace, so lists survive across nodes."""
    return [*left, *right]


class GraphState(TypedDict):
    """The one typed state dict every node reads and writes."""

    question: str
    evidence: Annotated[list, _extend]  # [{"title", "text", "index"}]
    answer: str
    critique: dict  # {"grounded", "complete", "revised_query"}
    loop_count: int
    history: Annotated[list, _extend]  # [{"role", "content"}] turns carried along


def _cosine(a: list[float], b: list[float]) -> float:
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _render_evidence(evidence: list[dict]) -> str:
    return "\n\n".join(f"[{e['title']}] {e['text']}" for e in evidence)


def build_graph(llm, embedder, passages: list[str], titles: list[str]) -> object:
    """Assemble and compile the plan/retrieve/generate/reflect graph."""
    paragraph_vecs = embedder.embed_documents(passages)  # embed once per question

    def plan(state: GraphState) -> dict:
        """Rewrite/decompose the question into a precise retrieval query."""
        rewritten = llm.invoke(
            "Rewrite or decompose the question into a precise retrieval "
            "query. If it is already precise, keep it nearly unchanged. "
            "Return ONLY the rewritten query text.\n\n"
            f"Question: {state['question']}"
        ).strip()
        return {"question": rewritten or state["question"]}

    def retrieve(state: GraphState) -> dict:
        """Cosine top-k over the question's paragraphs, skipping seen ones."""
        query_vec = embedder.embed_documents([state["question"]])[0]
        seen = {item["index"] for item in state["evidence"]}
        ranked = sorted(
            range(len(passages)),
            key=lambda i: _cosine(query_vec, paragraph_vecs[i]),
            reverse=True,
        )
        new_items = []
        for index in ranked:
            if index in seen:
                continue
            new_items.append(
                {"title": titles[index], "text": passages[index], "index": index}
            )
            if len(new_items) >= RETRIEVE_K:
                break
        return {"evidence": new_items}

    def generate(state: GraphState) -> dict:
        """Answer from the accumulated evidence, appending to history."""
        answer = llm.invoke(
            "Answer the question using ONLY the evidence passages below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Evidence:\n{_render_evidence(state['evidence'])}\n\n"
            f"Question: {state['question']}\n\n"
            "Answer in one or two sentences."
        ).strip()
        return {
            "answer": answer,
            "history": [{"role": "assistant", "content": answer}],
        }

    def reflect(state: GraphState) -> dict:
        """Critique the answer; feed a revised query back to ``plan``."""
        raw = llm.json_object(
            "Critique the generated answer for groundedness and completeness "
            "against the evidence.\n"
            "Rules:\n"
            '- "grounded": true if the answer is supported by the evidence.\n'
            '- "complete": true if the answer fully addresses the question.\n'
            '- "revised_query": a better query if the answer is weak, else '
            "keep the current question.\n"
            "- Output ONLY JSON: {\"grounded\": true, \"complete\": true, "
            "\"revised_query\": \"...\"}\n\n"
            f"Question: {state['question']}\n\n"
            f"Evidence:\n{_render_evidence(state['evidence'])}\n\n"
            f"Answer: {state['answer']}"
        )
        if not isinstance(raw, dict) or "error" in raw:
            raw = {}
        grounded = _as_bool(raw.get("grounded"))
        complete = _as_bool(raw.get("complete"))
        revised = str(raw.get("revised_query", "")).strip() or state["question"]
        return {
            "critique": {
                "grounded": grounded,
                "complete": complete,
                "revised_query": revised,
            },
            "loop_count": state["loop_count"] + 1,
            "question": revised,
        }

    def route(state: GraphState) -> str:
        """Loop to ``plan`` while the critique is negative and budget allows."""
        weak = not state["critique"].get("grounded") or not state["critique"].get("complete")
        if state["loop_count"] < MAX_LOOPS and weak:
            return "plan"
        return END

    builder = StateGraph(GraphState)
    builder.add_node("plan", plan)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", route, {"plan": "plan", END: END})
    return builder.compile()


def run_one_question(rec: dict, llm, embedder) -> dict:
    passages, titles = question_passages(rec)
    graph = build_graph(llm, embedder, passages, titles)

    initial: dict = {
        "question": rec["question"],
        "evidence": [],
        "answer": "",
        "critique": {},
        "loop_count": 0,
        "history": [{"role": "user", "content": rec["question"]}],
    }
    trace: list[str] = []
    final_state: dict = {}
    # stream_mode=["updates", "values"]: updates give the node trace, the last
    # values event is the reducer-merged final state (updates alone only show
    # raw node returns, so they cannot reconstruct reducer channels).
    for mode, data in graph.stream(initial, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node in data:
                trace.append(node)
        else:
            final_state = data

    return {
        "question": rec["question"],
        "trace": trace,
        "state": final_state,
        "gold": rec["answer"],
        "gold_titles": sorted(gold_paragraphs(rec)),
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
        "compiled": True,  # run_experiment cannot finish unless the graph compiled
        "total_s": total_s,
        "agg": {
            "questions": len(rows),
            "total_loops": sum(row["state"]["loop_count"] for row in rows),
        },
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 10-04 — ReAct + reflection over a LangGraph state machine")
    print(f"{exp['agg']['questions']} questions in {exp['total_s']:.1f}s")
    print("=" * 66)

    for i, row in enumerate(exp["rows"], start=1):
        state = row["state"]
        print(f"\nQ{i}: {row['question'][:90]}")
        print(f"    node trace : {' -> '.join(row['trace'])}")
        print(f"    loop_count : {state['loop_count']}")
        print(f"    critique   : {state['critique']}")
        print(f"    final ans  : {state.get('answer', '')[:120]}")
        print(f"    history    : {[h['role'] for h in state.get('history', [])]}")
        print(f"    gold       : {row['gold']}  "
              f"(paras: {', '.join(row['gold_titles'])[:70]})")

    print(f"\n[5] Takeaway")
    print("    The reflection loop is an explicit data structure: nodes are")
    print("    state transforms and the loop decision is a conditional edge.")
    print("    Because state is typed and shared, the whole run — including")
    print("    history — is inspectable after the fact. The gate checks the")
    print("    machine-level guarantees (termination, budget, non-empty")
    print("    answers, history carried) and leaves correctness to a later")
    print("    evaluation lab.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    rows = exp["rows"]

    checks.append(("graph compiled and ran", bool(exp.get("compiled"))))
    checks.append((f"exactly {N_QUESTIONS} questions processed",
                   exp["agg"]["questions"] == N_QUESTIONS))
    checks.append(("every run terminated with a non-empty node trace",
                   all(row["trace"] for row in rows)))
    checks.append(("every trace ends in reflect or generate",
                   all(row["trace"][-1] in ("reflect", "generate") for row in rows)))
    checks.append((f"loop_count <= {MAX_LOOPS + 1} for every run (no infinite loop)",
                   all(row["state"]["loop_count"] <= MAX_LOOPS + 1 for row in rows)))
    checks.append(("every final answer is non-empty",
                   all(row["state"].get("answer", "").strip() for row in rows)))
    checks.append(("history is carried (state holds the question + turns)",
                   all(
                       any(isinstance(h, dict) and h.get("role") == "user"
                           for h in row["state"].get("history", []))
                       for row in rows
                   )))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
