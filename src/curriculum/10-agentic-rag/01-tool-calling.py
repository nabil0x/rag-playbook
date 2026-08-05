"""Lab 01 — Tool-calling agent: retrieval as a model decision.

Plain RAG always retrieves. An agentic RAG pipeline instead *lets the model
decide* whether retrieval is needed at all, what to search for, and how many
searches to run before answering. That decision is the core idea of this lab.

We build a small in-memory index over 30 rag-mini-wikipedia passages (chosen
at a stride so they span many topics: Uruguay, Egypt, sea otters, elephants,
Finland, ...) and hand the LLM two tools:

* ``search_documents(query)``  -> top-4 snippets,
* ``read_document(doc_id)``    -> one full passage.

The agent loop is implemented *manually*: the LLM is prompted to either
answer in plain text or emit ONLY a JSON tool invocation
(``{"name": ..., "arguments": {...}}``), the JSON is parsed, the tool runs,
the observation is fed back, and the loop repeats up to ``max_tool_calls``.
This matters because the local model (qwen2.5-coder:7b via ChatOllama) never
populates structured ``tool_calls`` — it emits well-formed JSON as plain text
content, so ``bind_tools``-style agent frameworks silently never fire.

Three question types show the tradeoff:
(a) a no-retrieval question the model answers from its own knowledge
    (expect an EMPTY tool trace — retrieval is a decision, not a default),
(b) a single-retrieval factual question about the corpus,
(c) a two-retrieval comparison that must combine two passages.

Run from the repo root:
    python curriculum/10-agentic-rag/01-tool-calling.py
    python curriculum/10-agentic-rag/01-tool-calling.py --verify
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/10-agentic-rag/01-tool-calling.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
N_PASSAGES = 30  # index size; 30 passages embed in a few seconds on CPU
PASSAGE_STRIDE = 107  # spread the 30 passages across the corpus (diverse topics)
TOP_K = 4  # snippets returned by search_documents
MAX_TOOL_CALLS = 3  # per-question cap on tool executions (~6-10 LLM calls total)
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DEVICE = "cpu"  # shared GPU: Ollama holds most of VRAM

# (kind, question) — the three question types the lab demonstrates.
QUESTIONS: list[tuple[str, str]] = [
    ("no-retrieval", "What does RAG stand for?"),
    ("single-retrieval", "What prompted Egyptian opposition to take a "
                          "stronger stand against British occupation?"),
    ("two-retrieval", "Which marine mammal lacks insulating blubber, and "
                      "what animal is famed for its memory and high "
                      "intelligence?"),
]


# --------------------------------------------------------------------------
# 2. Load — 30 passages sampled at a stride across the corpus
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int, stride: int) -> list[str]:
    """Return ``n`` passage texts spread ``stride`` rows apart (deterministic)."""
    df = pd.read_parquet(path)
    return [str(text).strip() for text in df.iloc[0 : stride * n : stride]["passage"].tolist()]


# --------------------------------------------------------------------------
# 3. Experiment — a manual tool-calling agent loop
# --------------------------------------------------------------------------
class ToolCallingAgent:
    """LLM decides when/how often to query an in-memory passage index."""

    #: Plain-python tools: search returns snippets, read returns full text.
    SEARCH_DESCRIPTION = (
        "Search the passage store and return the top-%d snippets "
        "(as '[id] text...') for the query." % TOP_K
    )
    READ_DESCRIPTION = (
        "Return the full text of one passage by its id. Use this when a "
        "snippet from search_documents is not enough context."
    )

    SYSTEM_PROMPT = (
        "You are an agent with access to two tools.\n\n"
        f"Tool search_documents — {SEARCH_DESCRIPTION}\n"
        '  call shape: {"name": "search_documents", "arguments": {"query": "..."}}\n'
        f"Tool read_document — {READ_DESCRIPTION}\n"
        '  call shape: {"name": "read_document", "arguments": {"doc_id": "0"}}\n'
        "\n"
        "Protocol:\n"
        "- FIRST decide whether the tools are needed at all. General-knowledge "
        "questions (definitions, well-known acronyms like RAG, common facts) "
        "MUST be answered directly from your knowledge with NO tool call — "
        "searching for a question you can already answer is a failure.\n"
        "- Only call a tool when the question asks for a specific fact stored "
        "in the document store (articles about countries, animals, history, "
        "science).\n"
        "- When you call a tool, respond with ONLY one JSON tool invocation "
        "(no surrounding text); you may issue one call per turn and inspect "
        "the result before deciding the next step.\n"
        "- When you have enough evidence (or no tool is needed), respond with "
        "ONLY the final answer as plain text.\n"
        "- Never invent tool results; never cite passages you did not see."
    )

    def __init__(self, llm, passages: list[str], embedder, top_k: int = TOP_K,
                 max_tool_calls: int = MAX_TOOL_CALLS):
        self.llm = llm
        self.passages = passages
        self.embedder = embedder
        self.top_k = top_k
        self.max_tool_calls = max_tool_calls
        self.embeddings = embedder.embed_documents(passages)  # embed once

    # -- tools ------------------------------------------------------------
    def _cosine(self, a, b) -> float:
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)

    def search_documents(self, query: str) -> str:
        """Tool: top-``top_k`` snippets, tagged with their passage id."""
        query_vec = self.embedder.embed_documents([query])[0]
        ranked = sorted(
            range(len(self.embeddings)),
            key=lambda i: self._cosine(query_vec, self.embeddings[i]),
            reverse=True,
        )
        snippets = []
        for index in ranked[: self.top_k]:
            text = self.passages[index]
            snippets.append(f"[{index}] {text[:300]}{'...' if len(text) > 300 else ''}")
        return "\n\n".join(snippets)

    def read_document(self, doc_id: str) -> str:
        """Tool: full text of the passage whose id is ``doc_id``."""
        try:
            index = int(doc_id)
        except ValueError:
            return f"unknown document id: {doc_id}"
        if not 0 <= index < len(self.passages):
            return f"unknown document id: {doc_id}"
        return f"[{index}] {self.passages[index]}"

    # -- manual tool-calling loop ----------------------------------------
    @staticmethod
    def _strip_fence(text: str) -> str:
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_tool_call(self, text: str) -> tuple[str, dict] | None:
        """Parse a plain-text JSON tool invocation; None means 'final answer'."""
        candidate = self._strip_fence(text)
        try:
            obj = json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                return None
            try:
                obj = json.loads(match.group(0))
            except (ValueError, json.JSONDecodeError):
                return None
        if not isinstance(obj, dict):
            return None
        function = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        name = (obj.get("name") or obj.get("tool") or obj.get("action")
                or function.get("name"))
        if name not in ("search_documents", "read_document"):
            return None
        args = obj.get("arguments", obj.get("args", obj.get("action_input")))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, json.JSONDecodeError):
                args = {}
        return (name, args if isinstance(args, dict) else {})

    def _execute(self, name: str, args: dict) -> str:
        if name == "search_documents":
            return self.search_documents(str(args.get("query", "")))
        if name == "read_document":
            return self.read_document(str(args.get("doc_id", "")))
        return f"unknown tool: {name}"

    def answer(self, question: str) -> dict:
        """Run the loop; return {"answer", "trace", "llm_calls"}."""
        trace: list[tuple[str, str]] = []
        turns: list[str] = [self.SYSTEM_PROMPT, f"Question: {question}"]
        calls = 0
        for _ in range(self.max_tool_calls):  # at most max_tool_calls executions
            calls += 1
            raw = self.llm.invoke("\n\n".join(turns)).strip()
            parsed = self._extract_tool_call(raw)
            if parsed is None:
                return {"answer": raw, "trace": trace, "llm_calls": calls}
            name, args = parsed
            argument = args.get("query", args.get("doc_id", ""))
            trace.append((name, str(argument)))
            observation = self._execute(name, args)
            turns.append(f"Assistant:\n{raw}")
            turns.append(f"Tool result:\n{observation}")
        # Budget exhausted without a plain-text answer -> force a final answer,
        # giving the model the whole conversation (tool observations included).
        calls += 1
        final = self.llm.invoke(
            "\n\n".join(turns)
            + "\n\nAnswer the question now, using the tool results if any. "
            + "Respond with ONLY the answer."
        ).strip()
        return {"answer": final, "trace": trace, "llm_calls": calls}


def run_experiment() -> dict:
    passages = load_passages(PASSAGES_PATH, N_PASSAGES, PASSAGE_STRIDE)
    llm = OllamaLLM()  # local qwen2.5-coder:7b (json is parsed from plain text)
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device=BGE_DEVICE)

    agent = ToolCallingAgent(llm, passages, embedder)
    t0 = time.perf_counter()
    rows = [
        {"kind": kind, "question": question, **agent.answer(question)}
        for kind, question in QUESTIONS
    ]
    total_s = time.perf_counter() - t0

    return {
        "rows": rows,
        "n_passages": len(passages),
        "total_s": total_s,
        "total_tool_calls": sum(len(row["trace"]) for row in rows),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 10-01 — Tool-calling agent: retrieval as a model decision")
    print(f"{exp['n_passages']} passages indexed, run in {exp['total_s']:.1f}s")
    print("=" * 66)

    for i, row in enumerate(exp["rows"], start=1):
        print(f"\nQ{i} ({row['kind']}): {row['question']}")
        if row["trace"]:
            print(f"    tool trace   : {row['trace']}")
        else:
            print("    tool trace   : (none — answered from model knowledge)")
        print(f"    final answer : {row['answer'][:160]}")

    print(f"\n[4] Takeaway")
    print(f"    Total tool calls across the run: {exp['total_tool_calls']}.")
    print("    The no-retrieval question should have an empty trace: the model")
    print("    must NOT search when it already knows the answer. Retrieval is")
    print("    a decision the model makes per turn, not a fixed pipeline step.")
    print("    On a local 7B model a stray retrieval occasionally slips in —")
    print("    the verification gate tolerates that (trace <= 1).")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    rows = exp["rows"]
    known_tools = {"search_documents", "read_document"}

    checks.append(("all 3 questions produced a non-empty answer",
                   all(row["answer"].strip() for row in rows)))
    checks.append(("every recorded tool call names an existing tool",
                   all(name in known_tools for row in rows for name, _ in row["trace"])))
    checks.append((f"total tool calls <= {MAX_TOOL_CALLS * 3 + 1} "
                   f"(got {exp['total_tool_calls']})",
                   exp["total_tool_calls"] <= MAX_TOOL_CALLS * 3 + 1))
    no_retrieval = next(row for row in rows if row["kind"] == "no-retrieval")
    checks.append((f"no-retrieval question trace <= 1 (got "
                   f"{len(no_retrieval['trace'])})",
                   len(no_retrieval["trace"]) <= 1))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
