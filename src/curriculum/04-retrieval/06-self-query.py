"""Lab 06 — Self-Query: an LLM writes the metadata filter.

Plain top-k retrieval (lab 01) has a blind spot: the vector index only sees
the *meaning* of the query, never its constraints. "Who founded Montevideo?"
and "Who founded Montevideo? but only from bucket b1" embed almost identically,
yet they ask for different answer sets. Metadata filtering answers that with
an explicit ``filter`` at search time:

    vs.similarity_search(query, k=K, filter={"bucket": "b1"})

…but someone has to *write* that filter. Self-querying turns the filter
author into an LLM: a small prompt builds a structured query

    query='Who founded Montevideo?' filter=Comparison(eq, bucket, b1)

from the natural-language sentence, and the store applies it. The retriever
then becomes "ask in English, get scoped results" — no hand-written filter
logic, no API change on the store side.

This lab builds an ephemeral (in-memory) Chroma store over a deterministic
subset of ``Data/corpus/rag-mini-wikipedia`` with synthetic metadata attached
to every passage (``bucket`` b1/b2/b3 + ``source``), then runs three paths:

* (a) a pure semantic query — the LLM parses NO filter, the store returns
  top-K docs from any bucket;
* (b) the same sentence plus a natural-language constraint ("from bucket
  b1") — the LLM parses a filter, the store returns ONLY b1 docs;
* (c) the same filter written by hand (a plain ``similarity_search`` with
  ``filter={"bucket": "b1"}``) — identical store behaviour, so the only
  difference is WHO wrote the filter: the human/API or the LLM.

Note on the implementation: langchain-classic 1.0.8's ``SelfQueryRetriever``
crashes when it tries to auto-detect the store's query translator
(``ImportError: cannot import name 'DatabricksVectorSearch' …``), so the
translator is passed explicitly (``ChromaTranslator()``) — the documented
workaround for this version.

Run from the repo root (needs ``GROQ_API_KEY`` in ``.env`` for the query
parser; embeddings are local BGE):

    python src/curriculum/04-retrieval/06-self-query.py
    python src/curriculum/04-retrieval/06-self-query.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/04-retrieval/06-self-query.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langchain_chroma import Chroma  # noqa: E402
from langchain_classic.chains.query_constructor.schema import AttributeInfo  # noqa: E402
from langchain_classic.retrievers import SelfQueryRetriever  # noqa: E402
from langchain_community.query_constructors.chroma import ChromaTranslator  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
# (Gemini alternative: from langchain_google_genai import ChatGoogleGenerativeAI)

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
N_PASSAGES = 60  # deterministic head of the 3200-passage corpus (keeps runtime low)
K = 5  # top-k for every search path
BUCKETS = ("b1", "b2", "b3")  # synthetic metadata buckets assigned round-robin
SOURCE_NAME = "rag-mini-wikipedia"
QUERY_PLAIN = "Who founded Montevideo?"  # (a) pure semantic — no filter expected
QUERY_FILTERED = "Who founded Montevideo? from bucket b1"  # (b) NL constraint
FILTER_BUCKET = "b1"  # the bucket requested in QUERY_FILTERED / path (c)
PREVIEW = 62  # max characters of passage text shown next to each hit
MAX_PARSE_ATTEMPTS = 2  # retries for the LLM filter-parse on path (b) only
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq query parser, never the embedder
# (Gemini alternative: LLM_MODEL = "gemini-2.5-flash" — needs GOOGLE_API_KEY in .env)
LLM_TEMPERATURE = 0.0


# --------------------------------------------------------------------------
# 2. Load — corpus passages + synthetic metadata buckets
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> list[str]:
    """Return the first ``n`` passage texts (deterministic, no randomness)."""
    df = pd.read_parquet(path)
    return df["passage"].head(n).tolist()


def attach_metadata(texts: list[str]) -> list[Document]:
    """Wrap passages in Documents with synthetic, deterministic metadata."""
    return [
        Document(
            page_content=text,
            metadata={
                # Round-robin buckets: b1, b2, b3, b1, b2, … (every third doc in b1)
                "bucket": BUCKETS[i % len(BUCKETS)],
                "source": SOURCE_NAME,
            },
        )
        for i, text in enumerate(texts)
    ]


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


def buckets_of(docs: list[Document]) -> list[str]:
    """The metadata bucket of every returned doc, in rank order."""
    return [d.metadata.get("bucket", "?") for d in docs]


# --------------------------------------------------------------------------
# 3. Experiment — embed, index, self-query; returns every artifact the demo
#    and the verification gate need (no re-computation between the two paths)
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    texts = load_passages(PASSAGES_PATH, N_PASSAGES)
    docs = attach_metadata(texts)

    # --- Embed + index into an EPHEMERAL (in-memory) Chroma store ----------
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME, encode_kwargs={"normalize_embeddings": True}
    )
    vs = Chroma(embedding_function=embedder)  # no persist_directory, no collection
    t0 = time.perf_counter()
    vs.add_documents(docs)
    index_s = time.perf_counter() - t0
    n_indexed = len(vs.get()["ids"])

    # --- The query parser: Groq turns natural language into a filter --------
    load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives at the repo root
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    # (Gemini alternative: llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    #  temperature=LLM_TEMPERATURE))

    fields = [
        AttributeInfo(
            name="bucket",
            description="the topic bucket label, one of b1, b2 or b3",
            type="string",
        ),
        AttributeInfo(
            name="source", description="the corpus source name", type="string"
        ),
    ]
    # ChromaTranslator passed explicitly: langchain-classic 1.0.8's auto
    # translator detection crashes on langchain-community 1.x (VERIFIED).
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vs,
        document_contents="financial FAQ passages from a Q&A corpus",
        metadata_field_info=fields,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": K},
    )

    # --- (a) Pure semantic query: no filter expected -----------------------
    t0 = time.perf_counter()
    parsed_plain = retriever.query_constructor.invoke({"query": QUERY_PLAIN})
    parse_plain_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    res_plain = retriever.invoke(QUERY_PLAIN)
    plain_s = time.perf_counter() - t0

    # --- (b) Same sentence + NL constraint: a filter must be emitted --------
    t0 = time.perf_counter()
    parsed_filtered = retriever.query_constructor.invoke({"query": QUERY_FILTERED})
    parse_filtered_s = time.perf_counter() - t0

    # The gate's contract depends on the LLM actually emitting the filter.
    # If a rare parse failure slips through, retry ONLY this invoke (max 2);
    # the gate itself stays honest and reports whatever the last run returned.
    t0 = time.perf_counter()
    res_filtered, parse_attempts = _invoke_filtered(
        retriever, QUERY_FILTERED, FILTER_BUCKET
    )
    filtered_s = time.perf_counter() - t0

    # --- (c) The same filter written by hand, no LLM involved ---------------
    t0 = time.perf_counter()
    res_plain_filter = vs.similarity_search(
        QUERY_PLAIN, k=K, filter={"bucket": FILTER_BUCKET}
    )
    plain_filter_s = time.perf_counter() - t0

    return {
        "texts": texts,
        "docs": docs,
        "retriever": retriever,
        "n_indexed": n_indexed,
        "index_s": index_s,
        "dim": len(embedder.embed_query("probe")),
        "parsed_plain": parsed_plain,
        "parsed_filtered": parsed_filtered,
        "res_plain": res_plain,
        "res_filtered": res_filtered,
        "res_plain_filter": res_plain_filter,
        "parse_plain_s": parse_plain_s,
        "parse_filtered_s": parse_filtered_s,
        "plain_s": plain_s,
        "filtered_s": filtered_s,
        "plain_filter_s": plain_filter_s,
        "parse_attempts": parse_attempts,
    }


def _invoke_filtered(retriever: SelfQueryRetriever, query: str, bucket: str):
    """invoke() retried (max MAX_PARSE_ATTEMPTS) until the filter lands.

    Without the filter the store returns mixed buckets; with it, every doc
    comes from ``bucket``. That observable difference is the retry signal.
    Returns ``(result, attempts)`` so the demo can say whether a retry was
    needed.
    """
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        res = retriever.invoke(query)
        if res and all(d.metadata.get("bucket") == bucket for d in res):
            return res, attempt
    return res, MAX_PARSE_ATTEMPTS


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 06 — Self-Query: an LLM writes the metadata filter")
    print(f"{EMBED_MODEL_NAME} | ephemeral Chroma | {LLM_MODEL} as query parser")
    print("=" * 66)

    print(f"\n[1] Corpus + synthetic metadata:")
    print(f"    {exp['n_indexed']} passages (first {N_PASSAGES} of 3200)")
    print(f"    metadata per doc: bucket in {BUCKETS} (round-robin) "
          f"+ source={SOURCE_NAME}")
    print(f"    e.g. [{exp['docs'][0].metadata}] {preview(exp['docs'][0].page_content)}")

    print(f"\n[2] Embed + index (ephemeral — gone at process exit):")
    print(f"    {exp['n_indexed']} passages embedded (dim {exp['dim']}) "
          f"and indexed in {exp['index_s']:.2f}s")

    print(f"\n[3] (a) Pure semantic query — no filter:")
    print(f'    query: "{QUERY_PLAIN}"')
    print(f"    parsed by LLM: {exp['parsed_plain']} ({exp['parse_plain_s']:.1f}s)")
    print(f"    returned: {len(exp['res_plain'])} docs, "
          f"buckets = {buckets_of(exp['res_plain'])}")
    print("      ^ filter=None -> store searched ALL buckets; see how the")
    print("        buckets above span more than one label.")

    print(f"\n[4] (b) Same sentence + natural-language constraint:")
    print(f'    query: "{QUERY_FILTERED}"')
    print(f"    parsed by LLM: {exp['parsed_filtered']} ({exp['parse_filtered_s']:.1f}s)")
    print(f"    returned: {len(exp['res_filtered'])} docs, "
          f"buckets = {buckets_of(exp['res_filtered'])}")
    print("      ^ the LLM emitted Comparison(eq, bucket, b1); the store")
    print("        applied it and every returned doc is from bucket b1.")
    if exp["parse_attempts"] > 1:
        print(f"      (filter parse needed {exp['parse_attempts']} attempts)")
    for rank, doc in enumerate(exp["res_filtered"], 1):
        print(f"      {rank}. [bucket {doc.metadata['bucket']}] {preview(doc.page_content)}")

    print(f"\n[5] (c) The same filter written by hand — no LLM:")
    print(f'    query: "{QUERY_PLAIN}", filter={{ "bucket": "{FILTER_BUCKET}" }}')
    print(f"    returned: {len(exp['res_plain_filter'])} docs, "
          f"buckets = {buckets_of(exp['res_plain_filter'])}")
    print("      ^ identical store behaviour — the only difference vs (b) is")
    print("        WHO wrote the filter: a human/API here, the LLM in (b).")

    print("\n[6] Takeaway")
    print("    The vector index cannot see constraints; metadata filters can,")
    print("    but someone must author them. Self-querying makes the LLM that")
    print("    author: one prompt maps 'from bucket b1' to a structured")
    print("    Comparison filter, and the store stays a plain filtered search.")
    print("    Cost: one extra LLM call per query — the trade for scoped,")
    print("    English-driven retrieval.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Store structure matches the config.
    checks.append(("embedding dimension is 768 (BGE base)", exp["dim"] == EMBED_DIM))
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["n_indexed"] == N_PASSAGES))

    # (a) Pure semantic query: top-K returned, visibly unfiltered (the store
    # was free to pick any bucket — at least two different buckets appear).
    buckets_a = buckets_of(exp["res_plain"])
    checks.append(("query (a) returns exactly K docs", len(exp["res_plain"]) == K))
    checks.append(
        ("query (a) is unfiltered: >= 2 distinct buckets in top-K",
         len(set(buckets_a)) >= 2)
    )

    # (b) The contract: the LLM emitted a filter AND the store honoured it.
    buckets_b = buckets_of(exp["res_filtered"])
    checks.append(
        ("query (b): LLM parsed a metadata filter",
         exp["parsed_filtered"].filter is not None)
    )
    checks.append(
        (f"query (b): >= 1 doc returned", len(exp["res_filtered"]) >= 1)
    )
    checks.append(
        (f"query (b): every doc is bucket {FILTER_BUCKET} (0 violations)",
         len(buckets_b) >= 1 and all(b == FILTER_BUCKET for b in buckets_b))
    )

    # (c) Hand-written filter: same constraint, same result shape.
    buckets_c = buckets_of(exp["res_plain_filter"])
    checks.append(
        (f"query (c): exactly K docs, all bucket {FILTER_BUCKET}",
         len(exp["res_plain_filter"]) == K
         and all(b == FILTER_BUCKET for b in buckets_c))
    )

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
