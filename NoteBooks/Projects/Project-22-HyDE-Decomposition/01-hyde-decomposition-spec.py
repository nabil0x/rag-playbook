"""Project 22 -- HyDE & Query Decomposition -- notebook spec.

Source for src/scripts/gen_notebook.py. Generate the notebook with:

    python src/scripts/gen_notebook.py \
        NoteBooks/Projects/Project-22-HyDE-Decomposition/01-hyde-decomposition-spec.py \
        NoteBooks/Projects/Project-22-HyDE-Decomposition/01-hyde-decomposition.ipynb

Teaching style follows src/scripts/NOTEBOOK_TEMPLATE.md: teacher-first markdown
cells, one concept per code cell, real libraries (langchain + installed
deps, not the repo component modules). TODO comments mark the cells YOU
finish while executing the project.
"""

CELLS: list[dict] = [
    {
        "type": "md",
        "source": (
            "# Project 22 -- HyDE & Query Decomposition\n\n"
            "> Goal: two generate-then-retrieve strategies for complex "
            "questions -- HyDE embeds a hypothetical answer instead of the "
            "query, decomposition splits a multi-hop question into "
            "sub-questions and retrieves for each.\n\n"
            "Both wrap the same baseline retriever. No new dependencies."
        ),
    },
    {
        "type": "md",
        "source": (
            "## 0. Setup -- environment & imports\n\n"
            "Loads `.env` for the LLM key, then imports langchain pieces and the "
            "retriever stubs from `src/retrieval/`. No optional installs for this "
            "project -- you need an LLM to write hypothetical documents and "
            "sub-questions (Google AI key, or switch to OpenAI / local Ollama -- "
            "see the TODO below)."
        ),
    },
    {
        "type": "code",
        "source": (
            "import os\n"
            "from dotenv import load_dotenv\n\n"
            "load_dotenv()\n"
            "key = os.getenv(\"GOOGLE_API_KEY\", \"\")\n"
            "assert key, \"set GOOGLE_API_KEY in .env (see .env.example)\"\n"
            "print(\"LLM key ok:\", key[:4] + \"...\")\n"
            "# TODO: switch to your LLM of choice (llms/openai.py, local Ollama)"
        ),
    },
    {
        "type": "code",
        "source": (
            "from langchain_community.document_loaders import TextLoader\n"
            "from langchain_text_splitters import RecursiveCharacterTextSplitter\n"
            "from langchain_chroma import Chroma\n"
            "from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI\n\n"
            "# TODO: pick the corpus you'll eval on (Data/Waiting.txt is the classic)"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 1. Corpus & vector store\n\n"
            "Load, split, embed, store -- the Project 04 pattern, reused as-is. "
            "The multi-hop eval set is what matters here, not the pipeline."
        ),
    },
    {
        "type": "code",
        "source": (
            "loader = TextLoader(\"Data/Waiting.txt\")\n"
            "docs = loader.load()\n"
            "splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)\n"
            "chunks = splitter.split_documents(docs)\n"
            "print(f\"{len(chunks)} chunks\")\n\n"
            "embeddings = GoogleGenerativeAIEmbeddings(model=\"models/embedding-001\")\n"
            "store = Chroma.from_documents(chunks, embeddings)\n"
            "retriever = store.as_retriever(search_kwargs={\"k\": 5})\n"
            "print(\"vector store ready\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 2. Baseline -- retrieve the raw question\n\n"
            "A multi-hop question needs 2+ chunks stitched together, so plain "
            "top-k usually returns the single most-similar chunk and misses the "
            "rest. Measure the baseline before changing anything -- that number "
            "is what HyDE and decomposition have to beat."
        ),
    },
    {
        "type": "code",
        "source": (
            "# TODO: build 10-15 multi-hop questions as (question, gold_snippet_substring) pairs.\n"
            "# A multi-hop question is one NO single chunk can answer -- it needs 2+ chunks\n"
            "# stitched together, e.g. \"Who wrote Waiting, and in which year?\" Use 3-5 from\n"
            "# evaluation/golden.py and synthesize the rest from Data/Waiting.txt.\n"
            "EVAL = [\n"
            "    # (\"...\", \"...\"),\n"
            "]\n\n"
            "def hit_rate(retriever):\n"
            "    hits = 0\n"
            "    for q, gold in EVAL:\n"
            "        # langchain retrievers expose .invoke; the Project 22 wrappers\n"
            "        # expose .retrieve -- this helper covers both.\n"
            "        top = retriever.invoke(q) if hasattr(retriever, \"invoke\") else retriever.retrieve(q)\n"
            "        hits += any(gold in d.page_content for d in top)\n"
            "    return hits / len(EVAL)\n\n"
            "print(f\"baseline hit-rate: {hit_rate(retriever):.2f}\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 3. HyDE -- embed a hypothetical document\n\n"
            "Questions and source chunks live in different surface forms: the "
            "question never uses the chunk's vocabulary, so its embedding lands "
            "far away. HyDE's trick: the LLM writes a short hypothetical passage "
            "that would answer the question, in source-document style -- embedding "
            "THAT text lands near the real chunks. Finish "
            "`src/retrieval/hyde.py` (implement `_hypothetical`), then wrap the "
            "baseline retriever and re-run the same hit-rate measure."
        ),
    },
    {
        "type": "code",
        "source": (
            "from retrieval.hyde import HyDERetriever, HYDE_PROMPT\n\n"
            "llm = ChatGoogleGenerativeAI(model=\"gemini-2.0-flash\", temperature=0)\n"
            "hyde = HyDERetriever(hyde_llm=llm, retriever=retriever, top_k=5)\n\n"
            "# Show the hypothetical document for 3 eval questions before measuring:\n"
            "for q, _ in EVAL[:3]:\n"
            "    print(q, \"->\", hyde._hypothetical(q)[:100], \"...\")\n"
            "print(f\"hyde hit-rate: {hit_rate(hyde):.2f}\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 4. Decomposition -- retrieve per sub-question\n\n"
            "Split the multi-hop question into independent sub-questions, "
            "retrieve with the original AND each sub-question, dedupe. The "
            "DECOMPOSE_PROMPT asks for all sub-questions in one call (the "
            "parallel pattern); a sequential pattern would ask one sub-question "
            "at a time, using each answer to plan the next -- slower, but needed "
            "when facts depend on each other. Finish `src/retrieval/decompose.py` "
            "(implement `_decompose`), wrap the baseline retriever, measure again."
        ),
    },
    {
        "type": "code",
        "source": (
            "from retrieval.decompose import DecomposeRetriever, DECOMPOSE_PROMPT\n\n"
            "decomposer = DecomposeRetriever(decomposer_llm=llm, retriever=retriever, top_k=5)\n\n"
            "# Show the sub-questions for 3 eval questions before measuring:\n"
            "for q, _ in EVAL[:3]:\n"
            "    print(q, \"->\", decomposer._decompose(q))\n"
            "print(f\"decomposition hit-rate: {hit_rate(decomposer):.2f}\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 5. Answer quality spot-check\n\n"
            "Hit-rate only proves the right chunks were found. For 3-5 questions, "
            "generate an answer per strategy and judge quality -- eyeball it, or "
            "reuse your P20 LLMJudge. Multi-hop questions should produce "
            "noticeably more complete answers when both chunks were retrieved."
        ),
    },
    {
        "type": "code",
        "source": (
            "from langchain_core.prompts import ChatPromptTemplate\n\n"
            "prompt = ChatPromptTemplate.from_template(\n"
            "    \"Answer using ONLY the context.\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\"\n"
            ")\n\n"
            "# TODO: for 3-5 EVAL questions, build context from each strategy's top-k\n"
            "# and compare answers side by side (eyeball or P20 LLMJudge). Record\n"
            "# which strategy gives the most complete multi-hop answer."
        ),
    },
    {
        "type": "md",
        "source": (
            "## What you should notice\n\n"
            "- Which eval questions did HyDE fix? (indirect / paraphrased wording "
            "with no shared vocabulary with the chunk).\n"
            "- Which did decomposition fix? (facts spread across 2+ chunks).\n"
            "- Sequential vs parallel sub-questions -- when would you pay for the "
            "slower one?\n"
            "- Extra LLM latency per query (1 call for HyDE; 1 call + N retrieves "
            "for decomposition) -- is the hit-rate gain worth it?"
        ),
    },
    {
        "type": "md",
        "source": (
            "## Exercises\n\n"
            "- Combine both: decompose first, then run HyDE on each sub-question "
            "(and on the original).\n"
            "- Try sequential decomposition: one sub-question at a time, each "
            "planned from the previous answer.\n"
            "- Feed the sub-questions into Project 31's planner and compare the "
            "plans it produces."
        ),
    },
]
