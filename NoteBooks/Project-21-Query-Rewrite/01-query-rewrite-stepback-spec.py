"""Project 21 — Query Rewrite & Step-Back — notebook spec.

Source for src/scripts/gen_notebook.py. Generate the notebook with:

    python src/scripts/gen_notebook.py \
        NoteBooks/Project-21-Query-Rewrite/01-query-rewrite-stepback-spec.py \
        NoteBooks/Project-21-Query-Rewrite/01-query-rewrite-stepback.ipynb

Teaching style follows src/scripts/NOTEBOOK_TEMPLATE.md: teacher-first markdown
cells, one concept per code cell, real libraries (langchain + installed
deps, not the repo component modules). TODO comments mark the cells YOU
finish while executing the project.
"""

CELLS: list[dict] = [
    {
        "type": "md",
        "source": (
            "# Project 21 — Query Rewrite & Step-Back\n\n"
            "> Goal: improve retrieval by rewriting the question before it is\n"
            "> embedded — LLM rewrite fixes vague phrasing, step-back abstracts\n"
            "> narrow questions into broader ones.\n\n"
            "Two wrapper retrievers, one comparison. No new dependencies."
        ),
    },
    {
        "type": "md",
        "source": (
            "## 0 · Setup — environment & imports\n\n"
            "Loads `.env` for the LLM key, then imports langchain pieces and the "
            "retriever stubs from `src/retrieval/`. No optional installs for this "
            "project."
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
            "print(\"LLM key ok:\", key[:4] + \"…\")\n"
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
            "## 1 · Corpus & vector store\n\n"
            "Load, split, embed, store — the Project 04 pattern, reused as-is. "
            "The eval set is what matters here, not the pipeline."
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
            "## 2 · Baseline — retrieve the raw question\n\n"
            "Measure the baseline before changing anything. You need a small "
            "hand-labeled eval set: question → the chunk(s) that must be found."
        ),
    },
    {
        "type": "code",
        "source": (
            "# TODO: define your eval set as (question, gold_snippet_substring) pairs.\n"
            "# Include deliberately vague ones: \"what about the second one?\",\n"
            "# \"how does it split?\", \"tell me more about that\"\n"
            "EVAL = [\n"
            "    # (\"...\", \"...\"),\n"
            "]\n\n"
            "def hit_rate(retriever):\n"
            "    hits = 0\n"
            "    for q, gold in EVAL:\n"
            "        top = [d.page_content for d in retriever.invoke(q)]\n"
            "        hits += any(gold in t for t in top)\n"
            "    return hits / len(EVAL)\n\n"
            "print(f\"baseline hit-rate: {hit_rate(retriever):.2f}\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 3 · Query rewrite wrapper\n\n"
            "Finish `src/retrieval/query_rewrite.py` (implement `_rewrite`), then "
            "wrap the baseline retriever and re-run the same hit-rate measure."
        ),
    },
    {
        "type": "code",
        "source": (
            "from retrieval.query_rewrite import QueryRewriteRetriever, REWRITE_PROMPT\n\n"
            "llm = ChatGoogleGenerativeAI(model=\"gemini-2.0-flash\", temperature=0)\n"
            "rewriter = QueryRewriteRetriever(rewriter_llm=llm, retriever=retriever, top_k=5)\n\n"
            "# Show what the rewrite does to 3 eval questions before measuring:\n"
            "for q, _ in EVAL[:3]:\n"
            "    print(q, \"->\", rewriter._rewrite(q))\n"
            "print(f\"rewrite hit-rate: {hit_rate(rewriter):.2f}\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 4 · Step-back wrapper\n\n"
            "Finish `src/retrieval/step_back.py` (implement `_step_back`), wrap the "
            "baseline retriever, measure again. Step-back should win on questions "
            "that need general background."
        ),
    },
    {
        "type": "code",
        "source": (
            "from retrieval.step_back import StepBackRetriever\n\n"
            "stepback = StepBackRetriever(stepback_llm=llm, retriever=retriever, top_k=5)\n"
            "print(f\"step-back hit-rate: {hit_rate(stepback):.2f}\")"
        ),
    },
    {
        "type": "md",
        "source": (
            "## 5 · Answer quality spot-check\n\n"
            "Hit-rate only proves the right chunk was found. For 3–5 questions, "
            "generate an answer per strategy and eyeball (or judge with your P20 "
            "LLMJudge) whether the answer improved."
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
            "# and compare answers side by side. Record what you see."
        ),
    },
    {
        "type": "md",
        "source": (
            "## What you should notice\n\n"
            "- Which query types did rewrite fix? (vague / pronoun-heavy).\n"
            "- Which query types were a no-op? (already-specific questions).\n"
            "- Did step-back ever beat rewrite, and on what?\n"
            "- Extra LLM latency per query — is the quality gain worth it?"
        ),
    },
    {
        "type": "md",
        "source": (
            "## Exercises\n\n"
            "- Combine rewrite + step-back (rewrite first, then step-back).\n"
            "- Add a cheap auto-decision: only rewrite if the query is short or "
            "contains a pronoun — the seed of prompt routing (Project 27).\n"
            "- Log how often the top-5 changed after rewriting (feeds Project 36)."
        ),
    },
]
