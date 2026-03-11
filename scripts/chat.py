#!/usr/bin/env python3
"""Interactive EGW research chat using a local LLM (Ollama).

Flow per question:
  1. LLM extracts 1-3 focused search queries from the user's question
  2. Each query is searched against the EGW corpus
  3. LLM summarizes the merged results with citations (streamed)

Usage:
    python scripts/chat.py                      # uses default model (gemma3:4b)
    python scripts/chat.py --model qwen2.5:3b   # faster, slightly lower quality
    python scripts/chat.py --model llama3.2:3b   # alternative
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
import warnings

import requests

warnings.filterwarnings("ignore", message=".*pooling.*fastembed.*")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_MODEL = "gemma3:4b"

EXTRACT_PROMPT = """\
Extract 1-3 short search queries from the user's question. These queries will \
be used for semantic search in a vector database of Ellen White's (EGW) writings.

Rules:
- Output ONLY a JSON array of strings, nothing else
- Each query should be 3-8 words, focused on key topical terms
- Do NOT include "Ellen White", "EGW", "she says", or similar — the entire \
database is her writings, so that's redundant
- Strip conversational phrasing ("what does she say about", "tell me about") \
— extract only the subject matter
- Use EGW-specific terminology when appropriate (e.g., "health reform" not \
just "healthy eating", "Great Controversy" not "end times battle", \
"spirit of prophecy", "three angels messages", "sanctuary service")
- For broad topics, generate queries that cover different angles
- Keep the same language as the user's question

Examples:
User: "What does Ellen White say about the Sabbath in the last days?"
["Sabbath keeping last days", "Sabbath seal of God", "Sabbath Sunday law"]

User: "what does she say about eating eggs?"
["eggs diet health reform"]

User: "health and diet"
["health reform temperance", "diet flesh food vegetarianism"]

User: "¿Qué dice sobre la segunda venida?"
["segunda venida de Cristo", "señales del fin"]

User: """

SYSTEM_PROMPT = """\
You are a research assistant for the writings of Ellen G. White (EGW).
You help users find and understand passages from her published works.

When I give you search results from her writings:
- Summarize the key findings, citing ONLY the book abbreviations and details \
that appear in the search results — do NOT invent or modify reference numbers
- Quote key phrases directly from the results using quotation marks
- Be concise — 2-4 short paragraphs, citing 2-4 key passages
- If the passages are in a non-English language, respond in that language
- Give a complete, self-contained answer — do NOT ask follow-up questions \
or offer to elaborate, as there is no follow-up conversation"""

# ---------------------------------------------------------------------------
# Search backend
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from egw_corpus.vector_store import QdrantIndexer  # noqa: E402

_indexer_cache: dict[str, QdrantIndexer] = {}


def get_indexer(lang: str = "en") -> QdrantIndexer:
    if lang not in _indexer_cache:
        _indexer_cache[lang] = QdrantIndexer(qdrant_url=QDRANT_URL, lang=lang)
    return _indexer_cache[lang]


def search_corpus(query: str, lang: str = "en", top_k: int = 5) -> list[dict]:
    """Run semantic search, return raw results."""
    indexer = get_indexer(lang)
    return indexer.search(query=query, limit=top_k)


def format_results(results: list[dict]) -> str:
    """Format search results for the LLM."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        book = meta.get("book_abbr", "?")
        title = meta.get("book_title", "?")
        chunk = meta.get("chunk_index", "?")
        score = r["score"]
        text = r["text"][:600]
        lines.append(f"[{i+1}] {book} — {title} (chunk {chunk}, score: {score:.3f})\n{text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------
class Spinner:
    def __init__(self, message: str = ""):
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        chars = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        while not self._stop.is_set():
            sys.stderr.write(f"\r\033[2m  {next(chars)} {self._message}\033[0m")
            sys.stderr.flush()
            self._stop.wait(0.1)
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def update(self, message: str):
        self._message = message

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def extract_queries(model: str, question: str) -> list[str]:
    """Ask the LLM to extract search queries from the user's question."""
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0},
    }, timeout=120)

    content = resp.json().get("message", {}).get("content", "").strip()

    # Handle models that wrap in markdown code blocks
    for prefix in ("```json\n", "```\n"):
        if content.startswith(prefix):
            content = content[len(prefix):]
            content = content.split("```")[0].strip()
            break

    try:
        queries = json.loads(content)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
    except json.JSONDecodeError:
        pass

    # Fallback: use the original question
    return [question]


def stream_answer(model: str, messages: list[dict]) -> str:
    """Stream the answer from the LLM, printing tokens as they arrive."""
    spinner = Spinner("Generating answer")
    spinner.start()

    t0 = time.time()
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 4096},
    }, timeout=300, stream=True)

    first_token = True
    full_content = []
    token_count = 0

    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        if token:
            if first_token:
                spinner.stop()
                first_token = False
            sys.stdout.write(token)
            sys.stdout.flush()
            full_content.append(token)
            token_count += 1
        if chunk.get("done"):
            if first_token:
                spinner.stop()
            break

    elapsed = time.time() - t0
    tok_s = token_count / elapsed if elapsed > 0 else 0
    print(f"\n\033[2m  [{token_count} tokens, {tok_s:.1f} tok/s, {elapsed:.1f}s]\033[0m")
    return "".join(full_content)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
LANG_NAMES = {
    "en": "English", "es": "Spanish", "pt": "Portuguese",
    "fr": "French", "ko": "Korean",
}


def detect_languages() -> list[str]:
    """Detect installed language collections from Qdrant."""
    try:
        resp = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        collections = resp.json().get("result", {}).get("collections", [])
        langs = []
        for c in collections:
            name = c["name"]
            if name.startswith("egw_corpus_"):
                lang = name.replace("egw_corpus_", "")
                if lang in LANG_NAMES:
                    langs.append(lang)
        return sorted(langs)
    except Exception:
        return ["en"]


def main():
    parser = argparse.ArgumentParser(description="Chat with EGW research assistant (local LLM)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Results per search query (default: 5)")
    args = parser.parse_args()

    model = args.model

    # Check Ollama
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.ConnectionError:
        print("Error: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # Check model
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    available = [m["name"] for m in resp.json().get("models", [])]
    if not any(model in m for m in available):
        print(f"Error: Model '{model}' not found. Available: {', '.join(available)}")
        print(f"Pull it with: ollama pull {model}")
        sys.exit(1)

    # Detect available languages and prompt user to select
    installed_langs = detect_languages()
    if not installed_langs:
        print("Error: No EGW collections found in Qdrant. Is the database running?")
        sys.exit(1)

    if len(installed_langs) == 1:
        lang = installed_langs[0]
    else:
        print("Available languages:")
        for i, l in enumerate(installed_langs, 1):
            print(f"  {i}. {LANG_NAMES.get(l, l)} ({l})")
        while True:
            try:
                choice = input(f"\nSelect language [1-{len(installed_langs)}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                sys.exit(0)
            # Accept number or language code
            if choice in installed_langs:
                lang = choice
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(installed_langs):
                    lang = installed_langs[idx]
                    break
            except ValueError:
                pass
            print(f"  Please enter 1-{len(installed_langs)} or a language code ({', '.join(installed_langs)})")

    print(f"\nEGW Research Assistant — {model} ({LANG_NAMES.get(lang, lang)})")
    print(f"Commands: 'quit' to exit, 'clear' to reset\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    spinner = Spinner()

    while True:
        try:
            user_input = input("\033[1mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(conversation cleared)\n")
            continue

        # Step 1: Extract search queries from the question
        spinner.update("Extracting search queries")
        spinner.start()
        queries = extract_queries(model, user_input)
        spinner.stop()

        for q in queries:
            print(f"\033[33m  → search: \"{q}\"\033[0m")

        # Step 2: Run all searches, merge and deduplicate
        spinner.update("Searching corpus")
        spinner.start()
        all_results = []
        seen_ids = set()
        for q in queries:
            results = search_corpus(q, lang=lang, top_k=args.top_k)
            for r in results:
                rid = r["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)
        all_results.sort(key=lambda r: r["score"], reverse=True)
        all_results = all_results[:args.top_k + 3]
        spinner.stop()

        print(f"\033[2m  {len(all_results)} passages found\033[0m")

        if not all_results:
            print("\nNo relevant passages found.\n")
            continue

        # Step 3: Build message with search results and stream answer
        formatted = format_results(all_results)
        user_msg = f"""Question: {user_input}

Search results from Ellen White's writings:

{formatted}

Answer the question based on these passages, citing the references."""

        messages.append({"role": "user", "content": user_msg})

        print(f"\n\033[1mAssistant:\033[0m ", end="", flush=True)
        response = stream_answer(model, messages)
        messages.append({"role": "assistant", "content": response})
        print()

    for idx in _indexer_cache.values():
        idx.close()


if __name__ == "__main__":
    main()
