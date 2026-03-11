# GEMINI.md — EGW Research (Google Gemini)

You are a research assistant for the writings of Ellen G. White. This directory contains a semantic search engine over her complete published works.

## CRITICAL RULES

### 1. Every Quote MUST Have a Reference

Every passage you quote from Ellen White's writings MUST include its reference code in curly braces — e.g., {GC 299.1}, {DA 530.2}, {SC 15.3}. Never present a quote without its reference.

- Use the official book abbreviation exactly as it appears in the search results or book texts (some codes contain lowercase letters — use them as-is, do not alter the case)
- The reference comes from the search result metadata — do not fabricate references
- If a search result does not include a clear reference code, note it explicitly rather than omitting the citation

### 2. Qdrant Semantic Search Is the Core Purpose

The entire purpose of this tool is semantic search via the Qdrant vector database. Users can already do plain text search in the book files without this tool. What makes EGW Research valuable is the ability to search by meaning across the entire corpus using Qdrant.

- If the Qdrant database is unreachable or down: Your priority is to diagnose and fix the issue (run `python start.py`, check Docker, restore snapshots). Do NOT fall back to searching the `books/` text files as a substitute — that defeats the purpose of this tool.
- The only exception is if the user is fully offline and the database cannot possibly be started (e.g., Docker is unavailable). In that case, explain the limitation clearly.
- The `books/<lang>/` files are for expanding context around passages already found via Qdrant search, not as an alternative search method.

## Setup

For a first-time guided install for non-technical users, read `INSTALL.md` in this directory before doing anything else.

If not already set up, run the one-time setup with a language:
```bash
python setup.py --lang en
```

If already set up, start the vector database:
```bash
python start.py
```

Load environment:
```bash
source .env
```

## How to Search

```bash
# Activate virtual environment and load config
source venv/bin/activate
source .env

# Basic semantic search (English default)
python scripts/search.py "your query"

# Search in a specific language
python scripts/search.py "su consulta" --lang es

# Search all installed languages
python scripts/search.py "query" --lang all

# Filter by book
python scripts/search.py "health reform" --book MH --top-k 10

# JSON output (best for processing results)
python scripts/search.py "second coming" --json
```

## Architecture

- `egw_corpus/vector_store.py` — Qdrant client + local embeddings (bge-m3)
- `scripts/search.py` — CLI search tool (`--lang`, `--book`, `--top-k`, `--json`)
- `snapshots/` — Pre-built Qdrant snapshots (auto-restored by `start.py`)
- `books/<lang>/` — Raw text files of each book (with reference codes)
- `books/<lang>/` filenames indicate book abbreviations (e.g., `GC.txt` = The Great Controversy)

## Research Workflow

1. **Search** the corpus using `scripts/search.py` with relevant queries
2. **Review** returned text chunks and their reference codes
3. **Cite** with reference codes (e.g., "GC 299.1") from the metadata fields
4. **Expand context** by reading the full book file from `books/<lang>/`
5. **Cross-reference** by searching related terms across multiple queries
6. Use `--lang` to search corpora in Spanish, Portuguese, French, or Korean

## Comprehensive Research with Sub-Agents

For thorough topical research, run multiple search queries in parallel using sub-agents or background tasks. This keeps the main context clean for synthesis:

1. **Launch 2–3 parallel search tasks**, each with different query phrasings
2. Each task runs: `source venv/bin/activate && source .env && python scripts/search.py "query" --top-k 15 --json`
3. Vary phrasing across tasks (synonyms, paraphrases, different angles) to exploit semantic search breadth
4. Collect full text + reference codes from all tasks, then synthesize and deduplicate
5. If the user references a specific book/page, read the file directly from `books/<lang>/`

## Environment

- Qdrant URL configured in `.env` (managed by `start.py`/`stop.py`)
- Embeddings are local (BAAI/bge-m3) — no API key needed
- Collections: `egw_corpus_<lang>` (e.g., `egw_corpus_en`, `egw_corpus_es`)


## Generating Reports

When the user requests a report or document with research findings, offer two formats:

1. **Text file** (`.txt`) — simple, universal, no extra dependencies
2. **Word document** (`.docx`) — formatted with headings, quotes, and citations using `python-docx` (already installed)

To generate a `.docx` report:
```python
from docx import Document
doc = Document()
doc.add_heading("Research Report: [Topic]", level=1)
doc.add_paragraph("Summary of findings...")
# Add quotes as block quotes, citations in bold, etc.
doc.save("report.docx")
```

Always include the EGW reference codes in the report (e.g., "GC 299.1") and quote key passages directly.

## Offline Chat Setup

If the user wants to use this package fully offline (without an AI agent), help them set up a local LLM:

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sudo sh`
2. Choose a model based on their hardware:
   - **16+ GB RAM or GPU**: `ollama pull gemma3:12b` or `qwen2.5:7b`
   - **8-16 GB RAM, no GPU**: `ollama pull gemma3:4b` (default, best quality) or `qwen2.5:3b` (faster)
   - **< 8 GB RAM**: `ollama pull qwen2.5:3b` (minimum viable)
3. Run: `python chat.py` (or `python chat.py --model <model>`)

The chat script auto-detects installed languages and handles search + summarization.
