# EGW Research

Semantic search over the complete writings of Ellen G. White, powered by AI embeddings. Designed for use with AI coding agents (Claude Code, OpenAI Codex, Google Gemini) or direct command-line search.

## Requirements

- **Python 3.10+** — [python.org](https://python.org)
- **Docker** — [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)

### Platform Support

| Platform | Support |
|----------|---------|
| Linux | Native |
| macOS | Native |
| Windows | Native (Docker Desktop required) |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/DanielMunozT/egw-chat.git
cd egw-chat
```

### 2. Setup + download a language (one time)

```bash
python setup.py --lang en
```

This creates a Python virtual environment, installs dependencies, downloads the embedding model (~120MB), and downloads the English language package (books + search index).

Available languages: `en` (English), `es` (Spanish), `pt` (Portuguese), `fr` (French), `ko` (Korean).

To install multiple languages:

```bash
python setup.py --lang en,es,pt
```

### 3. Start the database

```bash
python start.py
```

This starts a Qdrant vector database in Docker and auto-restores the pre-built search index from the downloaded snapshots.

### 4. Search

```bash
source venv/bin/activate
source .env
python scripts/search.py "the love of God"
```

On Windows (PowerShell):
```powershell
venv\Scripts\activate
# Load .env manually or set: $env:QDRANT_URL="http://localhost:6333"
python scripts/search.py "the love of God"
```

### 5. Stop when done

```bash
python stop.py
```

## Updating

Code updates are delivered via git:

```bash
git pull
```

Language data packages (books + search index) rarely change. If a new version is available, re-run:

```bash
python setup.py --lang en
```

## Adding a Language Later

```bash
python setup.py --lang fr
```

This downloads and installs the French data package alongside your existing languages.

## Using with AI Agents

This package is designed to work with AI coding agents. Just open this directory in your agent of choice:

| Agent | Instructions File |
|-------|-------------------|
| Claude Code | `CLAUDE.md` (auto-detected) |
| OpenAI Codex | `AGENTS.md` (auto-detected) |
| Google Gemini | `GEMINI.md` |

The agent will automatically understand how to search and cite the corpus.

## Language Packages

Language packages contain the book text files and pre-built Qdrant snapshots for semantic search. They are downloaded automatically by `setup.py --lang`.

| Language | Code | Package |
|----------|------|---------|
| English | `en` | ~2.9 GB |
| Spanish | `es` | ~295 MB |
| Portuguese | `pt` | ~238 MB |
| French | `fr` | ~167 MB |
| Korean | `ko` | ~116 MB |

Direct download (if needed manually):
- https://munoz.tplinkdns.com/egw/packages/egw-research-en.tar.gz
- https://munoz.tplinkdns.com/egw/packages/egw-research-es.tar.gz
- https://munoz.tplinkdns.com/egw/packages/egw-research-pt.tar.gz
- https://munoz.tplinkdns.com/egw/packages/egw-research-fr.tar.gz
- https://munoz.tplinkdns.com/egw/packages/egw-research-ko.tar.gz

## Search Options

```
python scripts/search.py "query" [options]

Options:
  --lang LANG     Language: en, es, pt, fr, ko, or "all" (default: en)
  --book ABBR     Filter by book abbreviation (e.g., GC, DA, SC)
  --top-k N       Number of results (default: 8)
  --json          Output as JSON
  --qdrant-url    Qdrant URL (default: http://localhost:6333)
```

## Directory Structure

```
egw-chat/
├── setup.py                 # One-time setup + language download
├── start.py                 # Start Qdrant + restore snapshots
├── stop.py                  # Stop Qdrant
├── CLAUDE.md                # Claude Code instructions
├── AGENTS.md                # OpenAI Codex instructions
├── GEMINI.md                # Google Gemini instructions
├── egw_corpus/              # Python search library
├── scripts/search.py        # CLI search tool
├── chat.py                  # Offline chat with local LLM
├── books/<lang>/*.txt       # Full text of each book (downloaded)
└── snapshots/*.snapshot     # Pre-built vector DB snapshots (downloaded)
```

## Multi-User Setup

Multiple users on the same machine can share a single Qdrant instance (the database is read-only for search).

### Admin (first user)

1. Clone the repo and run setup: `python setup.py --lang en`
2. Run `python start.py` — Qdrant starts on port 6333
3. Share the URL with other users: `http://localhost:6333`

### Other users

1. Clone the repo: `git clone https://github.com/DanielMunozT/egw-chat.git`
2. Run `python setup.py` (no `--lang` needed — data comes from the shared Qdrant)
3. Set `QDRANT_URL=http://localhost:6333` in `.env` (pointing to the admin's instance)
4. Search: `python scripts/search.py "your query"`

## Offline Chat (Local LLM)

You can chat with the EGW corpus fully offline using a local AI model — no internet required after setup.

### Setup (requires internet, one time)

1. Install [Ollama](https://ollama.com):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sudo sh
   ```

2. Pull a model (choose based on your hardware):

   | Model | RAM needed | Best for |
   |-------|-----------|----------|
   | `gemma3:4b` (default) | ~5 GB | Best answer quality, slower |
   | `qwen2.5:3b` | ~4 GB | Faster responses, good quality |
   | `llama3.2:3b` | ~4 GB | Alternative option |

   ```bash
   ollama pull gemma3:4b
   ```

   **Choosing a model**: If you have 16+ GB RAM or a GPU, you can use larger models like `gemma3:12b` or `qwen2.5:7b` for better quality. With 8 GB RAM and no GPU, stick to the 3-4B models above.

### Usage

```bash
source venv/bin/activate
source .env
python chat.py
```

Use `--model` to try different models:

```bash
python chat.py --model qwen2.5:3b    # faster on limited hardware
```

## Troubleshooting

**Docker not running**: Start Docker Desktop or run `sudo systemctl start docker`.

**Port 6333 in use**: The start script auto-finds a free port. Check the output for the actual URL.

**No results**: Make sure Qdrant is running (`python start.py`) and the snapshot was restored. Check `docker logs qdrant-egw`.

**Slow first query**: The embedding model (~120MB) is downloaded on first use. Subsequent queries are fast (~200ms).

**Ollama not running**: If `chat.py` says Ollama is not running, start it with `ollama serve` or `sudo systemctl start ollama`.
