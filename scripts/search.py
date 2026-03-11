#!/usr/bin/env python3
"""Search EGW writings by semantic similarity."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egw_corpus.vector_store import QdrantIndexer


def search_single(args, lang: str) -> list[dict]:
    """Search a single language collection."""
    indexer = QdrantIndexer(qdrant_url=args.qdrant_url, lang=lang)
    filters = {}
    if args.book:
        filters["book_abbr"] = args.book.upper()

    results = indexer.search(
        query=args.query,
        limit=args.top_k,
        must_match=filters if filters else None,
    )
    for r in results:
        r["metadata"]["lang"] = lang
    indexer.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Search EGW writings")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=8, help="Number of results (default: 8)")
    parser.add_argument("--book", help="Filter by book abbreviation (e.g. GC, DA, SC)")
    parser.add_argument("--lang", default="en",
                        help="Language code (en, es, pt, fr, ko) or 'all' for multi-lang search")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"),
                        help="Qdrant server URL (default: http://localhost:6333)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.lang == "all":
        # Discover available language collections
        probe = QdrantIndexer(qdrant_url=args.qdrant_url, lang="en")
        langs = probe.list_language_collections()
        probe.close()
        if not langs:
            print("No language collections found.")
            return
        # Search each, merge by score
        all_results = []
        for lang in langs:
            all_results.extend(search_single(args, lang))
        all_results.sort(key=lambda r: r["score"], reverse=True)
        results = all_results[: args.top_k]
    else:
        results = search_single(args, args.lang)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No results found.")
        return

    lang_label = args.lang
    print(f"\n{'='*80}")
    print(f"Query: {args.query}")
    if args.book:
        print(f"Filter: book={args.book}")
    print(f"Language: {lang_label}")
    print(f"Results: {len(results)}")
    print(f"{'='*80}\n")

    for i, r in enumerate(results):
        meta = r["metadata"]
        book = meta.get("book_abbr", "?")
        title = meta.get("book_title", "?")
        chunk = meta.get("chunk_index", "?")
        total = meta.get("total_chunks", "?")
        lang = meta.get("lang", "?")
        score = r["score"]
        text = r["text"]

        lang_tag = f" [{lang}]" if args.lang == "all" else ""
        print(f"[{i+1}] {book} - {title} (chunk {chunk}/{total}, score: {score:.4f}){lang_tag}")
        print(f"    {text[:300]}...")
        print()


if __name__ == "__main__":
    main()
