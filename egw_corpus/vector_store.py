"""Qdrant-based vector store for EGW corpus using OpenAI embeddings."""
from __future__ import annotations

import json
import os
import re
import typing
import uuid

from collections import OrderedDict
from dataclasses import dataclass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


def tokenizer() -> typing.Any:
    """Return the embedding tokenizer, with a lightweight fallback."""
    try:
        import tiktoken  # type: ignore

        try:
            return tiktoken.encoding_for_model(QdrantIndexer.DEFAULT_MODEL)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, encoder: typing.Any | None = None) -> int:
    if not text:
        return 0
    encoder = encoder if encoder is not None else tokenizer()
    if encoder is None:
        return len(re.findall(r"\S+", text))
    return len(encoder.encode(text))


def resolve_pagination(
    page_size: int,
    page: int = 1,
    offset: int | None = None,
) -> dict[str, int]:
    """Resolve page/page_size into a Qdrant offset."""
    page_size = max(1, min(int(page_size), 100))
    if offset is not None:
        offset = max(0, int(offset))
        page = (offset // page_size) + 1
    else:
        page = max(1, int(page))
        offset = (page - 1) * page_size
    return {
        "page_size": page_size,
        "page": page,
        "offset": offset,
    }


@dataclass
class VectorDocument:
    document_id: str
    text: str
    metadata: dict[str, typing.Any]


class QdrantIndexer:
    """Qdrant vector index for EGW writings backed by OpenAI embeddings."""

    COLLECTION_PREFIX = "egw_corpus"
    DEFAULT_QDRANT_URL = "http://localhost:6333"
    DEFAULT_MODEL = "text-embedding-3-large"
    DEFAULT_VECTOR_SIZE = 3072
    DEFAULT_QUERY_CACHE_SIZE = 512

    def __init__(
        self,
        embedding_model: str = "",
        vector_size: int = 0,
        qdrant_url: str = "",
        lang: str = "en",
        query_cache_size: int = DEFAULT_QUERY_CACHE_SIZE,
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError(
                "qdrant-client is required. Install: pip install qdrant-client"
            )
        if OpenAI is None:
            raise RuntimeError(
                "openai is required. Install: pip install openai"
            )
        self.lang = lang.lower().strip()
        self.collection_name = f"{self.COLLECTION_PREFIX}_{self.lang}"
        self.embedding_model = (
            embedding_model
            or os.getenv("EGW_EMBEDDING_MODEL")
            or self.DEFAULT_MODEL
        ).strip()
        self.vector_size = int(vector_size or os.getenv("EGW_EMBEDDING_DIMENSIONS") or self.DEFAULT_VECTOR_SIZE)
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "") or self.DEFAULT_QDRANT_URL
        self.client = QdrantClient(url=self.qdrant_url, timeout=120)
        self.openai_client = OpenAI()
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_size = max(1, int(query_cache_size))

    def _point_id(self, raw_id: str) -> str:
        try:
            uuid.UUID(str(raw_id))
            return str(raw_id)
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id)))

    def _configured_vector_size(self, collection_name: str) -> int:
        info = self.client.get_collection(collection_name)
        vectors = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        if isinstance(vectors, dict):
            for val in vectors.values():
                size = int(getattr(val, "size", 0) or 0)
                if size:
                    return size
        return int(getattr(vectors, "size", 0) or 0)

    def ensure_collection(self, vector_size: int | None = None, recreate: bool = False) -> str:
        collection_name = self.collection_name
        vector_size = int(vector_size or self.vector_size or self.DEFAULT_VECTOR_SIZE)
        existing = [c.name for c in self.client.get_collections().collections]
        vectors_config = qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        )

        if recreate and collection_name in existing:
            self.client.delete_collection(collection_name=collection_name)
            existing = [c.name for c in self.client.get_collections().collections]

        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            return collection_name

        current_size = self._configured_vector_size(collection_name)
        if current_size and current_size != vector_size:
            self.client.delete_collection(collection_name=collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
        return collection_name

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            dimensions=self.vector_size,
        )
        vectors: list[list[float]] = []
        for item in response.data:
            embedding = list(map(float, item.embedding))
            if len(embedding) != self.vector_size:
                raise RuntimeError(
                    f"Embedding vector size mismatch: expected {self.vector_size}, got {len(embedding)}"
                )
            vectors.append(embedding)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        key = json.dumps(
            [self.embedding_model, self.vector_size, self.lang, query.strip()],
            ensure_ascii=False,
        )
        cached = self._query_embedding_cache.get(key)
        if cached is not None:
            self._query_embedding_cache.move_to_end(key)
            return cached
        embedding = self._embed_batch([query])[0]
        self._query_embedding_cache[key] = embedding
        self._query_embedding_cache.move_to_end(key)
        while len(self._query_embedding_cache) > self._query_cache_size:
            self._query_embedding_cache.popitem(last=False)
        return embedding

    def upsert(self, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0
        vectors = self._embed_batch([d.text for d in documents])
        return self.upsert_embeddings(documents, vectors)

    def upsert_embeddings(
        self,
        documents: list[VectorDocument],
        vectors: list[list[float]],
    ) -> int:
        if not documents:
            return 0
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors length mismatch")
        for vector in vectors:
            if len(vector) != self.vector_size:
                raise RuntimeError(
                    f"Embedding vector size mismatch: expected {self.vector_size}, got {len(vector)}"
                )
        self.ensure_collection(self.vector_size)
        points = []
        for i, doc in enumerate(documents):
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(doc.document_id),
                    vector=vectors[i],
                    payload={
                        "text": doc.text,
                        "document_id": doc.document_id,
                        "source_id": doc.metadata.get("source_id", doc.document_id),
                        "embedding_provider": "openai",
                        "embedding_model": self.embedding_model,
                        "embedding_dimensions": self.vector_size,
                        **doc.metadata,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(
        self,
        query: str,
        page_size: int = 8,
        page: int = 1,
        offset: int | None = None,
        must_match: typing.Optional[dict[str, typing.Any]] = None,
    ) -> list[dict[str, typing.Any]]:
        page_info = resolve_pagination(page_size=page_size, page=page, offset=offset)
        vector = self.embed_query(query)
        query_filter = None
        if must_match:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    )
                    for key, value in must_match.items()
                ]
            )
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=page_info["page_size"],
                offset=page_info["offset"],
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=page_info["page_size"],
                offset=page_info["offset"],
                query_filter=query_filter,
                with_payload=True,
            )
            hits = list(getattr(resp, "points", []) or [])
        out: list[dict[str, typing.Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            out.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.pop("text", ""),
                    "metadata": payload,
                }
            )
        return out

    def search_page(
        self,
        query: str,
        page_size: int = 8,
        page: int = 1,
        offset: int | None = None,
        must_match: typing.Optional[dict[str, typing.Any]] = None,
    ) -> dict[str, typing.Any]:
        page_info = resolve_pagination(page_size=page_size, page=page, offset=offset)
        results = self.search(
            query=query,
            page_size=page_info["page_size"],
            page=page_info["page"],
            offset=page_info["offset"],
            must_match=must_match,
        )
        more_results_possible = len(results) == page_info["page_size"]
        return {
            "page_size": page_info["page_size"],
            "page": page_info["page"],
            "offset": page_info["offset"],
            "next_page": page_info["page"] + 1 if more_results_possible else None,
            "next_offset": page_info["offset"] + page_info["page_size"] if more_results_possible else None,
            "more_results_possible": more_results_possible,
            "results": results,
        }

    def list_language_collections(self) -> list[str]:
        """Return language codes for all egw_corpus_* collections."""
        prefix = f"{self.COLLECTION_PREFIX}_"
        return [
            c.name[len(prefix):]
            for c in self.client.get_collections().collections
            if c.name.startswith(prefix)
        ]

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def close(self) -> None:
        if self.client:
            self.client.close()


# Backwards-compatible alias
LocalQdrantIndexer = QdrantIndexer


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """Legacy fixed-size chunking helper."""
    if not text:
        return []
    normalized = " ".join(str(text).split())
    if len(normalized) <= chunk_size:
        return [normalized]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(normalized), step):
        end = start + chunk_size
        part = normalized[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(normalized):
            break
    return chunks


_REFCODE_PATTERN = re.compile(
    r"\[(?:\d*\w+, )?((?:Lt|Ms) \d+, \d{4}(?:, par\. \d+)?)\]"
    r"|"
    r"\[(\d*[A-Z]\w* [\divxlc][\w.]*)\]"
)


def _extract_refcodes(text: str) -> list[str]:
    refcodes: list[str] = []
    for match in _REFCODE_PATTERN.finditer(text):
        ref = (match.group(1) or match.group(2) or "").strip()
        if ref and ref not in refcodes:
            refcodes.append(ref)
    return refcodes


def _strip_refcodes(text: str) -> str:
    return _REFCODE_PATTERN.sub("", text)


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00A0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_block_text(text: str) -> str:
    text = _strip_refcodes(text)
    text = re.sub(r"^#\s*(?:Abbreviation|Author):.*$", "", text, flags=re.M)
    text = re.sub(r"^#\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^(Item|Chapter|Section)\s+\d+[A-Za-z.\-]*\b", stripped):
        return True
    words = stripped.split()
    return (
        len(words) <= 12
        and len(stripped) <= 90
        and stripped.upper() == stripped
        and any(ch.isalpha() for ch in stripped)
    )


def _iter_blocks(text: str) -> list[dict[str, typing.Any]]:
    lines = _normalize_text(text).split("\n")
    blocks: list[dict[str, typing.Any]] = []
    current: list[str] = []
    start_line = 1
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if not current:
                start_line = idx
            current.append(line)
            continue
        if current:
            blocks.append(
                {
                    "text": "\n".join(current),
                    "line_start": start_line,
                    "line_end": idx - 1,
                }
            )
            current = []
    if current:
        blocks.append(
            {
                "text": "\n".join(current),
                "line_start": start_line,
                "line_end": len(lines),
            }
        )
    return blocks


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?;:"")\]])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_to_token_windows(text: str, max_tokens: int, encoder: typing.Any | None) -> list[str]:
    if not text:
        return []
    encoder = encoder if encoder is not None else tokenizer()
    if encoder is None:
        words = text.split()
        return [
            " ".join(words[i:i + max_tokens]).strip()
            for i in range(0, len(words), max_tokens)
            if words[i:i + max_tokens]
        ]
    token_ids = encoder.encode(text)
    windows = []
    for i in range(0, len(token_ids), max_tokens):
        windows.append(encoder.decode(token_ids[i:i + max_tokens]).strip())
    return [window for window in windows if window]


def _split_oversized_segment(text: str, max_tokens: int, encoder: typing.Any | None) -> list[str]:
    if count_tokens(text, encoder) <= max_tokens:
        return [text.strip()]
    parts: list[str] = []
    current: list[str] = []
    for sentence in _split_sentences(text):
        candidate = " ".join(current + [sentence]).strip()
        if current and count_tokens(candidate, encoder) > max_tokens:
            parts.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(" ".join(current).strip())
    final_parts: list[str] = []
    for part in parts:
        if count_tokens(part, encoder) <= max_tokens:
            final_parts.append(part)
        else:
            final_parts.extend(_split_to_token_windows(part, max_tokens, encoder))
    return [part for part in final_parts if part]


def _compose_chunk_text(segments: list[dict[str, typing.Any]]) -> str:
    if not segments:
        return ""
    heading_path = list(segments[0].get("heading_path", []) or [])
    prefix = " > ".join(heading_path[-3:])
    body = "\n\n".join(seg["text"] for seg in segments if seg.get("text"))
    if prefix and body:
        return f"{prefix}\n\n{body}"
    if prefix:
        return prefix
    return body


def _build_chunk(segments: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    chunk_text = _compose_chunk_text(segments)
    heading_path = list(segments[0].get("heading_path", []) or []) if segments else []
    prefix = " > ".join(heading_path[-3:])
    cursor = len(prefix) + 2 if prefix and chunk_text.startswith(prefix + "\n\n") else 0
    spans: list[dict[str, typing.Any]] = []
    refcodes: list[str] = []
    for index, seg in enumerate(segments):
        if index > 0:
            cursor += 2
        seg_start = cursor
        cursor += len(seg["text"])
        for ref in seg.get("refcodes", []):
            if ref not in refcodes:
                refcodes.append(ref)
            spans.append({"ref": ref, "start": seg_start, "end": cursor})
    return {
        "text": chunk_text,
        "refcodes": refcodes,
        "refcode_spans": spans,
        "heading_path": heading_path,
        "chunk_line_start": min(seg["line_start"] for seg in segments),
        "chunk_line_end": max(seg["line_end"] for seg in segments),
    }


def _tail_overlap_segments(
    segments: list[dict[str, typing.Any]],
    overlap_tokens: int,
    encoder: typing.Any | None,
) -> list[dict[str, typing.Any]]:
    if not segments or overlap_tokens <= 0:
        return []
    tail: list[dict[str, typing.Any]] = []
    for seg in reversed(segments):
        if tail and seg.get("heading_path") != tail[0].get("heading_path"):
            break
        tail.insert(0, dict(seg))
        if count_tokens(_compose_chunk_text(tail), encoder) >= overlap_tokens:
            break
    return tail


def chunk_paragraphs(
    text: str,
    chunk_tokens: int = 800,
    overlap_tokens: int = 400,
) -> list[dict[str, typing.Any]]:
    """Token-based chunking that preserves refcodes and line ranges."""
    if not text:
        return []

    encoder = tokenizer()
    heading_path: list[str] = []
    segments: list[dict[str, typing.Any]] = []

    for block in _iter_blocks(text):
        raw_block = block["text"]
        stripped = raw_block.strip()
        if not stripped:
            continue
        if _is_heading(stripped):
            clean_heading = _clean_block_text(stripped)
            if clean_heading:
                heading_path.append(clean_heading)
                heading_path = heading_path[-3:]
            continue

        clean_text = _clean_block_text(raw_block)
        if not clean_text:
            continue
        refcodes = _extract_refcodes(raw_block)
        for part in _split_oversized_segment(clean_text, chunk_tokens, encoder):
            segments.append(
                {
                    "text": part,
                    "refcodes": refcodes,
                    "line_start": block["line_start"],
                    "line_end": block["line_end"],
                    "heading_path": list(heading_path),
                }
            )

    chunks: list[dict[str, typing.Any]] = []
    current: list[dict[str, typing.Any]] = []
    previous_text = ""

    for seg in segments:
        if current and seg.get("heading_path") != current[0].get("heading_path"):
            chunk = _build_chunk(current)
            if chunk["text"] and chunk["text"] != previous_text:
                chunks.append(chunk)
                previous_text = chunk["text"]
            current = []

        candidate = current + [seg]
        if current and count_tokens(_compose_chunk_text(candidate), encoder) > chunk_tokens:
            chunk = _build_chunk(current)
            if chunk["text"] and chunk["text"] != previous_text:
                chunks.append(chunk)
                previous_text = chunk["text"]
            current = _tail_overlap_segments(current, overlap_tokens, encoder)
            if current and seg.get("heading_path") != current[0].get("heading_path"):
                current = []
            while current and count_tokens(_compose_chunk_text(current + [seg]), encoder) > chunk_tokens:
                current = current[1:]
            candidate = current + [seg]
        current = candidate

    if current:
        chunk = _build_chunk(current)
        if chunk["text"] and chunk["text"] != previous_text:
            chunks.append(chunk)

    return chunks
