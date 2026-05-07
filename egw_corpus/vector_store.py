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


LocalQdrantIndexer = QdrantIndexer


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
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
