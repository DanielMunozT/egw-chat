"""Qdrant-based vector store for EGW corpus with local embeddings (bge-m3)."""
from __future__ import annotations

import os
import typing
import uuid

from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


@dataclass
class VectorDocument:
    document_id: str
    text: str
    metadata: dict[str, typing.Any]


class QdrantIndexer:
    """Qdrant vector index for EGW writings (connects to Docker Qdrant)."""

    COLLECTION_PREFIX = "egw_corpus"
    DEFAULT_QDRANT_URL = "http://localhost:6333"
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        embedding_model: str = "",
        vector_size: int = 0,
        qdrant_url: str = "",
        lang: str = "en",
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError(
                "qdrant-client is required. Install: pip install qdrant-client"
            )
        self.lang = lang.lower().strip()
        self.collection_name = f"{self.COLLECTION_PREFIX}_{self.lang}"
        self.embedding_model = (
            embedding_model
            or os.getenv("EGW_EMBEDDING_MODEL")
            or self.DEFAULT_MODEL
        ).strip()
        self.vector_size = int(vector_size or 0)
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "") or self.DEFAULT_QDRANT_URL
        self.client = QdrantClient(url=self.qdrant_url, timeout=60)

        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "fastembed is required. Install: pip install fastembed"
            ) from exc
        # Use GPU if available (fastembed-gpu with onnxruntime-gpu)
        providers = None
        try:
            import onnxruntime
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers = ["CUDAExecutionProvider"]
        except Exception:
            pass
        kwargs: dict[str, typing.Any] = {"model_name": self.embedding_model}
        if providers:
            kwargs["providers"] = providers
        self.embedder = TextEmbedding(**kwargs)

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

    def ensure_collection(self, vector_size: int) -> str:
        collection_name = self.collection_name
        existing = [c.name for c in self.client.get_collections().collections]
        vectors_config = qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        )
        quantization_config = qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                always_ram=True,
            ),
        )

        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                quantization_config=quantization_config,
            )
            return collection_name

        current_size = self._configured_vector_size(collection_name)
        if current_size and current_size != vector_size:
            self.client.delete_collection(collection_name=collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                quantization_config=quantization_config,
            )
        return collection_name

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = list(self.embedder.embed(texts))
        return [list(map(float, vec)) for vec in vectors]

    def upsert(self, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0
        vectors = self._embed_batch([d.text for d in documents])
        vec_size = len(vectors[0]) if vectors and vectors[0] else 0
        if not vec_size:
            raise RuntimeError("Embedding vector size is empty.")
        self.ensure_collection(vec_size)
        self.vector_size = vec_size
        points = []
        for i, doc in enumerate(documents):
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(doc.document_id),
                    vector=vectors[i],
                    payload={
                        "text": doc.text,
                        "source_id": doc.document_id,
                        "embedding_model": self.embedding_model,
                        **doc.metadata,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(
        self,
        query: str,
        limit: int = 8,
        must_match: typing.Optional[dict[str, typing.Any]] = None,
    ) -> list[dict[str, typing.Any]]:
        vector = self._embed_batch([query])[0]
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
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
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


# Backwards-compatible alias
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
