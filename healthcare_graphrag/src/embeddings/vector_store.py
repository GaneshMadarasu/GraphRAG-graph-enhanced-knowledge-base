"""
ChromaDB vector store with medical metadata filters.

Collection: healthcare_chunks
Document content: chunk text
Metadata stored per chunk:
  - chunk_id, doc_id, source_type, title, date, section
  - All extra source-specific fields (authors, journal, department, etc.)

Embedding: OpenAI text-embedding-3-small (via chromadb's embedding function wrapper)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AsyncOpenAI

from src.ingestion.chunker import Chunk
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


# ── OpenAI embedding function for ChromaDB ────────────────────────────────────

class OpenAIEmbeddingFunction:
    """Sync embedding function interface expected by ChromaDB."""

    def __init__(self, api_key: str, model: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        response = self._client.embeddings.create(
            model=self._model,
            input=input,
        )
        return [item.embedding for item in response.data]


# ── Vector store ──────────────────────────────────────────────────────────────

class MedicalVectorStore:
    """ChromaDB client wrapper with async-compatible helpers."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Optional[chromadb.HttpClient] = None
        self._collection = None
        self._embed_fn: Optional[OpenAIEmbeddingFunction] = None

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        cfg = self._settings
        try:
            self._client = chromadb.HttpClient(
                host=cfg.chroma_host,
                port=cfg.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._embed_fn = OpenAIEmbeddingFunction(
                api_key=cfg.openai_api_key,
                model=cfg.openai_embedding_model,
            )
            self._collection = self._client.get_or_create_collection(
                name=cfg.chroma_collection,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB connected: %s:%d | collection: %s",
                cfg.chroma_host, cfg.chroma_port, cfg.chroma_collection,
            )
        except Exception as exc:
            logger.error("ChromaDB connection failed: %s", exc)
            self._client = None
            raise

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Upsert chunks into ChromaDB."""
        self._ensure_connected()
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_type": chunk.source_type,
                "section": chunk.section or "",
            }
            # Flatten chunk metadata (all string/int/float/bool values)
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif isinstance(v, list):
                    meta[k] = ", ".join(str(x) for x in v)
                elif v is not None:
                    meta[k] = str(v)
            metadatas.append(meta)

        # ChromaDB upsert (idempotent)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("ChromaDB upserted %d chunks", len(chunks))

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_types: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search, optionally filtered by source_type."""
        self._ensure_connected()

        # Build where clause
        if doc_types and len(doc_types) == 1:
            filter_clause = {"source_type": {"$eq": doc_types[0]}}
        elif doc_types and len(doc_types) > 1:
            filter_clause = {"source_type": {"$in": doc_types}}
        else:
            filter_clause = where  # type: ignore[assignment]

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_clause:
            kwargs["where"] = filter_clause

        try:
            result = self._collection.query(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        hits: List[Dict[str, Any]] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance: 0=identical, 2=opposite → convert to similarity
            similarity = 1 - (dist / 2)
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": round(similarity, 4),
                    "source_type": meta.get("source_type", "unknown"),
                    "chunk_id": meta.get("chunk_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "date": meta.get("date", ""),
                }
            )
        return hits

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_count(self) -> int:
        try:
            self._ensure_connected()
            return self._collection.count()
        except Exception:
            return 0

    def health_check(self) -> bool:
        try:
            self._ensure_connected()
            self._collection.count()
            return True
        except Exception:
            return False

    def delete_collection(self) -> None:
        try:
            self._ensure_connected()
            self._client.delete_collection(self._settings.chroma_collection)
            self._collection = None
            logger.info("ChromaDB collection deleted")
        except Exception as exc:
            logger.error("Failed to delete ChromaDB collection: %s", exc)


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[MedicalVectorStore] = None


def get_vector_store() -> MedicalVectorStore:
    global _store
    if _store is None:
        _store = MedicalVectorStore()
    return _store
