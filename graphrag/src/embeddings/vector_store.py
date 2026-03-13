"""ChromaDB vector store for chunk embeddings."""
import logging
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from src.ingestion.chunker import Chunk
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

COLLECTION_NAME = "graphrag_chunks"


def _get_client() -> chromadb.PersistentClient:
    """Return (or create) the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client created at '%s'", settings.chroma_persist_dir)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Return (or create) the chunks collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", COLLECTION_NAME)
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using OpenAI text-embedding-3-small."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def upsert_chunks(chunks: List[Chunk]) -> None:
    """Embed and upsert chunks into ChromaDB."""
    if not chunks:
        return
    collection = get_collection()

    texts = [c.text for c in chunks]
    ids = [c.id for c in chunks]
    metadatas = [
        {
            "chunk_id": c.id,
            "doc_id": c.doc_id,
            "filename": c.filename,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    # Batch embed in groups of 100
    batch_size = 100
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(embed_texts(batch))
        logger.debug("Embedded batch %d-%d", i, i + len(batch))

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into ChromaDB", len(chunks))


def query_vector_store(
    query_text: str, top_k: int = 5
) -> List[dict]:
    """Query ChromaDB for the most similar chunks.

    Returns list of {chunk_id, text, score, metadata}.
    """
    collection = get_collection()
    embeddings = embed_texts([query_text])

    results = collection.query(
        query_embeddings=embeddings,
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    items = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        items.append(
            {
                "chunk_id": meta.get("chunk_id", ""),
                "text": doc,
                "score": float(1.0 - dist),  # cosine similarity
                "metadata": meta,
            }
        )
    return items


def get_chunk_count() -> int:
    """Return the number of chunks stored in ChromaDB."""
    return get_collection().count()


def is_available() -> bool:
    """Return True if ChromaDB is accessible."""
    try:
        get_collection()
        return True
    except Exception:
        return False
