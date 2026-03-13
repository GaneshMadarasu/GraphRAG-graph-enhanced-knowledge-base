"""Semantic similarity vector retriever."""
import logging
from typing import List

from src.embeddings.vector_store import query_vector_store

logger = logging.getLogger(__name__)


def retrieve_from_vector(query: str, top_k: int = 5) -> List[dict]:
    """Retrieve top-K similar chunks from the vector store.

    Returns list of {chunk_id, text, score, metadata, type}.
    """
    try:
        results = query_vector_store(query, top_k=top_k)
        # Tag each result with retrieval type
        for r in results:
            r["type"] = "vector_chunk"
        logger.info("Vector retrieval: %d results for '%s'", len(results), query[:50])
        return results
    except Exception as exc:
        logger.error("Vector retrieval failed: %s", exc)
        return []
