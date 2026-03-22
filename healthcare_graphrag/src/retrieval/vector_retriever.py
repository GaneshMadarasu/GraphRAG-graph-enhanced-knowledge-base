"""
Semantic (vector) retriever wrapping MedicalVectorStore.
Adds source credibility to each result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.embeddings.vector_store import MedicalVectorStore
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, store: MedicalVectorStore) -> None:
        self._store = store
        self._settings = get_settings()

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search with credibility annotation."""
        try:
            hits = self._store.search(query=query, top_k=top_k, doc_types=doc_types)
        except Exception as exc:
            logger.error("VectorRetriever.search failed: %s", exc)
            return []

        enriched = []
        for hit in hits:
            source_type = hit.get("source_type", "research_paper")
            credibility = self._settings.get_credibility(source_type)
            enriched.append(
                {
                    **hit,
                    "credibility_score": credibility,
                    "retrieval_type": "vector",
                }
            )
        return enriched

    def health_check(self) -> bool:
        return self._store.health_check()
