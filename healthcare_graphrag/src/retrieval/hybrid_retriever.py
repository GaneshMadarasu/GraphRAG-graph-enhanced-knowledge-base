"""
Hybrid retriever: merges vector search results with graph traversal results.

Score fusion formula (from spec):
  final_score = (0.4 × vector_score) + (0.4 × graph_hop_score) + (0.2 × credibility_score)

Falls back to vector-only if Neo4j is unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.retrieval.graph_retriever import MedicalGraphRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


# ── Entity extraction from free-text query ───────────────────────────────────

# Known entity classes for quick detection
_DRUG_HINTS = re.compile(
    r"\b(metformin|semaglutide|ozempic|warfarin|aspirin|furosemide|spironolactone|"
    r"insulin|tamoxifen|doxorubicin|lisinopril|atorvastatin|acetaminophen|ibuprofen|"
    r"acetylsalicylic acid|liraglutide|empagliflozin|dapagliflozin|canagliflozin)\b",
    re.IGNORECASE,
)
_DISEASE_HINTS = re.compile(
    r"\b(diabetes|t2dm|heart failure|hypertension|breast cancer|alzheimer|"
    r"atrial fibrillation|cancer|obesity|pancreatitis|lactic acidosis|"
    r"cardiovascular disease|chronic kidney disease)\b",
    re.IGNORECASE,
)
_GENE_HINTS = re.compile(
    r"\b(brca1|brca2|tp53|ampk|egfr|kras|braf|erbb2|apoe|ins)\b",
    re.IGNORECASE,
)


def _extract_entities_from_query(
    question: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (drugs, diseases, genes) found in the question."""
    drugs = list({m.group(0).lower() for m in _DRUG_HINTS.finditer(question)})
    diseases = list({m.group(0).lower() for m in _DISEASE_HINTS.finditer(question)})
    genes = list({m.group(0).upper() for m in _GENE_HINTS.finditer(question)})
    return drugs, diseases, genes


# ── Score fusion ──────────────────────────────────────────────────────────────

def _fuse_score(
    vector_score: float,
    graph_hop_score: float,
    credibility_score: float,
    cfg,
) -> float:
    return (
        cfg.vector_weight * vector_score
        + cfg.graph_weight * graph_hop_score
        + cfg.credibility_weight * credibility_score
    )


# ── Result normalisation ──────────────────────────────────────────────────────

def _normalise_graph_result(row: Dict[str, Any], source_type: str, cfg) -> Dict[str, Any]:
    """Convert a raw graph row into the unified result schema."""
    text = (
        row.get("chunk_text")
        or row.get("evidence")
        or str({k: v for k, v in row.items() if k not in {"graph_hop_score", "query_type"}})
    )
    credibility = cfg.get_credibility(source_type)
    graph_hop = float(row.get("graph_hop_score", 0.5))
    return {
        "text": text,
        "source_type": source_type,
        "title": row.get("title", "Graph result"),
        "date": row.get("date", ""),
        "chunk_id": row.get("chunk_id", ""),
        "doc_id": row.get("doc_id", ""),
        "vector_score": 0.0,
        "graph_hop_score": graph_hop,
        "credibility_score": credibility,
        "final_score": _fuse_score(0.0, graph_hop, credibility, cfg),
        "retrieval_type": "graph",
        "raw": row,
    }


# ── Hybrid retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(
        self,
        vector: VectorRetriever,
        graph: Optional[MedicalGraphRetriever],
    ) -> None:
        self._vector = vector
        self._graph = graph
        self._settings = get_settings()
        self._neo4j_available = graph is not None

    async def retrieve(
        self,
        question: str,
        top_k: int = 5,
        doc_types: Optional[List[str]] = None,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            results: [unified result dicts ranked by final_score],
            graph_entities_used: [entity names queried],
            warnings: [drug interaction warnings],
            neo4j_available: bool,
          }
        """
        cfg = self._settings
        warnings: List[str] = []
        graph_entities_used: List[str] = []
        all_results: List[Dict[str, Any]] = []

        # ── 1. Vector search ──────────────────────────────────────────────────
        vector_hits = self._vector.search(query=question, top_k=top_k * 2, doc_types=doc_types)
        for hit in vector_hits:
            v_score = float(hit.get("score", 0.5))
            credibility = float(hit.get("credibility_score", 0.5))
            fused = _fuse_score(v_score, 0.5, credibility, cfg)
            all_results.append(
                {
                    **hit,
                    "vector_score": v_score,
                    "graph_hop_score": 0.5,
                    "final_score": fused,
                    "retrieval_type": "vector",
                }
            )

        # ── 2. Graph traversal ────────────────────────────────────────────────
        neo4j_available = self._neo4j_available
        if self._graph:
            try:
                drugs, diseases, genes = _extract_entities_from_query(question)
                graph_entities_used = drugs + diseases + genes

                # Drug interactions (most critical — always surface)
                for drug in drugs:
                    interactions = await self._graph.get_drug_interactions(drug)
                    for row in interactions:
                        drug2 = row.get("drug2", "")
                        if drug2:
                            warnings.append(
                                f"⚠️ WARNING: Interaction between {drug.title()} and {drug2} detected"
                            )
                        all_results.append(
                            _normalise_graph_result(row, "research_paper", cfg)
                        )

                    contraindications = await self._graph.get_drug_contraindications(drug)
                    for row in contraindications:
                        disease = row.get("disease", "")
                        if disease:
                            warnings.append(
                                f"⚠️ WARNING: {drug.title()} is contraindicated for {disease}"
                            )
                        all_results.append(
                            _normalise_graph_result(row, "research_paper", cfg)
                        )

                # Gene–disease associations
                for disease in diseases:
                    gene_assocs = await self._graph.get_gene_disease_associations(disease)
                    for row in gene_assocs:
                        all_results.append(
                            _normalise_graph_result(row, "research_paper", cfg)
                        )

                    comorbidities = await self._graph.get_comorbidities(disease)
                    for row in comorbidities:
                        all_results.append(
                            _normalise_graph_result(row, "internal_report", cfg)
                        )

                # Cross-document evidence for all entities
                for entity in set(drugs + diseases + genes):
                    cross_docs = await self._graph.get_cross_document_evidence(entity)
                    for row in cross_docs:
                        src = row.get("doc_type", "research_paper")
                        all_results.append(_normalise_graph_result(row, src, cfg))

            except Exception as exc:
                logger.error("Graph retrieval failed (falling back to vector-only): %s", exc)
                neo4j_available = False
                warnings.append(
                    "⚠️ Graph database unavailable — results based on vector search only"
                )

        # ── 3. Deduplicate by chunk_id, keep highest score ────────────────────
        seen: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            key = r.get("chunk_id") or r.get("text", "")[:80]
            if key not in seen or r["final_score"] > seen[key]["final_score"]:
                seen[key] = r

        ranked = sorted(seen.values(), key=lambda x: x["final_score"], reverse=True)

        # ── 4. Filter by min_confidence proxy (final_score) ───────────────────
        filtered = [r for r in ranked if r["final_score"] >= min_confidence * 0.5]
        top_results = filtered[:top_k] if filtered else ranked[:top_k]

        return {
            "results": top_results,
            "graph_entities_used": list(set(graph_entities_used)),
            "warnings": warnings,
            "neo4j_available": neo4j_available,
        }
