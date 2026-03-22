"""
Graph builder: writes extracted entities, relationships, chunks and documents
to Neo4j using MERGE to avoid duplicates.

All writes are batched for performance.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional

from src.ingestion.chunker import Chunk
from src.ingestion.document_loader import DocumentMetadata
from src.ingestion.entity_extractor import ExtractionResult, ExtractedEntity, ExtractedRelationship
from src.utils.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# ── Entity → Cypher MERGE statements ─────────────────────────────────────────

_ENTITY_MERGE: Dict[str, str] = {
    "Disease": """
        UNWIND $batch AS row
        MERGE (n:Disease {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Drug": """
        UNWIND $batch AS row
        MERGE (n:Drug {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Gene": """
        UNWIND $batch AS row
        MERGE (n:Gene {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Protein": """
        UNWIND $batch AS row
        MERGE (n:Protein {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Symptom": """
        UNWIND $batch AS row
        MERGE (n:Symptom {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Treatment": """
        UNWIND $batch AS row
        MERGE (n:Treatment {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "ClinicalTrial": """
        UNWIND $batch AS row
        MERGE (n:ClinicalTrial {trial_id: row.attributes.trial_id})
        ON CREATE SET n += row.attributes, n.name = row.name, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Patient": """
        UNWIND $batch AS row
        MERGE (n:Patient {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Researcher": """
        UNWIND $batch AS row
        MERGE (n:Researcher {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
    "Institution": """
        UNWIND $batch AS row
        MERGE (n:Institution {name: row.name})
        ON CREATE SET n += row.attributes, n.created_at = timestamp()
        ON MATCH  SET n += row.attributes
    """,
}

# ── Relationship MERGE ────────────────────────────────────────────────────────

_REL_MERGE = """
UNWIND $batch AS row
MATCH (src {name: row.source})
MATCH (tgt {name: row.target})
CALL apoc.merge.relationship(src, row.rel_type, {},
    {evidence: row.evidence, confidence: row.confidence, last_updated: timestamp()}, tgt)
YIELD rel
RETURN count(rel)
"""

# Fallback without APOC — per-relationship-type Cypher
_REL_MERGE_FALLBACK_TEMPLATE = """
UNWIND $batch AS row
MATCH (src {{name: row.source}})
MATCH (tgt {{name: row.target}})
MERGE (src)-[r:{rel_type}]->(tgt)
ON CREATE SET r.evidence = row.evidence, r.confidence = row.confidence,
              r.created_at = timestamp()
ON MATCH  SET r.evidence = row.evidence, r.confidence = row.confidence
"""

# ── Document + Chunk MERGE ────────────────────────────────────────────────────

_DOC_MERGE = """
UNWIND $batch AS row
MERGE (d:Document {id: row.doc_id})
ON CREATE SET d.title = row.title, d.type = row.source_type,
              d.date = row.date, d.file_path = row.file_path,
              d.department = row.department, d.journal = row.journal,
              d.doi = row.doi, d.publication = row.publication,
              d.created_at = timestamp()
ON MATCH SET  d.title = row.title, d.type = row.source_type,
              d.date = row.date
"""

_CHUNK_MERGE = """
UNWIND $batch AS row
MERGE (c:Chunk {id: row.chunk_id})
ON CREATE SET c.text = row.text, c.doc_id = row.doc_id,
              c.source_type = row.source_type, c.section = row.section,
              c.created_at = timestamp()
ON MATCH  SET c.text = row.text
WITH c, row
MATCH (d:Document {id: row.doc_id})
MERGE (c)-[:PART_OF]->(d)
"""

_CHUNK_MENTIONS = """
UNWIND $batch AS row
MATCH (c:Chunk {id: row.chunk_id})
MATCH (e {name: row.entity_name})
MERGE (c)-[r:MENTIONS]->(e)
ON CREATE SET r.confidence = row.confidence
"""


# ── Graph builder class ───────────────────────────────────────────────────────

class GraphBuilder:
    def __init__(self, client: Neo4jClient) -> None:
        self._client = client
        self._use_apoc: Optional[bool] = None

    async def _check_apoc(self) -> bool:
        if self._use_apoc is not None:
            return self._use_apoc
        try:
            await self._client.run_query("RETURN apoc.version() AS v")
            self._use_apoc = True
        except Exception:
            self._use_apoc = False
        return self._use_apoc  # type: ignore[return-value]

    # ── Public write methods ──────────────────────────────────────────────────

    async def upsert_document(self, meta: DocumentMetadata) -> None:
        row = {
            "doc_id": meta.doc_id,
            "title": meta.title,
            "source_type": meta.source_type,
            "date": meta.date,
            "file_path": meta.file_path,
            "department": meta.department,
            "journal": meta.journal,
            "doi": meta.doi,
            "publication": meta.publication,
        }
        await self._client.run_batch_write(_DOC_MERGE, [row])
        logger.debug("Upserted document %s", meta.doc_id)

    async def upsert_chunks(self, chunks: List[Chunk]) -> None:
        batch = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "source_type": c.source_type,
                "section": c.section,
            }
            for c in chunks
        ]
        await self._client.run_batch_write(_CHUNK_MERGE, batch)
        logger.debug("Upserted %d chunks", len(chunks))

    async def upsert_entities(self, entities: List[ExtractedEntity]) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for e in entities:
            if e.type not in _ENTITY_MERGE:
                continue
            attrs = {k: v for k, v in e.attributes.items() if v is not None}
            grouped.setdefault(e.type, []).append(
                {"name": e.name, "attributes": attrs}
            )
        for entity_type, batch in grouped.items():
            cypher = _ENTITY_MERGE[entity_type]
            await self._client.run_batch_write(cypher, batch)
        logger.debug("Upserted entities: %s", {k: len(v) for k, v in grouped.items()})

    async def upsert_relationships(
        self, relationships: List[ExtractedRelationship]
    ) -> None:
        use_apoc = await self._check_apoc()
        if use_apoc:
            batch = [
                {
                    "source": r.source,
                    "rel_type": r.relation,
                    "target": r.target,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                }
                for r in relationships
            ]
            try:
                await self._client.run_batch_write(_REL_MERGE, batch)
                return
            except Exception:
                logger.warning("APOC merge failed — falling back to per-type merge")

        # Fallback: group by relation type
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in relationships:
            grouped.setdefault(r.relation, []).append(
                {
                    "source": r.source,
                    "target": r.target,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                }
            )
        for rel_type, batch in grouped.items():
            cypher = _REL_MERGE_FALLBACK_TEMPLATE.format(rel_type=rel_type)
            try:
                await self._client.run_batch_write(cypher, batch)
            except Exception as exc:
                logger.error(
                    "Relationship upsert failed for %s: %s", rel_type, exc
                )

    async def upsert_chunk_mentions(
        self, chunk_id: str, entities: List[ExtractedEntity]
    ) -> None:
        batch = [
            {"chunk_id": chunk_id, "entity_name": e.name, "confidence": e.confidence}
            for e in entities
        ]
        if batch:
            try:
                await self._client.run_batch_write(_CHUNK_MENTIONS, batch)
            except Exception as exc:
                logger.error("Chunk mentions upsert failed: %s", exc)

    # ── Bulk ingestion orchestrator ───────────────────────────────────────────

    async def ingest_extraction_results(
        self,
        meta: DocumentMetadata,
        chunks: List[Chunk],
        results: List[ExtractionResult],
    ) -> Dict[str, int]:
        """Full pipeline: document → chunks → entities → rels → mentions."""
        await self.upsert_document(meta)
        await self.upsert_chunks(chunks)

        all_entities: List[ExtractedEntity] = []
        all_relationships: List[ExtractedRelationship] = []

        chunk_map = {c.chunk_id: c for c in chunks}

        for result in results:
            if not result.extraction_successful:
                continue
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)
            await self.upsert_chunk_mentions(result.chunk_id, result.entities)

        await self.upsert_entities(all_entities)
        await self.upsert_relationships(all_relationships)

        stats = {
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relationships": len(all_relationships),
        }
        logger.info("Graph ingest complete for %s: %s", meta.title, stats)
        return stats
