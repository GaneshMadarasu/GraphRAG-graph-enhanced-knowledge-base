"""Write extracted entities and relationships to Neo4j."""
import logging
from typing import List

from src.ingestion.chunker import Chunk
from src.ingestion.entity_extractor import ExtractionResult
from src.utils.neo4j_client import get_session

logger = logging.getLogger(__name__)

MERGE_ENTITY_QUERY = """
MERGE (e:Entity {name: $name, type: $type})
ON CREATE SET e.description = $description, e.created_at = timestamp()
ON MATCH SET e.description = CASE WHEN e.description = '' THEN $description ELSE e.description END
RETURN e
"""

MERGE_CHUNK_QUERY = """
MERGE (c:Chunk {id: $chunk_id})
ON CREATE SET
    c.text = $text,
    c.doc_id = $doc_id,
    c.filename = $filename,
    c.chunk_index = $chunk_index
RETURN c
"""

MERGE_ENTITY_CHUNK_REL_QUERY = """
MATCH (e:Entity {name: $name, type: $type})
MATCH (c:Chunk {id: $chunk_id})
MERGE (e)-[:MENTIONED_IN]->(c)
"""

MERGE_RELATIONSHIP_QUERY = """
MATCH (src:Entity {name: $source})
MATCH (tgt:Entity {name: $target})
CALL apoc.merge.relationship(src, $relation, {context: $context, chunk_id: $chunk_id}, {}, tgt, {}) YIELD rel
RETURN rel
"""

MERGE_RELATIONSHIP_FALLBACK_QUERY = """
MATCH (src:Entity {name: $source})
MATCH (tgt:Entity {name: $target})
MERGE (src)-[r:RELATED_TO {chunk_id: $chunk_id}]->(tgt)
ON CREATE SET r.relation_label = $relation, r.context = $context
RETURN r
"""


def store_chunk(chunk: Chunk) -> None:
    """Persist a chunk node in Neo4j."""
    with get_session() as session:
        session.run(
            MERGE_CHUNK_QUERY,
            chunk_id=chunk.id,
            text=chunk.text,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
        )


def store_extraction(result: ExtractionResult, chunk: Chunk) -> None:
    """Persist entities and relationships from an extraction result."""
    with get_session() as session:
        # Store entities
        for entity in result.entities:
            if not entity.name:
                continue
            session.run(
                MERGE_ENTITY_QUERY,
                name=entity.name,
                type=entity.type,
                description=entity.description,
            )
            session.run(
                MERGE_ENTITY_CHUNK_REL_QUERY,
                name=entity.name,
                type=entity.type,
                chunk_id=chunk.id,
            )

        # Store relationships
        for rel in result.relationships:
            if not rel.source or not rel.target:
                continue
            try:
                session.run(
                    MERGE_RELATIONSHIP_QUERY,
                    source=rel.source,
                    relation=rel.relation,
                    target=rel.target,
                    context=rel.context,
                    chunk_id=result.chunk_id,
                )
            except Exception:
                # Fallback without APOC if not available
                try:
                    session.run(
                        MERGE_RELATIONSHIP_FALLBACK_QUERY,
                        source=rel.source,
                        target=rel.target,
                        relation=rel.relation,
                        context=rel.context,
                        chunk_id=result.chunk_id,
                    )
                except Exception as exc2:
                    logger.warning(
                        "Could not store relationship %s->%s: %s",
                        rel.source,
                        rel.target,
                        exc2,
                    )


def build_graph(chunks: List[Chunk], results: List[ExtractionResult]) -> dict:
    """Store all chunks and extraction results. Returns summary stats."""
    chunk_map = {c.id: c for c in chunks}
    entity_set: set = set()
    rel_set: set = set()

    for i, result in enumerate(results):
        chunk = chunk_map.get(result.chunk_id)
        if chunk is None:
            continue
        store_chunk(chunk)
        store_extraction(result, chunk)
        for e in result.entities:
            entity_set.add((e.name, e.type))
        for r in result.relationships:
            rel_set.add((r.source, r.relation, r.target))
        logger.debug("Stored graph data for chunk %d/%d", i + 1, len(results))

    stats = {
        "chunks_stored": len(chunks),
        "unique_entities": len(entity_set),
        "unique_relationships": len(rel_set),
    }
    logger.info("Graph build complete: %s", stats)
    return stats
