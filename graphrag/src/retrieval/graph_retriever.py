"""Cypher-based graph traversal retriever."""
import logging
import re
from typing import List, Optional

import spacy

from src.utils.config import get_settings
from src.utils.neo4j_client import get_session, is_available

logger = logging.getLogger(__name__)

_nlp = None

ONE_HOP_QUERY = """
MATCH (e:Entity)-[r]-(neighbor:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
RETURN e.name AS source,
       type(r) AS relation,
       neighbor.name AS target,
       neighbor.type AS target_type,
       neighbor.description AS target_description,
       COALESCE(r.context, r.relation_label, '') AS context
LIMIT $limit
"""

SHORTEST_PATH_QUERY = """
MATCH (a:Entity), (b:Entity)
WHERE toLower(a.name) CONTAINS toLower($name_a)
  AND toLower(b.name) CONTAINS toLower($name_b)
MATCH path = shortestPath((a)-[*1..4]-(b))
UNWIND relationships(path) AS r
RETURN startNode(r).name AS source,
       type(r) AS relation,
       endNode(r).name AS target,
       COALESCE(r.context, r.relation_label, '') AS context
LIMIT $limit
"""

ENTITY_WITH_CHUNKS_QUERY = """
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE toLower(e.name) CONTAINS toLower($name)
RETURN e.name AS entity_name,
       e.type AS entity_type,
       e.description AS entity_description,
       c.text AS chunk_text,
       c.id AS chunk_id,
       c.filename AS filename
LIMIT $limit
"""


def _get_nlp():
    """Load spaCy model lazily."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found; falling back to simple NER")
            _nlp = None
    return _nlp


def extract_query_entities(query: str) -> List[str]:
    """Extract named entities from a query string using spaCy."""
    nlp = _get_nlp()
    if nlp is None:
        # Simple fallback: extract capitalized words
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        return list(set(words))

    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    # Also add noun chunks if no entities found
    if not entities:
        entities = [
            chunk.text
            for chunk in doc.noun_chunks
            if len(chunk.text) > 3
        ]
    return list(set(entities))


def retrieve_from_graph(
    query: str, top_k: int = 5
) -> List[dict]:
    """Retrieve graph context for a query.

    Returns list of graph triples and associated chunk context.
    Falls back to empty list if Neo4j is unavailable.
    """
    if not is_available():
        logger.warning("Neo4j unavailable; skipping graph retrieval")
        return []

    entities = extract_query_entities(query)
    logger.debug("Query entities extracted: %s", entities)

    if not entities:
        return []

    settings = get_settings()
    results = []

    with get_session() as session:
        # 1-hop retrieval for each entity
        for entity_name in entities[:3]:  # limit to first 3
            rows = session.run(
                ONE_HOP_QUERY,
                name=entity_name,
                limit=settings.graph_hop_limit,
            )
            for row in rows:
                results.append(
                    {
                        "type": "graph_triple",
                        "source": row["source"],
                        "relation": row["relation"],
                        "target": row["target"],
                        "context": row["context"],
                        "score": 0.8,  # base graph score
                    }
                )

        # Shortest path if multiple entities
        if len(entities) >= 2:
            rows = session.run(
                SHORTEST_PATH_QUERY,
                name_a=entities[0],
                name_b=entities[1],
                limit=settings.graph_hop_limit,
            )
            for row in rows:
                results.append(
                    {
                        "type": "path_triple",
                        "source": row["source"],
                        "relation": row["relation"],
                        "target": row["target"],
                        "context": row["context"],
                        "score": 0.9,  # paths are highly relevant
                    }
                )

        # Fetch associated chunks
        chunks_added = 0
        for entity_name in entities[:2]:
            rows = session.run(
                ENTITY_WITH_CHUNKS_QUERY,
                name=entity_name,
                limit=5,
            )
            for row in rows:
                results.append(
                    {
                        "type": "graph_chunk",
                        "entity_name": row["entity_name"],
                        "entity_type": row["entity_type"],
                        "text": row["chunk_text"],
                        "chunk_id": row["chunk_id"],
                        "filename": row["filename"],
                        "score": 0.75,
                    }
                )
                chunks_added += 1
                if chunks_added >= top_k:
                    break

    logger.info(
        "Graph retrieval for '%s': %d items", query[:50], len(results)
    )
    return results


def get_entity_neighborhood(entity_name: str) -> dict:
    """Return an entity and its neighbors as a JSON-serializable dict."""
    if not is_available():
        return {"error": "Neo4j unavailable"}

    with get_session() as session:
        rows = session.run(
            ONE_HOP_QUERY,
            name=entity_name,
            limit=50,
        )
        triples = [
            {
                "source": row["source"],
                "relation": row["relation"],
                "target": row["target"],
                "target_type": row["target_type"],
                "target_description": row["target_description"],
                "context": row["context"],
            }
            for row in rows
        ]

    return {
        "entity": entity_name,
        "neighbors": triples,
        "count": len(triples),
    }
