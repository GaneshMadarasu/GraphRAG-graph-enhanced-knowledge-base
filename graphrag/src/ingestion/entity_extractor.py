"""LLM-based entity and relationship extraction using GPT-4o."""
import json
import logging
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

from src.ingestion.chunker import Chunk
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {
    "PERSON", "ORG", "LOCATION", "CONCEPT", "EVENT", "TECHNOLOGY", "PRODUCT"
}

EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph builder. Extract structured information from text.

Return ONLY valid JSON (no markdown, no explanation) with this exact schema:
{
  "entities": [
    {"name": "string", "type": "PERSON|ORG|LOCATION|CONCEPT|EVENT|TECHNOLOGY|PRODUCT", "description": "string"}
  ],
  "relationships": [
    {"source": "string", "relation": "string", "target": "string", "context": "string"}
  ]
}

Rules:
- Entity names must be proper nouns or technical terms (not generic words)
- Relation labels should be SHORT (e.g., CREATED, WORKED_AT, INVENTED, INFLUENCED, PART_OF)
- Only include entities that appear in the text
- source and target in relationships must match entity names exactly
"""

EXTRACTION_USER_PROMPT = """Extract all named entities and relationships from the following text.

Text:
{chunk_text}"""


class EntityModel(BaseModel):
    """A named entity extracted from text."""

    name: str
    type: str
    description: str


class RelationshipModel(BaseModel):
    """A relationship between two entities."""

    source: str
    relation: str
    target: str
    context: str


class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction for a chunk."""

    chunk_id: str
    doc_id: str
    entities: List[EntityModel] = Field(default_factory=list)
    relationships: List[RelationshipModel] = Field(default_factory=list)


def extract_from_chunk(chunk: Chunk) -> ExtractionResult:
    """Extract entities and relationships from a single chunk using GPT-4o."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model=settings.extraction_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": EXTRACTION_USER_PROMPT.format(
                        chunk_text=chunk.text
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.error(
            "Extraction failed for chunk %s: %s", chunk.id, exc
        )
        data = {"entities": [], "relationships": []}

    # Validate and filter entities
    raw_entities = data.get("entities", [])
    entities = []
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        etype = e.get("type", "CONCEPT").upper()
        if etype not in VALID_ENTITY_TYPES:
            etype = "CONCEPT"
        entities.append(
            EntityModel(
                name=str(e.get("name", "")).strip(),
                type=etype,
                description=str(e.get("description", "")).strip(),
            )
        )

    # Build name set for relationship validation
    entity_names = {e.name for e in entities}

    raw_rels = data.get("relationships", [])
    relationships = []
    for r in raw_rels:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        if not src or not tgt:
            continue
        relationships.append(
            RelationshipModel(
                source=src,
                relation=str(r.get("relation", "RELATED_TO")).upper().replace(" ", "_"),
                target=tgt,
                context=str(r.get("context", "")).strip(),
            )
        )

    result = ExtractionResult(
        chunk_id=chunk.id,
        doc_id=chunk.doc_id,
        entities=entities,
        relationships=relationships,
    )
    logger.debug(
        "Chunk %s: %d entities, %d relationships",
        chunk.id,
        len(entities),
        len(relationships),
    )
    return result


def extract_from_chunks(chunks: List[Chunk]) -> List[ExtractionResult]:
    """Extract entities and relationships from multiple chunks."""
    results = []
    for i, chunk in enumerate(chunks):
        logger.info(
            "Extracting from chunk %d/%d (doc: %s)",
            i + 1,
            len(chunks),
            chunk.filename,
        )
        results.append(extract_from_chunk(chunk))
    return results
