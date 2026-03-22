"""
Medical entity + relationship extraction using GPT-4o.

Design:
- Uses the exact system prompt specified in requirements.
- Returns structured Pydantic models.
- Failures are logged and skipped — never breaks the pipeline.
- Async batch processing with configurable concurrency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from src.ingestion.chunker import Chunk
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# ── Allowed entity / relationship types ──────────────────────────────────────

ENTITY_TYPES = {
    "Disease", "Drug", "Gene", "Protein", "Symptom", "Treatment",
    "ClinicalTrial", "Patient", "Researcher", "Institution",
}

RELATIONSHIP_TYPES = {
    "TREATS", "INTERACTS_WITH", "CAUSES_SIDE_EFFECT", "CONTRAINDICATED_FOR",
    "ASSOCIATED_WITH", "ENCODES", "INVOLVED_IN", "COMORBID_WITH",
    "HAS_SYMPTOM", "USED_IN", "AUTHORED", "CONDUCTED", "MENTIONS", "PART_OF",
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    name: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ExtractedRelationship(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str = ""
    confidence: float = 0.5

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, v: str) -> str:
        if v not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unknown relationship type: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ExtractionResult(BaseModel):
    chunk_id: str
    doc_id: str
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    extraction_successful: bool = True
    error: Optional[str] = None


# ── System prompt (exact specification) ──────────────────────────────────────

_SYSTEM_PROMPT = """You are a medical knowledge extraction system. From the following clinical or \
research text, extract ALL medical entities and their relationships.

Return ONLY valid JSON:
{
  "entities": [
    {
      "name": "string",
      "type": "one of [Disease, Drug, Gene, Protein, Symptom, Treatment, ClinicalTrial, Researcher, Institution]",
      "attributes": {"key": "value pairs relevant to that type"},
      "confidence": "float 0.0-1.0"
    }
  ],
  "relationships": [
    {
      "source": "entity name",
      "relation": "one of [TREATS, INTERACTS_WITH, CAUSES_SIDE_EFFECT, CONTRAINDICATED_FOR, ASSOCIATED_WITH, ENCODES, INVOLVED_IN, COMORBID_WITH, HAS_SYMPTOM, USED_IN, AUTHORED, CONDUCTED, MENTIONS, PART_OF]",
      "target": "entity name",
      "evidence": "direct quote from text supporting this relationship (max 50 words)",
      "confidence": "float 0.0-1.0"
    }
  ]
}

Only include relationships you are confident about.
Always include the evidence quote."""


def _build_user_prompt(chunk_text: str) -> str:
    return f"Text: {chunk_text}"


# ── JSON parser (resilient) ───────────────────────────────────────────────────

def _parse_extraction_json(raw: str) -> Dict[str, Any]:
    """Extract JSON from GPT-4o response, stripping markdown fences."""
    raw = raw.strip()
    # Strip ```json ... ``` fences
    fence_m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if fence_m:
        raw = fence_m.group(1).strip()
    return json.loads(raw)


def _coerce_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure entities/relationships lists exist and types are strings."""
    entities = []
    for e in data.get("entities", []):
        if isinstance(e, dict):
            e["confidence"] = float(e.get("confidence", 0.5))
            entities.append(e)

    relationships = []
    for r in data.get("relationships", []):
        if isinstance(r, dict):
            r["confidence"] = float(r.get("confidence", 0.5))
            relationships.append(r)

    return {"entities": entities, "relationships": relationships}


# ── Extractor class ───────────────────────────────────────────────────────────

class MedicalEntityExtractor:
    """Async GPT-4o extractor.  Create once, call extract_chunk() per chunk."""

    def __init__(self, max_concurrency: int = 5) -> None:
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def extract_chunk(self, chunk: Chunk) -> ExtractionResult:
        async with self._semaphore:
            try:
                response = await self._client.chat.completions.create(
                    model=self._settings.openai_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(chunk.text)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or "{}"
                data = _coerce_extraction(_parse_extraction_json(raw))

                entities: List[ExtractedEntity] = []
                for e in data["entities"]:
                    try:
                        entities.append(ExtractedEntity(**e))
                    except Exception as ve:
                        logger.debug("Entity validation skip: %s | %s", e, ve)

                relationships: List[ExtractedRelationship] = []
                for r in data["relationships"]:
                    try:
                        relationships.append(ExtractedRelationship(**r))
                    except Exception as ve:
                        logger.debug("Relationship validation skip: %s | %s", r, ve)

                # Filter by confidence threshold
                threshold = self._settings.entity_extraction_confidence_threshold
                entities = [e for e in entities if e.confidence >= threshold]
                relationships = [r for r in relationships if r.confidence >= threshold]

                logger.debug(
                    "Chunk %s: %d entities, %d relationships",
                    chunk.chunk_id, len(entities), len(relationships),
                )
                return ExtractionResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    entities=entities,
                    relationships=relationships,
                )

            except Exception as exc:
                logger.error("Extraction failed for chunk %s: %s", chunk.chunk_id, exc)
                return ExtractionResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    extraction_successful=False,
                    error=str(exc),
                )

    async def extract_chunks(
        self, chunks: List[Chunk]
    ) -> List[ExtractionResult]:
        """Extract from all chunks concurrently (bounded by semaphore)."""
        tasks = [self.extract_chunk(c) for c in chunks]
        results = await asyncio.gather(*tasks)
        successful = sum(1 for r in results if r.extraction_successful)
        logger.info(
            "Extraction complete: %d/%d chunks successful", successful, len(results)
        )
        return list(results)
