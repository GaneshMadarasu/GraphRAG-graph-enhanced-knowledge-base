"""
Medical answer generator with:
- Per-source citations
- Drug interaction warnings (always surfaced)
- Confidence score 0–100%
- Medical safety disclaimer
- Hallucination guard (no drug dosages / clinical recommendations)

Uses Claude Sonnet with a system prompt that enforces safe, citable medical writing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "⚕️ DISCLAIMER: This information is for research purposes only. "
    "Always consult a licensed medical professional before making any "
    "clinical or treatment decisions."
)

_SYSTEM_PROMPT = """You are a medical research assistant generating evidence-based answers
from retrieved documents.

RULES:
1. ONLY use the provided context. Never invent drug dosages, clinical recommendations,
   or medical facts not present in the context.
2. Cite every fact with [Document: <title> (<type>)] inline.
3. Mark ALL drug interactions with ⚠️ WARNING prefix.
4. At the end include a Confidence: X% line (based on evidence quality and breadth).
5. If the context is insufficient, say "Insufficient evidence found across documents".
6. Separate findings by document type when multiple types are available.
7. Do NOT suggest specific doses, treatment plans, or diagnostic conclusions.

Context is tagged with [source_type: research_paper|internal_report|news_article|email_slack].
Treat research_paper as highest credibility, email_slack as lowest."""


def _build_context_block(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        src = r.get("source_type", "unknown")
        date = r.get("date", "")
        text = r.get("text", "")
        score = r.get("final_score", 0.0)
        blocks.append(
            f"[{i}] [source_type: {src}] {title} ({date}) [relevance: {score:.2f}]\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def _estimate_confidence(results: List[Dict[str, Any]], warnings: List[str]) -> float:
    if not results:
        return 0.0
    avg_score = sum(r.get("final_score", 0) for r in results) / len(results)
    # Count unique source types for breadth bonus
    src_types = {r.get("source_type") for r in results}
    breadth_bonus = min(0.1 * len(src_types), 0.2)
    raw = avg_score + breadth_bonus
    return round(min(max(raw, 0.0), 1.0), 3)


# ── Pydantic models ───────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    title: str
    type: str
    date: Optional[str] = None
    relevant_excerpt: str = ""


class AnswerResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_pct: int
    sources: List[SourceCitation] = Field(default_factory=list)
    graph_entities_used: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    neo4j_available: bool = True


# ── Generator ─────────────────────────────────────────────────────────────────

class MedicalAnswerGenerator:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)

    async def generate(
        self,
        question: str,
        retrieval_output: Dict[str, Any],
    ) -> AnswerResponse:
        results: List[Dict[str, Any]] = retrieval_output.get("results", [])
        warnings: List[str] = retrieval_output.get("warnings", [])
        graph_entities: List[str] = retrieval_output.get("graph_entities_used", [])
        neo4j_available: bool = retrieval_output.get("neo4j_available", True)

        confidence = _estimate_confidence(results, warnings)

        # Build source citations
        sources = []
        seen_titles: set = set()
        for r in results:
            t = r.get("title", "Unknown")
            if t not in seen_titles:
                seen_titles.add(t)
                sources.append(
                    SourceCitation(
                        title=t,
                        type=r.get("source_type", "unknown"),
                        date=r.get("date"),
                        relevant_excerpt=r.get("text", "")[:200],
                    )
                )

        # Confidence guard
        if confidence < 0.6 or not results:
            answer = (
                "Insufficient evidence found across documents to answer this question with confidence. "
                "Please ensure relevant documents have been ingested."
            )
            if warnings:
                answer = "\n".join(warnings) + "\n\n" + answer
            return AnswerResponse(
                answer=answer,
                confidence=confidence,
                confidence_pct=int(confidence * 100),
                sources=sources,
                graph_entities_used=graph_entities,
                warnings=warnings,
                neo4j_available=neo4j_available,
            )

        # Build prompt
        context = _build_context_block(results[:10])
        user_msg = (
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Drug interaction warnings from graph:\n"
            + ("\n".join(warnings) if warnings else "None detected")
        )

        try:
            response = await self._client.messages.create(
                model=self._settings.generation_model,
                max_tokens=1500,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_msg},
                ],
            )
            raw_answer = response.content[0].text or ""

            # Always prepend drug interaction warnings prominently
            if warnings:
                warning_block = "\n".join(warnings)
                raw_answer = f"{warning_block}\n\n{raw_answer}"

            # Append disclaimer
            final_answer = f"{raw_answer}\n\n{DISCLAIMER}"

        except Exception as exc:
            logger.error("Claude answer generation failed: %s", exc)
            final_answer = (
                "Answer generation failed due to an API error. "
                "Raw evidence is available in the sources below.\n\n" + DISCLAIMER
            )

        return AnswerResponse(
            answer=final_answer,
            confidence=confidence,
            confidence_pct=int(confidence * 100),
            sources=sources,
            graph_entities_used=graph_entities,
            warnings=warnings,
            neo4j_available=neo4j_available,
        )
