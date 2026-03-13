"""Generate grounded answers using GPT-4o with merged context."""
import logging
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise knowledge assistant with access to a knowledge graph and document corpus.

Rules:
- Answer using ONLY the provided context
- Be concise and factual
- Always cite which entities or source documents you used
- If the context is insufficient, say so explicitly
- Do not hallucinate facts not in the context"""

USER_PROMPT = """Question: {query}

Context (from knowledge graph and document search):
{formatted_context}

Provide a clear, concise answer. At the end, list:
- Entities used: [comma-separated entity names]
- Source chunks: [comma-separated chunk IDs]"""


class AnswerResult(BaseModel):
    """The result of answer generation."""

    answer: str
    sources: List[str] = Field(default_factory=list)
    entities_used: List[str] = Field(default_factory=list)
    context_items_used: int = 0


def _format_context(context_items: List[dict]) -> str:
    """Format retrieved context items into a readable string."""
    parts = []
    for i, item in enumerate(context_items, 1):
        item_type = item.get("type", "unknown")
        score = item.get("score", 0.0)

        if item_type in ("vector_chunk", "graph_chunk"):
            text = item.get("text", "")
            filename = item.get("metadata", {}).get("filename") or item.get("filename", "")
            chunk_id = item.get("chunk_id", "")
            parts.append(
                f"[{i}] (score={score:.3f}, source={filename}, chunk={chunk_id[:8]}...)\n{text}"
            )
        elif item_type in ("graph_triple", "path_triple"):
            src = item.get("source", "")
            rel = item.get("relation", "")
            tgt = item.get("target", "")
            ctx = item.get("context", "")
            parts.append(
                f"[{i}] Graph fact (score={score:.3f}): {src} --[{rel}]--> {tgt}"
                + (f"\n    Context: {ctx}" if ctx else "")
            )

    return "\n\n".join(parts) if parts else "No context available."


def generate_answer(query: str, context_items: List[dict]) -> AnswerResult:
    """Generate a grounded answer from query + retrieved context."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    formatted = _format_context(context_items)

    try:
        response = client.chat.completions.create(
            model=settings.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        query=query, formatted_context=formatted
                    ),
                },
            ],
            temperature=0.1,
        )
        answer_text = response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        answer_text = f"Error generating answer: {exc}"

    # Extract sources and entities from answer
    chunk_ids = [
        item.get("chunk_id", "")
        for item in context_items
        if item.get("chunk_id")
    ]
    entities = list(
        set(
            item.get("entity_name", "") or item.get("source", "") or item.get("target", "")
            for item in context_items
            if item.get("type") in ("graph_triple", "path_triple", "graph_chunk")
        )
    )
    entities = [e for e in entities if e]

    return AnswerResult(
        answer=answer_text,
        sources=chunk_ids,
        entities_used=entities,
        context_items_used=len(context_items),
    )
