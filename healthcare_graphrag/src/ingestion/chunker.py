"""
Medical-aware chunker.

Key design decisions:
- Sentence-boundary splitting: never cut mid-sentence — critical for clinical text.
- Section header preservation: each chunk gets its parent section label in metadata.
- Abstract is always kept as one atomic chunk (peer-reviewed papers).
- Overlap is token-aware and defaults to 150 chars (≈25–30 tokens).
"""

from __future__ import annotations

import hashlib
import re
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.ingestion.document_loader import LoadedDocument, DocumentMetadata
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page_number: Optional[int] = None
    section: Optional[str] = None          # e.g. "Introduction", "Results"
    source_type: str                        # propagated from parent document
    metadata: Dict = Field(default_factory=dict)


# ── Section header patterns ───────────────────────────────────────────────────

_SECTION_PATTERNS = re.compile(
    r"^(?:abstract|introduction|background|methods?|results?|discussion|"
    r"conclusion|references?|appendix|acknowledgements?|case report|"
    r"patient demographics?|readmission|drug interactions?|"
    r"side effects?|summary|findings?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _detect_section(text: str) -> Optional[str]:
    """Return the first section header found at the start of a block."""
    m = _SECTION_PATTERNS.match(text.strip())
    return m.group(0).strip().title() if m else None


# ── Sentence splitter (no nltk dependency) ───────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(])")


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Core chunking logic ───────────────────────────────────────────────────────

def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
    h = hashlib.md5(text[:100].encode()).hexdigest()[:8]
    return f"{doc_id}_{index}_{h}"


def _chunk_text(
    text: str,
    doc_id: str,
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: Dict,
) -> List[Chunk]:
    """Split *text* into overlapping sentence-boundary-aligned chunks."""
    chunks: List[Chunk] = []

    # Split on paragraph breaks first
    paragraphs = re.split(r"\n{2,}", text)

    buffer = ""
    current_section: Optional[str] = None
    chunk_index = 0

    def flush(buf: str, section: Optional[str]) -> None:
        nonlocal chunk_index
        buf = buf.strip()
        if not buf:
            return
        cid = _make_chunk_id(doc_id, chunk_index, buf)
        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                text=buf,
                section=section,
                source_type=source_type,
                metadata=base_metadata,
            )
        )
        chunk_index += 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect section headers
        sec = _detect_section(para)
        if sec:
            current_section = sec

        # Special case: abstract → one atomic chunk
        if current_section and current_section.lower() == "abstract":
            flush(para, current_section)
            continue

        # Accumulate sentences
        sentences = _split_sentences(para)
        for sent in sentences:
            if len(buffer) + len(sent) + 1 <= chunk_size:
                buffer = (buffer + " " + sent).strip()
            else:
                if buffer:
                    flush(buffer, current_section)
                    # Overlap: keep tail of buffer
                    overlap_text = buffer[-chunk_overlap:] if len(buffer) > chunk_overlap else buffer
                    buffer = (overlap_text + " " + sent).strip()
                else:
                    # Sentence itself is longer than chunk_size — force split
                    for i in range(0, len(sent), chunk_size):
                        flush(sent[i : i + chunk_size], current_section)

    if buffer:
        flush(buffer, current_section)

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(document: LoadedDocument) -> List[Chunk]:
    """Chunk a loaded document into ``Chunk`` objects."""
    cfg = get_settings()
    meta = document.metadata

    base_metadata = {
        "doc_id": meta.doc_id,
        "title": meta.title,
        "source_type": meta.source_type,
        "date": meta.date,
        "file_path": meta.file_path,
    }
    # Enrich metadata by source type
    if meta.source_type == "research_paper":
        base_metadata.update(
            {
                "authors": meta.authors,
                "journal": meta.journal,
                "doi": meta.doi,
            }
        )
    elif meta.source_type == "internal_report":
        base_metadata.update(
            {
                "department": meta.department,
                "patient_cohort": meta.patient_cohort,
            }
        )
    elif meta.source_type == "news_article":
        base_metadata.update(
            {
                "publication": meta.publication,
                "journalist": meta.journalist,
            }
        )
    elif meta.source_type == "email_slack":
        base_metadata.update(
            {
                "sender_role": meta.sender_role,
                "thread_id": meta.thread_id,
                "participants": meta.participants,
            }
        )

    chunks = _chunk_text(
        text=document.raw_text,
        doc_id=meta.doc_id,
        source_type=meta.source_type,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        base_metadata=base_metadata,
    )
    logger.info(
        "Chunked document %s (%s) into %d chunks",
        meta.title,
        meta.source_type,
        len(chunks),
    )
    return chunks
