"""Load .txt, .pdf, and .md files into Document objects."""
import logging
import uuid
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """Represents a loaded document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content: str
    metadata: dict = Field(default_factory=dict)


def load_document(path: Path) -> Document:
    """Load a single document from path. Supports .txt, .md, .pdf."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        content = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            content = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise ImportError("pypdf is required to load PDF files")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    doc = Document(
        filename=path.name,
        content=content,
        metadata={"source": str(path), "file_type": suffix},
    )
    logger.info("Loaded document '%s' (%d chars)", path.name, len(content))
    return doc


def load_directory(directory: Path) -> List[Document]:
    """Load all supported documents from a directory."""
    docs: List[Document] = []
    for pattern in ("*.txt", "*.md", "*.pdf"):
        for path in sorted(directory.glob(pattern)):
            try:
                docs.append(load_document(path))
            except Exception as exc:
                logger.error("Failed to load '%s': %s", path, exc)
    logger.info("Loaded %d documents from '%s'", len(docs), directory)
    return docs
