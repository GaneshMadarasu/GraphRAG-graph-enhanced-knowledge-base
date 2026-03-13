"""Split documents into overlapping chunks."""
import logging
import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.ingestion.document_loader import Document
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """A text chunk derived from a parent document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


def chunk_document(doc: Document) -> List[Chunk]:
    """Split a document into chunks using RecursiveCharacterTextSplitter."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    texts = splitter.split_text(doc.content)
    chunks = [
        Chunk(
            doc_id=doc.id,
            filename=doc.filename,
            text=text,
            chunk_index=i,
            metadata={**doc.metadata, "chunk_index": i},
        )
        for i, text in enumerate(texts)
    ]
    logger.debug(
        "Document '%s' split into %d chunks", doc.filename, len(chunks)
    )
    return chunks


def chunk_documents(docs: List[Document]) -> List[Chunk]:
    """Chunk a list of documents."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info(
        "Total chunks created: %d from %d documents", len(all_chunks), len(docs)
    )
    return all_chunks
