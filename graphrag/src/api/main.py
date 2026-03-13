"""FastAPI application exposing GraphRAG endpoints."""
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embeddings.vector_store import get_chunk_count, is_available as chroma_available
from src.generation.answer_generator import generate_answer
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_directory
from src.ingestion.entity_extractor import extract_from_chunks
from src.ingestion.graph_builder import build_graph
from src.embeddings.vector_store import upsert_chunks
from src.retrieval.hybrid_retriever import hybrid_retrieve_async
from src.utils.config import get_settings
from src.utils.neo4j_client import is_available as neo4j_available, get_session
from src.retrieval.graph_retriever import get_entity_neighborhood

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GraphRAG API",
    description="Graph-Enhanced Retrieval Augmented Generation",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response from /query endpoint."""

    answer: str
    sources: List[str]
    entities_used: List[str]
    context_items_used: int


class IngestResponse(BaseModel):
    """Response from /ingest endpoint."""

    documents_loaded: int
    chunks_created: int
    entities_extracted: int
    relationships_extracted: int
    message: str


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Ingest all documents from data/sample_docs/."""
    data_dir = Path("data/sample_docs")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory {data_dir} not found")

    docs = load_directory(data_dir)
    if not docs:
        raise HTTPException(status_code=400, detail="No documents found to ingest")

    chunks = chunk_documents(docs)
    extraction_results = extract_from_chunks(chunks)

    total_entities = sum(len(r.entities) for r in extraction_results)
    total_rels = sum(len(r.relationships) for r in extraction_results)

    # Store in vector DB
    upsert_chunks(chunks)

    # Store in graph DB
    if neo4j_available():
        build_graph(chunks, extraction_results)
    else:
        logger.warning("Neo4j unavailable; skipping graph storage")

    return IngestResponse(
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        entities_extracted=total_entities,
        relationships_extracted=total_rels,
        message=f"Successfully ingested {len(docs)} documents",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using hybrid retrieval."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    context_items = await hybrid_retrieve_async(request.question, top_k=request.top_k)
    result = generate_answer(request.question, context_items)

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        entities_used=result.entities_used,
        context_items_used=result.context_items_used,
    )


@app.get("/graph/entity/{name}")
async def get_entity(name: str):
    """Return an entity and its graph neighbors."""
    if not neo4j_available():
        raise HTTPException(status_code=503, detail="Neo4j unavailable")
    return get_entity_neighborhood(name)


@app.get("/health")
async def health():
    """Return connectivity status of Neo4j and ChromaDB."""
    neo4j_ok = neo4j_available()
    chroma_ok = chroma_available()
    status = "healthy" if (neo4j_ok and chroma_ok) else "degraded"
    return {
        "status": status,
        "neo4j": "connected" if neo4j_ok else "disconnected",
        "chromadb": "connected" if chroma_ok else "disconnected",
    }


@app.get("/stats")
async def stats():
    """Return counts of nodes, edges, and chunks."""
    chunk_count = get_chunk_count() if chroma_available() else -1
    node_count = -1
    edge_count = -1

    if neo4j_available():
        with get_session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS count")
            node_count = result.single()["count"]
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            edge_count = result.single()["count"]

    return {
        "entity_nodes": node_count,
        "relationships": edge_count,
        "chunks": chunk_count,
    }
