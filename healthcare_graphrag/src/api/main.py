"""
FastAPI application for Healthcare GraphRAG.

Endpoints:
  POST /ingest                                  — full ingestion pipeline
  POST /query                                   — hybrid QA
  GET  /graph/drug/{drug_name}/interactions     — drug interaction graph
  GET  /graph/disease/{disease_name}/full-profile
  GET  /graph/cross-document/{entity_name}
  GET  /stats
  GET  /health
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.embeddings.vector_store import get_vector_store
from src.generation.answer_generator import MedicalAnswerGenerator, AnswerResponse
from src.ingestion.chunker import chunk_document
from src.ingestion.document_loader import load_directory
from src.ingestion.entity_extractor import MedicalEntityExtractor
from src.ingestion.entity_normalizer import normalize_extraction_result
from src.ingestion.graph_builder import GraphBuilder
from src.retrieval.graph_retriever import MedicalGraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.utils.config import get_settings
from src.utils.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    # Connect Neo4j
    neo4j = get_neo4j_client()
    try:
        await neo4j.connect()
        logger.info("Neo4j connected on startup")
    except Exception as exc:
        logger.warning("Neo4j unavailable on startup (vector-only mode): %s", exc)

    # Pre-warm vector store connection
    try:
        vs = get_vector_store()
        vs.health_check()
        logger.info("ChromaDB connected on startup")
    except Exception as exc:
        logger.warning("ChromaDB unavailable on startup: %s", exc)

    yield

    await neo4j.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Healthcare GraphRAG API",
    description="Hybrid Neo4j + ChromaDB medical knowledge retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class IngestResponse(BaseModel):
    docs_processed: int
    entities_extracted: int
    relationships_mapped: int
    chunks_stored: int
    time_taken: float
    errors: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    doc_types: List[str] = Field(
        default=["research_paper", "internal_report", "news_article", "email_slack"]
    )
    top_k: int = Field(default=5, ge=1, le=20)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


# ── Helper: build retrieval + generation pipeline ─────────────────────────────

def _build_hybrid_retriever(neo4j_available: bool) -> HybridRetriever:
    vs = get_vector_store()
    vector_ret = VectorRetriever(vs)
    if neo4j_available:
        graph_ret = MedicalGraphRetriever(get_neo4j_client())
    else:
        graph_ret = None
    return HybridRetriever(vector=vector_ret, graph=graph_ret)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    neo4j = get_neo4j_client()
    neo4j_ok = await neo4j.health_check()
    chroma_ok = get_vector_store().health_check()
    return {
        "status": "ok" if (neo4j_ok and chroma_ok) else "degraded",
        "neo4j": neo4j_ok,
        "chromadb": chroma_ok,
    }


@app.get("/stats", tags=["system"])
async def stats():
    neo4j = get_neo4j_client()
    chroma_count = get_vector_store().get_count()

    if not neo4j.is_connected:
        return {
            "chromadb_chunks": chroma_count,
            "neo4j": "unavailable",
        }

    neo4j_stats = await neo4j.get_stats()
    return {
        "chromadb_chunks": chroma_count,
        "neo4j": neo4j_stats,
    }


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest():
    """Trigger full ingestion pipeline over all documents in data/."""
    cfg = get_settings()
    start = time.time()
    errors: List[str] = []

    neo4j = get_neo4j_client()
    neo4j_ok = neo4j.is_connected
    if not neo4j_ok:
        try:
            await neo4j.connect()
            neo4j_ok = True
        except Exception as exc:
            errors.append(f"Neo4j unavailable: {exc}")

    vs = get_vector_store()
    extractor = MedicalEntityExtractor(max_concurrency=5)
    graph_builder = GraphBuilder(neo4j) if neo4j_ok else None

    total_docs = 0
    total_chunks = 0
    total_entities = 0
    total_relationships = 0

    try:
        documents = load_directory(cfg.data_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load documents: {exc}")

    for doc in documents:
        try:
            chunks = chunk_document(doc)
            results = await extractor.extract_chunks(chunks)

            for result in results:
                normalize_extraction_result(result)

            vs.add_chunks(chunks)

            if graph_builder:
                stats_dict = await graph_builder.ingest_extraction_results(
                    meta=doc.metadata,
                    chunks=chunks,
                    results=results,
                )
                total_entities += stats_dict.get("entities", 0)
                total_relationships += stats_dict.get("relationships", 0)
            else:
                for r in results:
                    total_entities += len(r.entities)
                    total_relationships += len(r.relationships)

            total_chunks += len(chunks)
            total_docs += 1
        except Exception as exc:
            err = f"Failed to process {doc.metadata.title}: {exc}"
            logger.error(err)
            errors.append(err)

    return IngestResponse(
        docs_processed=total_docs,
        entities_extracted=total_entities,
        relationships_mapped=total_relationships,
        chunks_stored=total_chunks,
        time_taken=round(time.time() - start, 2),
        errors=errors,
    )


@app.post("/query", response_model=AnswerResponse, tags=["query"])
async def query(request: QueryRequest):
    """Hybrid graph + vector QA over ingested medical documents."""
    neo4j = get_neo4j_client()
    neo4j_ok = neo4j.is_connected

    retriever = _build_hybrid_retriever(neo4j_ok)
    generator = MedicalAnswerGenerator()

    retrieval = await retriever.retrieve(
        question=request.question,
        top_k=request.top_k,
        doc_types=request.doc_types,
        min_confidence=request.min_confidence,
    )

    answer = await generator.generate(
        question=request.question,
        retrieval_output=retrieval,
    )
    return answer


@app.get(
    "/graph/drug/{drug_name}/interactions",
    tags=["graph"],
    summary="Get all drug interactions for a drug",
)
async def drug_interactions(
    drug_name: str = Path(..., description="Drug name (generic preferred)"),
):
    neo4j = get_neo4j_client()
    if not neo4j.is_connected:
        raise HTTPException(status_code=503, detail="Neo4j unavailable")
    retriever = MedicalGraphRetriever(neo4j)
    from src.ingestion.entity_normalizer import normalize_entity_name
    canonical = normalize_entity_name(drug_name, "Drug")
    interactions = await retriever.get_drug_interactions(canonical)
    contraindications = await retriever.get_drug_contraindications(canonical)
    return {
        "drug": canonical,
        "interactions": interactions,
        "contraindications": contraindications,
        "warnings": [
            f"⚠️ WARNING: {canonical} interacts with {r.get('drug2', '')}"
            for r in interactions
            if r.get("drug2")
        ],
    }


@app.get(
    "/graph/disease/{disease_name}/full-profile",
    tags=["graph"],
    summary="Full disease profile: symptoms, drugs, genes, trials",
)
async def disease_profile(
    disease_name: str = Path(..., description="Disease name"),
):
    neo4j = get_neo4j_client()
    if not neo4j.is_connected:
        raise HTTPException(status_code=503, detail="Neo4j unavailable")
    retriever = MedicalGraphRetriever(neo4j)
    from src.ingestion.entity_normalizer import normalize_entity_name
    canonical = normalize_entity_name(disease_name, "Disease")
    return await retriever.get_disease_full_profile(canonical)


@app.get(
    "/graph/cross-document/{entity_name}",
    tags=["graph"],
    summary="All chunks mentioning this entity across all document types",
)
async def cross_document(
    entity_name: str = Path(..., description="Entity name"),
    limit: int = Query(default=10, ge=1, le=50),
):
    neo4j = get_neo4j_client()
    if not neo4j.is_connected:
        raise HTTPException(status_code=503, detail="Neo4j unavailable")
    retriever = MedicalGraphRetriever(neo4j)
    from src.ingestion.entity_normalizer import normalize_entity_name
    canonical = normalize_entity_name(entity_name)
    results = await retriever.get_cross_document_evidence(canonical, limit=limit)
    return {
        "entity": canonical,
        "mentions": results,
        "total": len(results),
    }
