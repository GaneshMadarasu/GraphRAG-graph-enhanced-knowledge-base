"""MCP server exposing GraphRAG as tools for Claude Desktop / Claude Code."""
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from graphrag/
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

from src.embeddings.vector_store import get_chunk_count, is_available as chroma_available
from src.generation.answer_generator import generate_answer
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_directory
from src.ingestion.entity_extractor import extract_from_chunks
from src.ingestion.graph_builder import build_graph
from src.embeddings.vector_store import upsert_chunks
from src.retrieval.graph_retriever import get_entity_neighborhood
from src.retrieval.hybrid_retriever import hybrid_retrieve_async
from src.utils.config import get_settings
from src.utils.neo4j_client import is_available as neo4j_available, get_session

logging.basicConfig(level=logging.WARNING)

mcp = FastMCP(
    "GraphRAG",
    instructions=(
        "A graph-enhanced knowledge base. Use query_knowledge_base for questions, "
        "get_entity to explore graph neighbors, ingest_documents to add new docs, "
        "get_stats for counts, and health_check for service status."
    ),
)


@mcp.tool()
async def query_knowledge_base(question: str, top_k: int = 5) -> str:
    """Answer a question using hybrid graph + vector retrieval over the knowledge base.

    Args:
        question: The question to answer.
        top_k: Number of context items to retrieve (default 5).

    Returns:
        A grounded answer with cited sources and entities used.
    """
    if not question.strip():
        return "Error: question cannot be empty."

    context_items = await hybrid_retrieve_async(question, top_k=top_k)
    result = generate_answer(question, context_items)

    output = {
        "answer": result.answer,
        "sources": result.sources,
        "entities_used": result.entities_used,
        "context_items_used": result.context_items_used,
    }
    return json.dumps(output, indent=2)


@mcp.tool()
def get_entity(name: str) -> str:
    """Return an entity and its direct graph neighbors (1-hop neighborhood).

    Args:
        name: Entity name to look up (case-insensitive partial match).

    Returns:
        JSON with the entity and its neighbors, relations, and context.
    """
    if not neo4j_available():
        return json.dumps({"error": "Neo4j is not available."})

    result = get_entity_neighborhood(name)
    return json.dumps(result, indent=2)


@mcp.tool()
def ingest_documents(data_dir: str = "data/sample_docs") -> str:
    """Ingest documents from a directory into the knowledge base.

    Loads, chunks, extracts entities, builds the graph, and indexes vectors.

    Args:
        data_dir: Path to the directory containing documents (default: data/sample_docs).

    Returns:
        Summary of documents loaded, chunks created, entities and relationships extracted.
    """
    doc_path = Path(data_dir)
    if not doc_path.exists():
        return json.dumps({"error": f"Directory '{data_dir}' not found."})

    docs = load_directory(doc_path)
    if not docs:
        return json.dumps({"error": "No documents found in the directory."})

    chunks = chunk_documents(docs)
    extraction_results = extract_from_chunks(chunks)

    total_entities = sum(len(r.entities) for r in extraction_results)
    total_rels = sum(len(r.relationships) for r in extraction_results)

    upsert_chunks(chunks)

    if neo4j_available():
        build_graph(chunks, extraction_results)

    result = {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "entities_extracted": total_entities,
        "relationships_extracted": total_rels,
        "message": f"Successfully ingested {len(docs)} documents from '{data_dir}'.",
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Return knowledge base statistics: entity node count, relationship count, chunk count.

    Returns:
        JSON with entity_nodes, relationships, and chunks counts.
    """
    chunk_count = get_chunk_count() if chroma_available() else -1
    node_count = -1
    edge_count = -1

    if neo4j_available():
        with get_session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS count")
            node_count = result.single()["count"]
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            edge_count = result.single()["count"]

    stats = {
        "entity_nodes": node_count,
        "relationships": edge_count,
        "chunks": chunk_count,
    }
    return json.dumps(stats, indent=2)


@mcp.tool()
def health_check() -> str:
    """Check connectivity to Neo4j and ChromaDB.

    Returns:
        JSON with overall status and per-service connectivity.
    """
    neo4j_ok = neo4j_available()
    chroma_ok = chroma_available()
    status = "healthy" if (neo4j_ok and chroma_ok) else "degraded"

    result = {
        "status": status,
        "neo4j": "connected" if neo4j_ok else "disconnected",
        "chromadb": "connected" if chroma_ok else "disconnected",
    }
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
