"""
CLI ingestion script.

Usage:
    python -m scripts.ingest [--data-dir data/] [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.vector_store import get_vector_store
from src.ingestion.chunker import chunk_document
from src.ingestion.document_loader import load_directory
from src.ingestion.entity_extractor import MedicalEntityExtractor
from src.ingestion.entity_normalizer import normalize_extraction_result
from src.ingestion.graph_builder import GraphBuilder
from src.utils.config import get_settings
from src.utils.neo4j_client import get_neo4j_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_ingestion(data_dir: str, reset: bool = False) -> None:
    cfg = get_settings()
    neo4j = get_neo4j_client()
    neo4j_ok = False

    print(f"\n{'='*60}")
    print("  Healthcare GraphRAG — Ingestion Pipeline")
    print(f"{'='*60}\n")

    # Connect Neo4j
    try:
        await neo4j.connect()
        neo4j_ok = True
        print("✅ Neo4j connected")
    except Exception as exc:
        print(f"⚠️  Neo4j unavailable — ingesting to ChromaDB only: {exc}")

    # Connect ChromaDB
    vs = get_vector_store()
    try:
        vs.health_check()
        print("✅ ChromaDB connected")
        if reset:
            vs.delete_collection()
            print("🗑️  ChromaDB collection reset")
    except Exception as exc:
        print(f"❌ ChromaDB unavailable: {exc}")
        return

    start = time.time()
    extractor = MedicalEntityExtractor(max_concurrency=5)
    graph_builder = GraphBuilder(neo4j) if neo4j_ok else None

    documents = load_directory(data_dir)
    print(f"\n📄 Found {len(documents)} documents in {data_dir}\n")

    totals = {"docs": 0, "chunks": 0, "entities": 0, "relationships": 0, "errors": 0}

    for doc in documents:
        print(f"  Processing: {doc.metadata.title[:60]} [{doc.metadata.source_type}]")
        try:
            chunks = chunk_document(doc)
            results = await extractor.extract_chunks(chunks)

            for result in results:
                normalize_extraction_result(result)

            vs.add_chunks(chunks)

            if graph_builder:
                stats = await graph_builder.ingest_extraction_results(
                    meta=doc.metadata,
                    chunks=chunks,
                    results=results,
                )
                totals["entities"] += stats["entities"]
                totals["relationships"] += stats["relationships"]
            else:
                for r in results:
                    totals["entities"] += len(r.entities)
                    totals["relationships"] += len(r.relationships)

            totals["chunks"] += len(chunks)
            totals["docs"] += 1
            print(f"    ✓ {len(chunks)} chunks | {totals['entities']} entities so far")

        except Exception as exc:
            logger.error("Failed: %s | %s", doc.metadata.title, exc)
            totals["errors"] += 1

    elapsed = round(time.time() - start, 1)

    print(f"\n{'='*60}")
    print("  Ingestion Complete")
    print(f"{'='*60}")
    print(f"  Documents processed : {totals['docs']}")
    print(f"  Chunks stored       : {totals['chunks']}")
    print(f"  Entities extracted  : {totals['entities']}")
    print(f"  Relationships mapped: {totals['relationships']}")
    print(f"  Errors              : {totals['errors']}")
    print(f"  Time taken          : {elapsed}s")
    print(f"{'='*60}\n")

    await neo4j.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare GraphRAG ingestion pipeline")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument(
        "--reset", action="store_true", help="Reset ChromaDB collection before ingesting"
    )
    args = parser.parse_args()
    asyncio.run(run_ingestion(data_dir=args.data_dir, reset=args.reset))


if __name__ == "__main__":
    main()
