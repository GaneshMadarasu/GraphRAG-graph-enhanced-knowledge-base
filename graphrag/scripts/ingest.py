#!/usr/bin/env python3
"""CLI script to ingest documents into the GraphRAG system."""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.vector_store import upsert_chunks
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_directory
from src.ingestion.entity_extractor import extract_from_chunks
from src.ingestion.graph_builder import build_graph
from src.utils.config import get_settings
from src.utils.neo4j_client import is_available as neo4j_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the full ingestion pipeline."""
    settings = get_settings()
    data_dir = Path("data/sample_docs")

    print(f"\n{'='*60}")
    print("GraphRAG Ingestion Pipeline")
    print(f"{'='*60}\n")

    # Step 1: Load documents
    print(f"Loading documents from '{data_dir}'...")
    docs = load_directory(data_dir)
    if not docs:
        print("ERROR: No documents found. Add .txt, .md, or .pdf files to data/sample_docs/")
        sys.exit(1)
    print(f"  Loaded {len(docs)} documents\n")

    # Step 2: Chunk
    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks\n")

    # Step 3: Extract entities
    print("Extracting entities and relationships (this may take a while)...")
    results = []
    total_entities = 0
    total_rels = 0
    for i, chunk in enumerate(chunks):
        from src.ingestion.entity_extractor import extract_from_chunk
        result = extract_from_chunk(chunk)
        results.append(result)
        total_entities += len(result.entities)
        total_rels += len(result.relationships)
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            print(
                f"  [{i+1}/{len(chunks)}] Processing '{chunk.filename}' "
                f"-- so far: {total_entities} entities, {total_rels} relationships"
            )

    print(f"\n  Total extracted: {total_entities} entities, {total_rels} relationships\n")

    # Step 4: Store in vector DB
    print("Embedding and storing chunks in ChromaDB...")
    upsert_chunks(chunks)
    print(f"  Stored {len(chunks)} chunks in ChromaDB\n")

    # Step 5: Build knowledge graph
    if neo4j_available():
        print("Building knowledge graph in Neo4j...")
        stats = build_graph(chunks, results)
        print(f"  Chunks stored: {stats['chunks_stored']}")
        print(f"  Unique entities: {stats['unique_entities']}")
        print(f"  Unique relationships: {stats['unique_relationships']}")
    else:
        print("WARNING: Neo4j is not available. Skipping graph storage.")
        print("  Start Neo4j and re-run to enable graph features.\n")

    print(f"\n{'='*60}")
    print("Ingestion complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
