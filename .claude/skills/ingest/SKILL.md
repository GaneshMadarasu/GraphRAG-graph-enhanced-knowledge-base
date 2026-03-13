---
name: ingest
description: Run the GraphRAG ingestion pipeline to load documents into Neo4j and ChromaDB
---

# Ingest Documents

Run the full ingestion pipeline: load documents → chunk → extract entities → store in Neo4j graph + ChromaDB vector store.

## Steps

1. Check if the API server is running:
```bash
curl -s http://localhost:8000/health
```

2. **If API is running**, trigger ingestion via the endpoint:
```bash
curl -X POST http://localhost:8000/ingest
```

3. **If API is not running**, run the CLI script directly:
```bash
cd graphrag && python scripts/ingest.py
```

4. Report what was ingested — number of documents, chunks, entities, and relationships created. Pull this from the response or from `/stats` after ingestion.

## Notes
- Documents must be placed in `graphrag/data/sample_docs/` before ingesting
- Supported formats: `.txt`, `.md`, `.pdf`
- Re-ingestion is safe — upsert semantics prevent duplicates
- If Neo4j is not running, start it first with `docker-compose up -d` from the `graphrag/` directory
