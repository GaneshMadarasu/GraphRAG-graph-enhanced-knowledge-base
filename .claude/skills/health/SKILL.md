---
name: health
description: Check the health of Neo4j and ChromaDB services in the GraphRAG stack
---

# Health Check

Check whether all GraphRAG services (API, Neo4j, ChromaDB) are up and running.

## Steps

1. Check the API health endpoint:
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

2. If the API is unreachable, check Docker container status:
```bash
docker-compose -f graphrag/docker-compose.yml ps
```

3. Check Neo4j directly:
```bash
curl -s http://localhost:7474 > /dev/null && echo "Neo4j: UP" || echo "Neo4j: DOWN"
```

4. Report the status of each service clearly:
   - **API** (port 8000)
   - **Neo4j** (port 7474 / 7687)
   - **ChromaDB** (embedded, check via API response)

5. If any service is down, suggest the fix:
   - API down → `cd graphrag && uvicorn src.api.main:app --reload --port 8000`
   - Neo4j/stack down → `cd graphrag && docker-compose up -d`
