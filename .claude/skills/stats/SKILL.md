---
name: stats
description: Show GraphRAG knowledge base stats — node, relationship, and chunk counts
---

# Knowledge Base Stats

Fetch and display the current state of the GraphRAG knowledge base.

## Steps

1. Call the stats endpoint:
```bash
curl -s http://localhost:8000/stats | python3 -m json.tool
```

2. If the API is unreachable, query Neo4j directly via Cypher:
```bash
docker exec -it graphrag-neo4j-1 cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC;"
```

3. Present the stats in a readable table:

| Metric | Count |
|---|---|
| Entities (nodes) | ... |
| Relationships (edges) | ... |
| Document chunks | ... |

4. If counts are 0, the knowledge base is empty — suggest running `/ingest` first.
