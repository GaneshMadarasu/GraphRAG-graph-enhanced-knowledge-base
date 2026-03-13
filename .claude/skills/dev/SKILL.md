---
name: dev
description: Start the GraphRAG local development server and supporting services
---

# Start Dev Server

Start the GraphRAG stack for local development.

## Steps

1. Check if services are already running:
```bash
curl -s http://localhost:8000/health && echo "Already running"
```

2. If not running, start Neo4j via Docker Compose:
```bash
docker-compose -f graphrag/docker-compose.yml up -d neo4j
```

3. Wait for Neo4j to be ready (~15 seconds), then start the FastAPI dev server:
```bash
cd graphrag && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Confirm everything is up:
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

5. Report the running endpoints:
   - API + docs: `http://localhost:8000` / `http://localhost:8000/docs`
   - Neo4j Browser: `http://localhost:7474`

## Notes
- Requires `.env` to be configured with `OPENAI_API_KEY`
- For full Docker stack (API + Neo4j together): `docker-compose -f graphrag/docker-compose.yml up -d`
- Hot reload is enabled — changes to `src/` restart the server automatically
