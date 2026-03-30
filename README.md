# GraphRAG — Graph-Enhanced Knowledge Base

A hybrid retrieval-augmented generation pipeline that combines **Neo4j knowledge graphs** with **ChromaDB vector search**. Documents are automatically parsed into entities and relationships, stored as a graph, and queried using both Cypher traversal and semantic similarity for grounded, explainable answers.

---

## Architecture

```
Documents (.txt / .md / .pdf)
        │
        ▼
  Document Loader
        │
        ▼
    Chunker (512 chars, 64 overlap)
        │
        ├──► Entity Extractor (Claude Haiku) ──► Neo4j Graph
        │
        └──► Embeddings (all-MiniLM-L6-v2) ──► ChromaDB
                                    │
                             Query Time
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
      Graph Retriever (40%)                  Vector Retriever (60%)
      (Cypher + spaCy NER)                 (Semantic similarity)
              │                                           │
              └─────────────── Hybrid Merger ────────────┘
                                    │
                                    ▼
                       Answer Generator (Claude Sonnet)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             FastAPI REST                    MCP Server
           (/query, /ingest            (Claude Desktop /
            /health, /stats)            Claude Code tools)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph DB | Neo4j 5.25 (APOC plugin) |
| Vector DB | ChromaDB 0.5.17 (HNSW) |
| LLM | Claude Haiku (extraction) + Claude Sonnet (generation) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local) |
| NLP | spaCy en_core_web_sm |
| API | FastAPI + Uvicorn |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| MCP | mcp>=1.0.0 (FastMCP stdio server) |
| PDF Parsing | pypdf |
| Infra | Docker + Docker Compose |

---

## MCP Integration

`graphrag/mcp_server.py` exposes the knowledge base as **MCP (Model Context Protocol) tools**, letting Claude Desktop and Claude Code query it directly mid-conversation.

### Tools

| Tool | Description |
|---|---|
| `query_knowledge_base` | Hybrid graph + vector retrieval with a Claude-generated answer |
| `get_entity` | 1-hop graph neighborhood for any entity |
| `ingest_documents` | Load → chunk → extract → store a directory of documents |
| `get_stats` | Entity node, relationship, and chunk counts |
| `health_check` | Neo4j + ChromaDB connectivity status |

### Quick connect (Claude Desktop)

```json
{
  "mcpServers": {
    "graphrag": {
      "command": "/path/to/python",
      "args": ["/path/to/graphrag/mcp_server.py"],
      "cwd": "/path/to/graphrag"
    }
  }
}
```

---

## Features

- **Automatic knowledge graph construction** — extracts entities (PERSON, ORG, LOCATION, CONCEPT, EVENT, TECHNOLOGY, PRODUCT) and typed relationships (CREATED, INVENTED, WORKED_AT, INFLUENCED, etc.) using Claude Haiku
- **Hybrid retrieval** — fuses graph (40%) and vector (60%) scores, deduplicates by chunk ID, re-ranks
- **Graph traversal** — 1-hop neighbor lookup + shortest path queries (up to 4 hops)
- **Grounded generation** — answers cite source chunks and entity triples; no hallucination beyond retrieved context
- **Multi-format ingestion** — `.txt`, `.md`, `.pdf`
- **Graceful degradation** — falls back to vector-only if Neo4j is unavailable
- **Upsert semantics** — safe to re-ingest documents without duplication

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Anthropic API key

### Run with Docker

```bash
git clone <repo-url>
cd graphrag
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
docker-compose up -d
```

Services:
- API: `http://localhost:8000`
- Neo4j Browser: `http://localhost:7474`

### Local Development

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn src.api.main:app --reload --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Run full ingestion pipeline on `data/sample_docs/` |
| `POST` | `/query` | Answer a question using hybrid retrieval |
| `GET` | `/graph/entity/{name}` | Explore entity neighborhood |
| `GET` | `/health` | Neo4j + ChromaDB health check |
| `GET` | `/stats` | Node, edge, and chunk counts |

### Ingest documents

```bash
curl -X POST http://localhost:8000/ingest
# or via CLI:
python scripts/ingest.py
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who invented the transistor?", "top_k": 5}'
# or via CLI:
python scripts/query.py "Who invented the transistor?" 5
```

---

## Configuration

Key variables in `.env`:

```env
ANTHROPIC_API_KEY=your_key

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval
VECTOR_TOP_K=5
GRAPH_HOP_LIMIT=20

# Models
EXTRACTION_MODEL=claude-haiku-4-5-20251001
EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_MODEL=claude-sonnet-4-6
```

---

## Project Structure

```
graphrag/
├── mcp_server.py       # MCP server (Claude Desktop / Claude Code integration)
├── src/
│   ├── api/            # FastAPI server
│   ├── ingestion/      # Loader, chunker, entity extractor, graph builder
│   ├── embeddings/     # ChromaDB vector store
│   ├── retrieval/      # Graph, vector, and hybrid retrievers
│   ├── generation/     # Claude Sonnet answer generator
│   └── utils/          # Config, Neo4j client
├── scripts/            # CLI runners (ingest.py, query.py)
├── data/sample_docs/   # Drop documents here
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## License

See [LICENSE](LICENSE).
