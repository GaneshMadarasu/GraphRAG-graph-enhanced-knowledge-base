# GraphRAG — Graph-Enhanced Retrieval Augmented Generation

A production-quality knowledge base system that combines a Neo4j knowledge graph with ChromaDB vector search to answer questions with richer context than either approach alone.

---

## Architecture

```
                         INGESTION PIPELINE
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  Raw Documents (.txt / .pdf / .md)                      │
  │       │                                                 │
  │       ▼                                                 │
  │  ┌─────────────┐                                        │
  │  │  Chunker    │  RecursiveCharacterTextSplitter        │
  │  │  (512 tok,  │  chunk_size=512, overlap=64            │
  │  │  64 overlap)│                                        │
  │  └──────┬──────┘                                        │
  │         │                                               │
  │         ├─────────────────────────┐                     │
  │         ▼                         ▼                     │
  │  ┌─────────────────┐   ┌──────────────────────┐        │
  │  │ Entity Extractor│   │  sentence-transformers│        │
  │  │  (Claude Haiku) │   │  all-MiniLM-L6-v2    │        │
  │  │  Entities +     │   │  (local, no API)      │        │
  │  │  Relationships  │   └──────────┬───────────┘        │
  │  └────────┬────────┘              │                     │
  │           │                       │                     │
  │           ▼                       ▼                     │
  │  ┌─────────────────┐   ┌──────────────────────┐        │
  │  │  Neo4j Graph DB │   │   ChromaDB Vector DB │        │
  │  │  Entity nodes   │   │   Chunk embeddings   │        │
  │  │  Typed edges    │   │   Cosine similarity  │        │
  │  │  Chunk links    │   │   HNSW index         │        │
  │  └─────────────────┘   └──────────────────────┘        │
  └─────────────────────────────────────────────────────────┘

                          QUERY PIPELINE
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  User Question                                          │
  │       │                                                 │
  │       ├──────────────────────┬──────────────────────┐  │
  │       ▼                      ▼                       │  │
  │  ┌─────────────┐    ┌──────────────────┐            │  │
  │  │   spaCy NER │    │  sentence-       │            │  │
  │  │   Entity    │    │  transformers    │            │  │
  │  │   Extraction│    └────────┬─────────┘            │  │
  │  └──────┬──────┘             │                      │  │
  │         │                    ▼                      │  │
  │         ▼             ┌──────────────────┐          │  │
  │  ┌─────────────┐      │  ChromaDB ANN    │          │  │
  │  │ Graph       │      │  Vector Search   │          │  │
  │  │ Retriever   │      │  top-K chunks    │          │  │
  │  │ 1-hop +     │      └────────┬─────────┘          │  │
  │  │ shortest    │               │                    │  │
  │  │ path        │               │                    │  │
  │  └──────┬──────┘               │                    │  │
  │         │                      │                    │  │
  │         └──────────┬───────────┘                    │  │
  │                    ▼                                 │  │
  │           ┌──────────────────┐                      │  │
  │           │  Hybrid Merger   │  Score Fusion:       │  │
  │           │  vector * 0.6 +  │  deduplicate,        │  │
  │           │  graph  * 0.4    │  re-rank, top-K      │  │
  │           └────────┬─────────┘                      │  │
  │                    ▼                                 │  │
  │           ┌──────────────────┐                      │  │
  │           │  Answer          │  Claude Sonnet with  │  │
  │           │  Generator       │  grounded context    │  │
  │           │  (Claude Sonnet) │  + citation          │  │
  │           └────────┬─────────┘                      │  │
  │                    ▼                                 │  │
  │              Final Answer + Sources                  │  │
  └─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
graphrag/
├── docker-compose.yml          # Neo4j + app services
├── .env.example                # Environment variable template
├── requirements.txt            # Pinned Python dependencies
├── Dockerfile                  # Container build instructions
├── README.md                   # This file
├── data/
│   └── sample_docs/            # Drop your documents here
│       ├── turing.txt
│       ├── von_neumann.txt
│       ├── bell_labs.txt
│       ├── intel_microprocessors.txt
│       └── early_computers.txt
├── mcp_server.py               # MCP server (Claude Desktop / Claude Code integration)
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  # Load .txt, .pdf, .md files
│   │   ├── chunker.py          # RecursiveCharacterTextSplitter
│   │   ├── entity_extractor.py # Claude Haiku entity/relationship extraction
│   │   └── graph_builder.py    # Write entities+edges to Neo4j
│   ├── embeddings/
│   │   └── vector_store.py     # ChromaDB + sentence-transformers embeddings
│   ├── retrieval/
│   │   ├── graph_retriever.py  # Cypher traversal (1-hop + paths)
│   │   ├── vector_retriever.py # ChromaDB ANN search
│   │   └── hybrid_retriever.py # Parallel merge with score fusion
│   ├── generation/
│   │   └── answer_generator.py # Claude Sonnet grounded answer generation
│   ├── api/
│   │   └── main.py             # FastAPI: /ingest /query /health /stats
│   └── utils/
│       ├── config.py           # Pydantic settings
│       └── neo4j_client.py     # Connection singleton + retry
└── scripts/
    ├── ingest.py               # CLI ingestion runner
    └── query.py                # CLI query runner
```

---

## Hybrid Retrieval Scoring

The hybrid retriever queries Neo4j and ChromaDB in parallel (via `asyncio.gather`), then fuses their scores:

```
final_score = (vector_cosine_similarity * 0.6) + (graph_relevance_score * 0.4)
```

**Vector scores** are cosine similarities from ChromaDB (range: 0.0 to 1.0), scaled by **0.6**.

**Graph scores** are fixed relevance weights based on retrieval type:
| Graph Result Type | Raw Score | After 0.4x Weight |
|-------------------|-----------|-------------------|
| Path triple (shortest path) | 0.90 | 0.36 |
| 1-hop graph triple | 0.80 | 0.32 |
| Graph-linked chunk | 0.75 | 0.30 |

After fusion, results are deduplicated by `chunk_id` (keeping highest score), sorted descending, and truncated to `top_k`.

The rationale: vector search finds semantically similar text even without exact entity matches; graph traversal finds factual connections that may be in non-similar-sounding chunks. Combining both with a 60/40 weighting rewards semantic relevance while adding relational context from the knowledge graph.

---

## Setup

### Prerequisites

- Docker and Docker Compose
- An Anthropic API key

### 1. Clone and configure

```bash
git clone <repo-url>
cd graphrag
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_actual_key
```

### 2. Start services with Docker Compose

```bash
docker-compose up -d
```

This starts:
- **Neo4j** on ports 7474 (browser) and 7687 (bolt)
- **App** on port 8000

Wait ~30 seconds for Neo4j to become healthy, then verify:

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status": "healthy", "neo4j": "connected", "chromadb": "connected"}
```

### 3. Run without Docker (local development)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start Neo4j separately (or use Neo4j Desktop / AuraDB)
# Then set NEO4J_URI in .env

# Run the API server
uvicorn src.api.main:app --reload --port 8000
```

---

## Ingesting Documents

### Via API

```bash
curl -X POST http://localhost:8000/ingest
```

### Via CLI script

```bash
python scripts/ingest.py
```

The ingestion pipeline:
1. Loads all `.txt`, `.md`, and `.pdf` files from `data/sample_docs/`
2. Splits them into 512-character overlapping chunks
3. Calls Claude Haiku to extract entities (PERSON, ORG, LOCATION, CONCEPT, EVENT, TECHNOLOGY, PRODUCT) and typed relationships
4. Embeds all chunks with `all-MiniLM-L6-v2` (sentence-transformers, runs locally) and stores them in ChromaDB
5. Writes entity nodes, chunk nodes, and relationship edges to Neo4j

---

## Querying

### Via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What did Alan Turing contribute to Bletchley Park?", "top_k": 5}'
```

### Via CLI script

```bash
python scripts/query.py "What did Alan Turing contribute to Bletchley Park?"
python scripts/query.py "How are the transistor and the microprocessor related?" 7
```

### Example queries to try

```bash
# Cross-document entity relationships
python scripts/query.py "What is the connection between Bell Labs and the transistor?"

# Multi-hop graph reasoning
python scripts/query.py "How did von Neumann's stored-program concept relate to Turing's work?"

# Entity-centric lookup
python scripts/query.py "What organizations did Gordon Moore found or work at?"

# Historical causality
python scripts/query.py "How did wartime codebreaking at Bletchley Park influence early computer design?"

# Technical concept tracing
python scripts/query.py "Explain how Moore's Law shaped the semiconductor industry"
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Load and ingest all documents from `data/sample_docs/` |
| POST | `/query` | Answer a question using hybrid retrieval |
| GET | `/graph/entity/{name}` | Explore an entity's graph neighborhood |
| GET | `/health` | Neo4j and ChromaDB connectivity status |
| GET | `/stats` | Counts of entity nodes, relationships, and chunks |

### Query request body

```json
{
  "question": "Who invented the transistor?",
  "top_k": 5
}
```

### Query response

```json
{
  "answer": "The transistor was invented at Bell Labs by John Bardeen and Walter Brattain...",
  "sources": ["chunk-uuid-1", "chunk-uuid-2"],
  "entities_used": ["Bell Labs", "John Bardeen", "Walter Brattain"],
  "context_items_used": 5
}
```

---

## Adding Your Own Documents

1. Drop `.txt`, `.md`, or `.pdf` files into `data/sample_docs/`
2. Call `POST /ingest` or run `python scripts/ingest.py`

The system handles multiple ingestion calls safely — existing chunks and entities are upserted, not duplicated.

To adjust chunking parameters, edit `.env`:
```
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

To change the number of retrieval results:
```
VECTOR_TOP_K=5
GRAPH_HOP_LIMIT=20
```

---

## Neo4j Browser

Navigate to `http://localhost:7474` and log in with `neo4j` / `password`.

Useful Cypher queries to explore the graph:

```cypher
// See all entity types
MATCH (e:Entity) RETURN DISTINCT e.type, count(*) ORDER BY count(*) DESC

// Explore Alan Turing's neighborhood
MATCH (e:Entity {name: "Alan Turing"})-[r]-(n) RETURN e, r, n LIMIT 25

// Find shortest path between two entities
MATCH (a:Entity {name: "Alan Turing"}), (b:Entity {name: "Bell Labs"})
MATCH path = shortestPath((a)-[*1..5]-(b))
RETURN path

// See which chunks mention an entity
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e.name = "transistor"
RETURN c.text LIMIT 5
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `EXTRACTION_MODEL` | `claude-haiku-4-5-20251001` | Claude model for entity extraction |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local) |
| `GENERATION_MODEL` | `claude-sonnet-4-6` | Claude model for answer generation |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `VECTOR_TOP_K` | `5` | Number of vector results to retrieve |
| `GRAPH_HOP_LIMIT` | `20` | Max graph traversal results |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## MCP Server

`mcp_server.py` exposes the knowledge base as **MCP tools** so Claude Desktop and Claude Code can query it directly mid-conversation — no curl, no terminal.

### Tools

| Tool | Description |
|---|---|
| `query_knowledge_base` | Hybrid graph + vector retrieval with a Claude-generated answer |
| `get_entity` | 1-hop graph neighborhood for any entity |
| `ingest_documents` | Load → chunk → extract → store a directory of documents |
| `get_stats` | Entity node, relationship, and chunk counts |
| `health_check` | Neo4j + ChromaDB connectivity status |

### Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### Connect to Claude Code

```bash
claude mcp add graphrag /path/to/python \
  -- /path/to/graphrag/mcp_server.py \
  --cwd /path/to/graphrag
```

> Neo4j and ChromaDB must be running before starting a session that uses the MCP tools (`docker-compose up -d` covers this).

---

## License

See [LICENSE](../LICENSE).
