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
        ├──► Entity Extractor (GPT-4o) ──► Neo4j Graph
        │
        └──► Embeddings (text-embedding-3-small) ──► ChromaDB
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
                         Answer Generator (GPT-4o)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph DB | Neo4j 5.25 (APOC plugin) |
| Vector DB | ChromaDB 0.5.17 (HNSW) |
| LLM | GPT-4o (extraction + generation) |
| Embeddings | OpenAI text-embedding-3-small |
| NLP | spaCy en_core_web_sm |
| API | FastAPI + Uvicorn |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| PDF Parsing | pypdf |
| Infra | Docker + Docker Compose |

---

## Features

- **Automatic knowledge graph construction** — extracts entities (PERSON, ORG, LOCATION, CONCEPT, EVENT, TECHNOLOGY, PRODUCT) and typed relationships (CREATED, INVENTED, WORKED_AT, INFLUENCED, etc.) using GPT-4o
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
- OpenAI API key

### Run with Docker

```bash
git clone <repo-url>
cd graphrag
cp .env.example .env
# Set OPENAI_API_KEY in .env
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
OPENAI_API_KEY=your_key

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
EXTRACTION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o
```

---

## Project Structure

```
graphrag/
├── src/
│   ├── api/            # FastAPI server
│   ├── ingestion/      # Loader, chunker, entity extractor, graph builder
│   ├── embeddings/     # ChromaDB vector store
│   ├── retrieval/      # Graph, vector, and hybrid retrievers
│   ├── generation/     # GPT-4o answer generator
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
