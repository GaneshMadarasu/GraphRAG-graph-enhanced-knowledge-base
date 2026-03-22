# Healthcare GraphRAG

Production-grade medical knowledge graph system for answering complex clinical and
research questions by connecting information across hundreds of heterogeneous documents.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │               Healthcare GraphRAG                    │
                        └─────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  INGESTION PIPELINE                                                          │
  │                                                                              │
  │  ┌──────────────┐    ┌──────────┐    ┌─────────────────┐    ┌───────────┐   │
  │  │ Document     │    │ Medical  │    │ Entity          │    │ Graph     │   │
  │  │ Loader       │───▶│ Chunker  │───▶│ Extractor       │───▶│ Builder   │   │
  │  │              │    │          │    │ (GPT-4o)        │    │           │   │
  │  │ • PDF        │    │ 800 char │    │ • Entities      │    │ MERGE     │   │
  │  │ • DOCX       │    │ overlap  │    │ • Relations     │    │ (no dupes)│   │
  │  │ • TXT/HTML   │    │ sentence │    │ • Evidence      │    └─────┬─────┘   │
  │  │ • JSON/Slack │    │ boundary │    └────────┬────────┘          │         │
  │  └──────────────┘    └──────────┘             │                   │         │
  │                                    ┌──────────▼────────┐          │         │
  │                                    │ Entity Normalizer │          │         │
  │                                    │ • Drug synonyms   │          │         │
  │                                    │ • Disease aliases │          │         │
  │                                    │ • Fuzzy matching  │          │         │
  │                                    └───────────────────┘          │         │
  └──────────────────────────────────────────────────────────────────┼─────────┘
                                                                       │
           ┌────────────────────────────────────────────────────────┐  │
           │                   STORAGE LAYER                        │  │
           │                                                        │  │
           │  ┌─────────────────────────┐   ┌────────────────────┐ │  │
           │  │       Neo4j 5.x         │◀──┤  ChromaDB          │ │  │
           │  │   Knowledge Graph       │   │  Vector Store      │ │  │
           │  │                         │   │                    │ │  │
           │  │  Disease ──TREATS──▶ Drug  │   │  text-embedding-   │ │  │
           │  │  Drug ──INTERACTS──▶ Drug  │   │  3-small           │ │  │
           │  │  Gene ──ASSOC──▶ Disease   │   │  800-char chunks   │ │  │
           │  │  Chunk ──MENTIONS──▶ Entity│   │  + source metadata │ │  │
           │  │  Chunk ──PART_OF──▶ Doc    │   │                    │ │  │
           │  └─────────────────────────┘   └────────────────────┘ │◀─┘
           └────────────────────────────────────────────────────────┘
                                      │
           ┌───────────────────────────────────────────────────────────┐
           │                   RETRIEVAL LAYER                          │
           │                                                            │
           │  ┌─────────────────┐   ┌──────────────────┐               │
           │  │ Vector Retriever│   │  Graph Retriever  │               │
           │  │                 │   │                   │               │
           │  │ Semantic search │   │ • Drug interactions│              │
           │  │ + credibility   │   │ • Gene-disease     │              │
           │  │   weighting     │   │ • Comorbidities    │              │
           │  └────────┬────────┘   │ • Cross-document   │              │
           │           │            │ • Treatment paths  │              │
           │           │            └────────┬───────────┘              │
           │           └────────────┬────────┘                          │
           │                        ▼                                    │
           │              ┌──────────────────┐                          │
           │              │  Hybrid Retriever │                          │
           │              │                  │                          │
           │              │  Score Fusion:   │                          │
           │              │  0.4×vector      │                          │
           │              │  0.4×graph_hop   │                          │
           │              │  0.2×credibility │                          │
           │              └────────┬─────────┘                          │
           └───────────────────────┼───────────────────────────────────┘
                                   │
           ┌───────────────────────────────────────────────────────────┐
           │              GENERATION LAYER                              │
           │                                                            │
           │              ┌──────────────────────┐                     │
           │              │  Answer Generator     │                     │
           │              │  (GPT-4o)             │                     │
           │              │                       │                     │
           │              │  • Per-source cites   │                     │
           │              │  • ⚠️ Drug warnings   │                     │
           │              │  • Confidence score   │                     │
           │              │  • Safety disclaimer  │                     │
           │              └──────────┬────────────┘                    │
           └─────────────────────────┼─────────────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   FastAPI REST API   │
                          │   POST /query        │
                          │   POST /ingest       │
                          │   GET /graph/...     │
                          │   GET /stats         │
                          └──────────────────────┘

  Source Credibility Weights:
    research_paper  ████████████████████ 1.00
    internal_report ████████████████▌    0.85
    news_article    ████████████▌        0.60
    email_slack     ████████             0.40
```

---

## Quick Start

### 1. Clone & configure

```bash
cd healthcare_graphrag
cp .env.example .env
# Edit .env — set OPENAI_API_KEY
```

### 2. Start infrastructure

```bash
docker compose up neo4j chromadb -d
# Wait ~30s for Neo4j to initialise
```

### 3. Create sample documents

```bash
python -m scripts.seed_sample_data
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Ingest all documents

```bash
python -m scripts.ingest --data-dir data/
```

### 6. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 7. Run example queries

```bash
# CLI
python -m scripts.query "What are all known drug interactions with Metformin?"

# API
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are all known drug interactions with Metformin?"}' \
  | python -m json.tool
```

---

## Docker (full stack)

```bash
docker compose up --build
# App: http://localhost:8000
# Neo4j browser: http://localhost:7474
# API docs: http://localhost:8000/docs
```

---

## API Reference

### `POST /ingest`
Trigger full ingestion pipeline over `data/` directory.

```json
// Response
{
  "docs_processed": 5,
  "entities_extracted": 147,
  "relationships_mapped": 89,
  "chunks_stored": 63,
  "time_taken": 42.3
}
```

### `POST /query`
Hybrid graph + vector QA.

```json
// Request
{
  "question": "Which drugs interact negatively with Metformin?",
  "doc_types": ["research_paper", "internal_report"],
  "top_k": 5,
  "min_confidence": 0.6
}

// Response
{
  "answer": "⚠️ WARNING: Metformin interacts with...\n\nBased on research papers...",
  "confidence": 0.82,
  "confidence_pct": 82,
  "sources": [{"title": "...", "type": "research_paper", "date": "2024-03-15", "relevant_excerpt": "..."}],
  "graph_entities_used": ["metformin", "t2dm"],
  "warnings": ["⚠️ WARNING: Metformin interacts with furosemide"],
  "disclaimer": "⚕️ DISCLAIMER: This information is for research purposes only..."
}
```

### `GET /graph/drug/{drug_name}/interactions`
All drug interactions from the graph.

### `GET /graph/disease/{disease_name}/full-profile`
Disease profile: symptoms + drugs + genes + clinical trials + comorbidities.

### `GET /graph/cross-document/{entity_name}`
All chunks mentioning an entity across ALL document types.

### `GET /stats`
Node counts by type, edge counts by type, vector chunk count.

### `GET /health`
Neo4j + ChromaDB health status.

---

## Knowledge Graph Schema

### Node Types
| Label | Key Properties |
|---|---|
| `Disease` | name, icd10_code, category |
| `Drug` | name, generic_name, drug_class, fda_status |
| `Gene` | name, symbol, chromosome, function |
| `Protein` | name, function, pathway |
| `Symptom` | name, severity, body_system |
| `Treatment` | name, type, evidence_level |
| `ClinicalTrial` | trial_id, phase, status |
| `Researcher` | name, institution, specialty |
| `Institution` | name, type, location |
| `Document` | id, title, type, date, source |
| `Chunk` | id, text, page_number, doc_id |

### Relationship Types
```
(Drug)-[:TREATS]->(Disease)
(Drug)-[:INTERACTS_WITH]->(Drug)         ← drug safety
(Drug)-[:CAUSES_SIDE_EFFECT]->(Symptom)
(Drug)-[:CONTRAINDICATED_FOR]->(Disease)
(Gene)-[:ASSOCIATED_WITH]->(Disease)
(Gene)-[:ENCODES]->(Protein)
(Protein)-[:INVOLVED_IN]->(Treatment)
(Disease)-[:COMORBID_WITH]->(Disease)
(Disease)-[:HAS_SYMPTOM]->(Symptom)
(Treatment)-[:USED_IN]->(ClinicalTrial)
(Researcher)-[:AUTHORED]->(Document)
(Institution)-[:CONDUCTED]->(ClinicalTrial)
(Chunk)-[:MENTIONS]->(Drug|Disease|Gene|Symptom)
(Chunk)-[:PART_OF]->(Document)
```

---

## Example Queries

### 1. Drug interaction traversal
```bash
python -m scripts.query \
  "What are all known drug interactions with Metformin?"
```
Expected: traverses graph + pulls evidence from `research_paper_1` +
`internal_report_1`. Flags furosemide, contrast media, alcohol interactions.

### 2. Cross-document gene research
```bash
python -m scripts.query \
  "What genes are linked to diseases mentioned in our internal reports?"
```
Expected: `internal_report_1` → Disease (T2DM, Heart Failure) → Gene (AMPK)
from `research_paper_1`. Multi-hop cross-document traversal.

### 3. Entity aggregation across all sources
```bash
python -m scripts.query \
  "Summarize everything we know about semaglutide across all document types"
```
Expected: pulls from `news_article_1` (Ozempic FDA warning) + any research papers.
Results ranked by credibility (research_paper > news_article).

### 4. Cross-document contradiction detection
```bash
python -m scripts.query \
  "Are there any drug interactions mentioned in emails that contradict research papers?"
```
Expected: surfaces warfarin + aspirin interaction from `email_slack_1`, compares
with research evidence.

### 5. Multi-hop treatment pathway
```bash
python -m scripts.query \
  "What treatments have been used in clinical trials for diseases comorbid with T2DM?"
```
Expected: T2DM → COMORBID_WITH → Heart Failure/Breast Cancer → Treatment →
USED_IN → ClinicalTrial (3-hop graph traversal).

---

## Sample Data Cross-Document Graph

```
Metformin ──── research_paper_1 (primary study)
    │────────── internal_report_1 (withheld in heart failure)
    │────────── news_article_1 (combination with Ozempic)
    └────────── email_slack_1 (interaction with warfarin)

T2DM ──────────── ALL 5 documents

AMPK gene ──────── research_paper_1 (metformin mechanism)
    └───────────── research_paper_2 (AMPK-BRCA1 link)

Warfarin ──────── email_slack_1 (atrial fibrillation case)

BRCA1/BRCA2 ──── research_paper_2 (triple-negative breast cancer)

Heart Failure ─── internal_report_1 (Q3 readmission report)
    └───────────── research_paper_1 (metformin contraindication)
```

---

## Project Structure

```
healthcare_graphrag/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── README.md
├── data/
│   ├── research_papers/     ← .pdf, .txt
│   ├── internal_reports/    ← .pdf, .docx, .txt
│   ├── news_articles/       ← .txt, .html
│   └── emails_slack/        ← .txt, .json
└── src/
    ├── utils/
    │   ├── config.py           ← Pydantic Settings
    │   └── neo4j_client.py     ← Async Neo4j driver + schema init
    ├── ingestion/
    │   ├── document_loader.py  ← 4 source types + metadata extraction
    │   ├── chunker.py          ← Medical-aware sentence-boundary chunking
    │   ├── entity_extractor.py ← GPT-4o extraction with Pydantic models
    │   ├── entity_normalizer.py← Drug/disease/gene synonym resolution
    │   └── graph_builder.py    ← Neo4j MERGE (no duplicates)
    ├── embeddings/
    │   └── vector_store.py     ← ChromaDB + OpenAI embeddings
    ├── retrieval/
    │   ├── graph_retriever.py  ← 6 Cypher query templates
    │   ├── vector_retriever.py ← Semantic search + credibility
    │   └── hybrid_retriever.py ← Score fusion (0.4/0.4/0.2)
    ├── generation/
    │   └── answer_generator.py ← GPT-4o with citations + safety
    └── api/
        └── main.py             ← FastAPI endpoints
```

---

## Fallback Behaviour

If Neo4j is unreachable, the system automatically falls back to vector-only search
and includes a warning in the API response:

```json
{
  "warnings": ["⚠️ Graph database unavailable — results based on vector search only"],
  "neo4j_available": false
}
```

Drug interaction warnings from the graph are **always surfaced** regardless of
confidence score.

---

## Safety

- All drug interaction flags use `⚠️ WARNING` prefix
- All answers include: *"This is for research purposes only. Always consult a
  licensed medical professional."*
- If confidence < 60%: returns "Insufficient evidence found across documents"
- GPT-4o is explicitly instructed never to hallucinate dosages or clinical recommendations
- Patient data in documents must be anonymized before ingestion
