---
name: query
description: Query the GraphRAG knowledge base using hybrid graph + vector retrieval
---

# Query the Knowledge Base

Send a question to the GraphRAG pipeline and return a grounded answer using hybrid retrieval (60% vector + 40% graph).

## Usage

The user will provide a question. Optionally they may specify `top_k` (default: 5).

## Steps

1. Check if the API server is running:
```bash
curl -s http://localhost:8000/health
```

2. **If API is running**, query via the endpoint:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "<user question>", "top_k": 5}'
```

3. **If API is not running**, use the CLI script:
```bash
cd graphrag && python scripts/query.py "<user question>" 5
```

4. Present the answer clearly, including:
   - The generated answer
   - Source chunks cited
   - Entities referenced from the graph

## Notes
- Answers are grounded only in retrieved context — do not add external knowledge
- Graph retrieval uses spaCy NER to extract entities from the question
- If results seem poor, suggest the user ingest more relevant documents first
