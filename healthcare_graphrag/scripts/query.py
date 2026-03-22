"""
CLI query script.

Usage:
    python -m scripts.query "What are all known drug interactions with Metformin?"
    python -m scripts.query --doc-types research_paper internal_report "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.vector_store import get_vector_store
from src.generation.answer_generator import MedicalAnswerGenerator
from src.retrieval.graph_retriever import MedicalGraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.utils.neo4j_client import get_neo4j_client

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ALL_DOC_TYPES = ["research_paper", "internal_report", "news_article", "email_slack"]


async def run_query(
    question: str,
    doc_types: list,
    top_k: int,
    min_confidence: float,
    json_output: bool,
) -> None:
    neo4j = get_neo4j_client()
    neo4j_ok = False

    try:
        await neo4j.connect()
        neo4j_ok = True
    except Exception:
        pass

    vs = get_vector_store()
    vector_ret = VectorRetriever(vs)
    graph_ret = MedicalGraphRetriever(neo4j) if neo4j_ok else None
    retriever = HybridRetriever(vector=vector_ret, graph=graph_ret)
    generator = MedicalAnswerGenerator()

    retrieval = await retriever.retrieve(
        question=question,
        top_k=top_k,
        doc_types=doc_types,
        min_confidence=min_confidence,
    )
    answer = await generator.generate(question=question, retrieval_output=retrieval)

    if json_output:
        print(json.dumps(answer.model_dump(), indent=2))
    else:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)
        print(f"\n{answer.answer}\n")
        print(f"Confidence: {answer.confidence_pct}%")
        if not neo4j_ok:
            print("⚠️  Neo4j unavailable — vector-only results")
        print("\nSOURCES:")
        for i, src in enumerate(answer.sources, 1):
            print(f"  [{i}] {src.title} ({src.type}) — {src.date or 'n/d'}")
        if answer.warnings:
            print("\nWARNINGS:")
            for w in answer.warnings:
                print(f"  {w}")
        if answer.graph_entities_used:
            print(f"\nGraph entities queried: {', '.join(answer.graph_entities_used)}")
        print(f"\n{answer.disclaimer}")
        print("=" * 70 + "\n")

    await neo4j.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare GraphRAG query interface")
    parser.add_argument("question", help="Medical question to answer")
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=ALL_DOC_TYPES,
        choices=ALL_DOC_TYPES,
        help="Filter by document types",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    asyncio.run(
        run_query(
            question=args.question,
            doc_types=args.doc_types,
            top_k=args.top_k,
            min_confidence=args.min_confidence,
            json_output=args.json,
        )
    )


if __name__ == "__main__":
    main()
