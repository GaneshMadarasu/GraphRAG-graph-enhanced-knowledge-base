#!/usr/bin/env python3
"""CLI script to query the GraphRAG system."""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.answer_generator import generate_answer
from src.retrieval.hybrid_retriever import hybrid_retrieve_async

logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner CLI output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _format_context_item(item: dict, idx: int) -> str:
    item_type = item.get("type", "unknown")
    score = item.get("score", 0.0)
    source = item.get("retrieval_source", "unknown")

    if item_type in ("vector_chunk", "graph_chunk"):
        text = item.get("text", "")[:200]
        filename = item.get("metadata", {}).get("filename") or item.get("filename", "")
        return f"  [{idx}] {source.upper()} (score={score:.3f}, {filename})\n      {text}..."
    elif item_type in ("graph_triple", "path_triple"):
        src = item.get("source", "")
        rel = item.get("relation", "")
        tgt = item.get("target", "")
        return f"  [{idx}] GRAPH FACT (score={score:.3f}): {src} --[{rel}]--> {tgt}"
    return f"  [{idx}] {item_type} (score={score:.3f})"


async def async_main(question: str, top_k: int = 5):
    print(f"\n{'='*60}")
    print(f"Query: {question}")
    print(f"{'='*60}\n")

    print("Retrieving context...")
    context_items = await hybrid_retrieve_async(question, top_k=top_k)

    # Display retrieved context
    graph_items = [i for i in context_items if i.get("retrieval_source") == "graph"]
    vector_items = [i for i in context_items if i.get("retrieval_source") == "vector"]

    print(f"\nGraph Context ({len(graph_items)} items):")
    for idx, item in enumerate(graph_items, 1):
        print(_format_context_item(item, idx))

    print(f"\nVector Context ({len(vector_items)} items):")
    for idx, item in enumerate(vector_items, 1):
        print(_format_context_item(item, idx))

    print("\n" + "-"*60)
    print("Generating answer...")
    result = generate_answer(question, context_items)

    print(f"\nAnswer:\n{result.answer}")
    if result.entities_used:
        print(f"\nEntities used: {', '.join(result.entities_used)}")
    if result.sources:
        print(f"Source chunks: {', '.join(s[:8] + '...' for s in result.sources[:5])}")
    print(f"\n{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py \"Your question here\" [top_k]")
        sys.exit(1)

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(async_main(question, top_k))


if __name__ == "__main__":
    main()
