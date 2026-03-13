"""Hybrid retriever: merges graph + vector results with score fusion."""
import asyncio
import logging
from typing import List

from src.retrieval.graph_retriever import retrieve_from_graph
from src.retrieval.vector_retriever import retrieve_from_vector

logger = logging.getLogger(__name__)

# Weights for score fusion
VECTOR_WEIGHT = 0.6
GRAPH_WEIGHT = 0.4


def _deduplicate(items: List[dict]) -> List[dict]:
    """Remove duplicate chunk_ids, keeping highest score."""
    seen: dict[str, dict] = {}
    for item in items:
        key = item.get("chunk_id") or item.get("text", "")[:100]
        if key not in seen or item["score"] > seen[key]["score"]:
            seen[key] = item
    return list(seen.values())


def _fuse_scores(
    vector_results: List[dict], graph_results: List[dict]
) -> List[dict]:
    """Combine vector and graph scores into a unified relevance score."""
    fused = []

    # Normalize vector scores (already 0-1 cosine similarity)
    for item in vector_results:
        item = dict(item)
        item["raw_score"] = item["score"]
        item["score"] = item["score"] * VECTOR_WEIGHT
        item["retrieval_source"] = "vector"
        fused.append(item)

    # Graph items: scores already 0.75-0.9, apply graph weight
    for item in graph_results:
        item = dict(item)
        item["raw_score"] = item["score"]
        item["score"] = item["score"] * GRAPH_WEIGHT
        item["retrieval_source"] = "graph"
        fused.append(item)

    return fused


async def _async_retrieve_graph(query: str, top_k: int) -> List[dict]:
    """Run graph retrieval in an executor to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retrieve_from_graph, query, top_k)


async def _async_retrieve_vector(query: str, top_k: int) -> List[dict]:
    """Run vector retrieval in an executor to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retrieve_from_vector, query, top_k)


async def hybrid_retrieve_async(
    query: str, top_k: int = 5
) -> List[dict]:
    """Retrieve from both graph and vector stores in parallel, then merge."""
    graph_task = asyncio.create_task(_async_retrieve_graph(query, top_k))
    vector_task = asyncio.create_task(_async_retrieve_vector(query, top_k))

    graph_results, vector_results = await asyncio.gather(
        graph_task, vector_task, return_exceptions=True
    )

    # Handle exceptions gracefully
    if isinstance(graph_results, Exception):
        logger.warning("Graph retrieval error: %s", graph_results)
        graph_results = []
    if isinstance(vector_results, Exception):
        logger.warning("Vector retrieval error: %s", vector_results)
        vector_results = []

    fused = _fuse_scores(vector_results, graph_results)
    deduped = _deduplicate(fused)
    sorted_results = sorted(deduped, key=lambda x: x["score"], reverse=True)
    top_results = sorted_results[:top_k]

    logger.info(
        "Hybrid retrieval: %d graph + %d vector -> %d merged (top %d)",
        len(graph_results),
        len(vector_results),
        len(deduped),
        len(top_results),
    )
    return top_results


def hybrid_retrieve(query: str, top_k: int = 5) -> List[dict]:
    """Synchronous wrapper for hybrid_retrieve_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, hybrid_retrieve_async(query, top_k)
                )
                return future.result()
        else:
            return loop.run_until_complete(hybrid_retrieve_async(query, top_k))
    except Exception as exc:
        logger.error("Hybrid retrieval failed: %s", exc)
        # Fall back to vector-only
        return retrieve_from_vector(query, top_k)
