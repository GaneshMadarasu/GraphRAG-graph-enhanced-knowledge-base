"""
Neo4j client with connection pooling, health checks, and schema initialisation.
All driver operations are wrapped so the rest of the app never touches neo4j-driver directly.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# ── Constraints / indexes created on first connect ───────────────────────────
_SCHEMA_STATEMENTS = [
    # Uniqueness constraints (also create implicit indexes)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug)           REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (dis:Disease)      REQUIRE dis.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene)           REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein)        REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom)        REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment)      REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (ct:ClinicalTrial) REQUIRE ct.trial_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher)     REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution)    REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (doc:Document)     REQUIRE doc.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)          REQUIRE c.id IS UNIQUE",
    # Full-text indexes for fuzzy search
    "CREATE FULLTEXT INDEX IF NOT EXISTS entity_name_fts FOR (n:Drug|Disease|Gene|Protein|Symptom|Treatment) ON EACH [n.name]",
]


class Neo4jClient:
    """Async Neo4j client.  Instantiate once, reuse across the app lifetime."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._driver: Optional[AsyncDriver] = None
        self._schema_initialised = False

    async def connect(self) -> None:
        if self._driver is not None:
            return
        cfg = self._settings
        try:
            self._driver = AsyncGraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_user, cfg.neo4j_password),
                max_connection_pool_size=50,
            )
            await self._driver.verify_connectivity()
            logger.info("Neo4j connected: %s", cfg.neo4j_uri)
            await self._ensure_schema()
        except (ServiceUnavailable, AuthError) as exc:
            logger.error("Neo4j connection failed: %s", exc)
            self._driver = None
            raise

    async def _ensure_schema(self) -> None:
        if self._schema_initialised:
            return
        async with self.session() as s:
            for stmt in _SCHEMA_STATEMENTS:
                try:
                    await s.run(stmt)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Schema statement failed (non-fatal): %s | %s", stmt, exc)
        self._schema_initialised = True
        logger.info("Neo4j schema initialised")

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialised — call connect() first")
        async with self._driver.session(
            database=self._settings.neo4j_database
        ) as session:
            yield session

    # ── Convenience helpers ───────────────────────────────────────────────────

    async def run_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a read/write query, return list of record dicts."""
        async with self.session() as s:
            result = await s.run(cypher, params or {})
            records = await result.data()
            return records

    async def run_write(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run a write query (no return value needed)."""
        async with self.session() as s:
            await s.run(cypher, params or {})

    async def run_batch_write(
        self,
        cypher: str,
        batch: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """Execute a parameterised write query in batches using UNWIND."""
        for i in range(0, len(batch), batch_size):
            chunk = batch[i : i + batch_size]
            async with self.session() as s:
                await s.run(cypher, {"batch": chunk})

    async def health_check(self) -> bool:
        try:
            if not self._driver:
                return False
            result = await self.run_query("RETURN 1 AS ok")
            return bool(result)
        except Exception:
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Return node / relationship counts grouped by label / type."""
        node_counts = await self.run_query(
            """
            CALL apoc.meta.stats()
            YIELD labels
            RETURN labels
            """
        )
        # Fallback without APOC
        if not node_counts:
            node_counts = await self.run_query(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
                """
            )
        rel_counts = await self.run_query(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
            """
        )
        return {"nodes": node_counts, "relationships": rel_counts}


# ── Module-level singleton ────────────────────────────────────────────────────
_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
