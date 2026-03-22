"""
Configuration management for Healthcare GraphRAG.
All settings loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("healthcare_password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_host: str = Field("localhost", env="CHROMA_HOST")
    chroma_port: int = Field(8001, env="CHROMA_PORT")
    chroma_collection: str = Field(
        "healthcare_chunks", env="CHROMA_COLLECTION"
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(150, env="CHUNK_OVERLAP")
    entity_extraction_confidence_threshold: float = Field(
        0.5, env="ENTITY_CONFIDENCE_THRESHOLD"
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = Field(5, env="DEFAULT_TOP_K")
    min_confidence: float = Field(0.6, env="MIN_CONFIDENCE")

    # ── Data paths ────────────────────────────────────────────────────────────
    data_dir: str = Field("data", env="DATA_DIR")

    # ── Source credibility weights ────────────────────────────────────────────
    credibility_research_paper: float = 1.0
    credibility_internal_report: float = 0.85
    credibility_news_article: float = 0.6
    credibility_email_slack: float = 0.4

    # ── Score fusion weights ──────────────────────────────────────────────────
    vector_weight: float = 0.4
    graph_weight: float = 0.4
    credibility_weight: float = 0.2

    @field_validator("openai_api_key")
    @classmethod
    def api_key_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("OPENAI_API_KEY must not be empty")
        return v

    def get_credibility(self, source_type: str) -> float:
        mapping = {
            "research_paper": self.credibility_research_paper,
            "internal_report": self.credibility_internal_report,
            "news_article": self.credibility_news_article,
            "email_slack": self.credibility_email_slack,
        }
        return mapping.get(source_type, 0.5)

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
