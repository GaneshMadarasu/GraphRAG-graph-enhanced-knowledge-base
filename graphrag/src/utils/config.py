"""Configuration management using pydantic-settings."""
import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    chroma_persist_dir: str = "./chroma_db"
    log_level: str = "INFO"

    # Model settings
    extraction_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval settings
    vector_top_k: int = 5
    graph_hop_limit: int = 20


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
