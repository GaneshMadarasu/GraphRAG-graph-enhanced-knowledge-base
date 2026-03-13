"""Neo4j connection singleton with retry logic."""
import logging
from contextlib import contextmanager
from typing import Generator

from neo4j import GraphDatabase, Driver, Session
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_driver: Driver | None = None


def get_driver() -> Driver:
    """Return (or create) the Neo4j driver singleton."""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j driver created: %s", settings.neo4j_uri)
    return _driver


def close_driver() -> None:
    """Close the Neo4j driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager yielding a Neo4j session."""
    driver = get_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def verify_connectivity() -> bool:
    """Verify Neo4j is reachable. Returns True on success."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return True
    except Exception as exc:
        logger.warning("Neo4j connectivity check failed: %s", exc)
        raise


def is_available() -> bool:
    """Return True if Neo4j is currently reachable."""
    try:
        return verify_connectivity()
    except Exception:
        return False
