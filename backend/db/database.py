"""
Setup SQLAlchemy 2.0 per NeuralForge.

- Engine SQLite con check_same_thread=False
- DeclarativeBase esportata come Base
- get_session() come dependency FastAPI
- init_db() chiamata al startup
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class per tutti i modelli ORM di NeuralForge."""


engine: Engine = create_engine(
    settings.database_url_resolved,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_fk(dbapi_connection: Any, _: Any) -> None:
    """Abilita foreign keys su SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def init_db() -> None:
    """Crea tutte le tabelle. Idempotente."""
    settings.ensure_directories()
    # In M2+ verranno aggiunti qui gli import dei modelli ORM:
    # from db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database inizializzato: %s", settings.database_url_resolved)


def get_session() -> Iterator[Session]:
    """Dependency FastAPI."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()