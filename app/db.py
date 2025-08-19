# app/db.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

logger = logging.getLogger("uvicorn.error")

ENGINE: Optional[Engine] = None  # global engine


def _default_sqlite_url() -> str:
    """
    Prefer Render's persistent disk at /var/data if present,
    otherwise fall back to a local project file.
    """
    if Path("/var/data").exists():
        return "sqlite:////var/data/seo_insight.db"
    return "sqlite:///./seo_insight.db"


def _resolve_sqlite_url(candidate_url: str) -> Tuple[str, dict]:
    """
    For SQLite URLs:
      - ensure parent directory exists
      - verify we can write there
      - set check_same_thread=False
      - if not writable, fall back to ./seo_insight.db
    Returns (db_url, connect_args)
    """
    url = make_url(candidate_url)
    connect_args: dict = {}

    if url.get_backend_name() != "sqlite":
        return candidate_url, connect_args

    db_path = url.database  # e.g., /var/data/seo_insight.db or ./seo_insight.db
    connect_args = {"check_same_thread": False}

    if not db_path:
        # Extremely rare, but guard anyway — use local file
        fallback = "sqlite:///./seo_insight.db"
        logger.warning("SQLite path missing in URL. Falling back to %s", fallback)
        return fallback, connect_args

    parent = Path(db_path).parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        # Write test to confirm permissions
        test = parent / ".writable_test"
        with open(test, "w") as f:
            f.write("ok")
        try:
            test.unlink()
        except Exception:
            pass
        return candidate_url, connect_args
    except Exception as e:
        # Permission denied or filesystem read-only: fall back to project-local DB
        fallback = "sqlite:///./seo_insight.db"
        logger.warning(
            "Cannot write to %s (%r). Falling back to %s",
            parent, e, fallback
        )
        # Ensure local dir exists too
        Path(".").mkdir(parents=True, exist_ok=True)
        return fallback, connect_args


def init_db(db_url: str | None = None) -> Engine:
    """
    Initialize the global engine and create tables.
    - Reads DATABASE_URL from env if not provided.
    - Safely handles SQLite directory creation & fallback.
    """
    global ENGINE

    candidate = db_url or os.getenv("DATABASE_URL") or _default_sqlite_url()

    # If SQLite, prepare dir and maybe fall back; else leave as-is
    url = make_url(candidate)
    if url.get_backend_name() == "sqlite":
        resolved_url, connect_args = _resolve_sqlite_url(candidate)
    else:
        resolved_url, connect_args = candidate, {}

    engine = create_engine(
        resolved_url,
        connect_args=connect_args,
        pool_pre_ping=True,  # drops stale connections
    )

    # Import models BEFORE create_all so tables are registered
    try:
        from . import models  # noqa: F401
    except Exception:
        # Fine if you don't have a models module yet.
        pass

    SQLModel.metadata.create_all(engine)
    ENGINE = engine
    logger.info("Database initialized: %s", resolved_url)
    return engine


def get_engine() -> Engine:
    """
    Return the initialized engine (initialize if needed).
    """
    global ENGINE
    if ENGINE is None:
        ENGINE = init_db()
    return ENGINE


def create_session() -> Session:
    """
    Convenience helper: returns a Session. Remember to close it.
    """
    return Session(get_engine())


def get_session() -> Iterator[Session]:
    """
    FastAPI dependency:
        from fastapi import Depends
        @app.get("/items")
        def handler(session: Session = Depends(get_session)):
            ...
    """
    with Session(get_engine()) as session:
        yield session
