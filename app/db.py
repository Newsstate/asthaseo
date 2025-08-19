# app/db.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url


ENGINE: Optional[Engine] = None  # global engine


def _default_sqlite_url() -> str:
    """
    Prefer Render's persistent disk at /var/data if present,
    otherwise fall back to a local project file.
    """
    if Path("/var/data").exists():
        return "sqlite:////var/data/seo_insight.db"
    return "sqlite:///./seo_insight.db"


def _prepare_sqlite(db_url: str) -> dict:
    """
    For SQLite: ensure the parent directory exists and set connect args.
    """
    url = make_url(db_url)
    connect_args: dict = {}
    if url.get_backend_name() == "sqlite":
        db_path = url.database  # e.g., /var/data/seo_insight.db or ./seo_insight.db
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Required for SQLite when used in a multi-threaded app server
        connect_args = {"check_same_thread": False}
    return connect_args


def init_db(db_url: str | None = None) -> Engine:
    """
    Initialize the global engine and create tables.
    - Reads DATABASE_URL from env if not provided.
    - Safely handles SQLite directory creation.
    """
    global ENGINE

    resolved_url = db_url or os.getenv("DATABASE_URL") or _default_sqlite_url()
    connect_args = _prepare_sqlite(resolved_url)

    engine = create_engine(
        resolved_url,
        connect_args=connect_args,
        pool_pre_ping=True,  # drops stale connections
    )

    # Import your models BEFORE create_all so tables get registered
    try:
        from . import models  # noqa: F401
    except Exception:
        # It's fine if you don't have a models module yet.
        pass

    SQLModel.metadata.create_all(engine)
    ENGINE = engine
    return engine


def get_engine() -> Engine:
    """
    Return the initialized engine (initialize it if needed).
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
