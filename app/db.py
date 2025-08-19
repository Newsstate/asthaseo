# app/db.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import SQLModel, Session, create_engine, select
from .models import Analysis, ScheduledScan

ENGINE = None

def init_db(db_url: str = "sqlite:///seo_insight.db"):
    global ENGINE
    if ENGINE is None:
        ENGINE = create_engine(db_url, echo=False)
    SQLModel.metadata.create_all(ENGINE)

def _session() -> Session:
    assert ENGINE is not None, "DB not initialized. Call init_db() at startup."
    return Session(ENGINE)

# ---------- Analyses ----------
def save_analysis(
    *,
    url: str,
    result: Dict[str, Any],
    status_code: int,
    load_time_ms: int,
    content_length: int,
    is_amp: bool,
) -> Analysis:
    """Create and persist an Analysis row, return the saved row."""
    with _session() as s:
        row = Analysis(
            url=url,
            created_at=datetime.utcnow(),
            is_amp=is_amp,
            load_time_ms=load_time_ms,
            content_length=content_length,
            status_code=status_code,
            result=result,  # JSON column
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row

def list_analyses(limit: int = 50) -> List[Analysis]:
    with _session() as s:
        stmt = select(Analysis).order_by(Analysis.id.desc()).limit(limit)
        return list(s.exec(stmt))

# ---------- Scheduled scans (simple stubs) ----------
def create_scheduled(
    *,
    url: str,
    frequency: str = "daily",
    user_email: Optional[str] = None,
    cron: Optional[str] = None,
    timezone: Optional[str] = None,
) -> ScheduledScan:
    with _session() as s:
        row = ScheduledScan(
            url=url,
            frequency=frequency,
            user_email=user_email,
            cron=cron,
            timezone=timezone,
            last_run_at=None,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row

def list_scheduled() -> List[ScheduledScan]:
    with _session() as s:
        stmt = select(ScheduledScan).order_by(ScheduledScan.id.desc())
        return list(s.exec(stmt))

def delete_scheduled(id: int) -> None:
    with _session() as s:
        obj = s.get(ScheduledScan, id)
        if obj:
            s.delete(obj)
            s.commit()
