# app/models.py
from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from sqlalchemy.types import JSON

class Analysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_amp: bool = False
    load_time_ms: int = 0
    content_length: int = 0
    status_code: int = 0
    # Store the parsed analysis dict as a real JSON column
    result: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

class ScheduledScan(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str
    user_email: Optional[str] = None
    frequency: str = "daily"  # daily | weekly
    cron: Optional[str] = None
    timezone: Optional[str] = None
    last_run_at: Optional[datetime] = None
