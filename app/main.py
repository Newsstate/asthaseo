# app/main.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# ---------- Paths & App ----------
FILE_DIR = Path(__file__).parent          # app/
PROJECT_ROOT = FILE_DIR.parent            # repo root (where /static lives)

app = FastAPI(title="SEO & Performance Dashboard", version="1.0.0")
logger = logging.getLogger("uvicorn.error")

# ---------- Static ----------
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # avoid mount errors if folder missing
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------- Templates ----------
templates = Jinja2Templates(directory=str(FILE_DIR / "templates"))

# ---------- Database (optional) ----------
init_db = None
try:
    from .db import init_db  # type: ignore
except Exception as e:
    logger.warning("DB module not found or failed to import: %r", e)

@app.on_event("startup")
async def on_startup():
    # Initialize DB if available
    if init_db is not None:
        db_url = os.getenv("DATABASE_URL") or (
            "sqlite:////var/data/seo_insight.db" if Path("/var/data").exists() else "sqlite:///./seo_insight.db"
        )
        try:
            init_db(db_url)
            logger.info("Database initialized: %s", db_url)
        except Exception:
            logger.exception("Database initialization failed")
            raise

    # Warn if PageSpeed key missing
    if not os.getenv("PAGESPEED_API_KEY"):
        logger.warning("PAGESPEED_API_KEY not set; PageSpeed features may be disabled")

# ---------- Import analyzer (sync function) ----------
def _fallback_pagespeed(url: str) -> dict:
    # Safe placeholder in case seo.py fails to import
    return {"ok": False, "message": "seo.get_pagespeed_data not found", "url": url}

get_pagespeed_data = _fallback_pagespeed
try:
    from .seo import get_pagespeed_data as _real_get_pagespeed_data  # type: ignore
    get_pagespeed_data = _real_get_pagespeed_data
except Exception as e:
    logger.warning("seo.get_pagespeed_data unavailable, using fallback: %r", e)

# ---------- Helpers ----------
def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    parsed = urlparse(u)
    if not parsed.scheme:
        u = "https://" + u
    return u

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render the main dashboard with empty result
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Accepts HTML form posts (action points here)
@app.post("/analyze", name="analyze_form", response_class=HTMLResponse)
@app.post("/analyze/", response_class=HTMLResponse)
async def analyze_form(
    request: Request,
    url: Optional[str] = Form(None),
    target_url: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
):
    target = _normalize_url(url or target_url or website or "")
    if not target:
        raise HTTPException(status_code=400, detail="Missing URL (expected 'url', 'target_url', or 'website')")

    # Run analyzer in a thread so we don't block the event loop
    data = await run_in_threadpool(get_pagespeed_data, target)

    # Render same page with result data
    return templates.TemplateResponse("index.html", {"request": request, "result": data})

# GET /analyze -> redirect home (avoid 404 when someone hits it directly)
@app.get("/analyze")
@app.get("/analyze/")
async def analyze_get_redirect():
    return RedirectResponse(url="/", status_code=307)

# Handle common typo '/analyz' (no 'e')
@app.get("/analyz")
@app.get("/analyz/")
async def analyz_get_redirect():
    return RedirectResponse(url="/", status_code=307)

@app.post("/analyz")
@app.post("/analyz/")
async def analyz_post_redirect(url: Optional[str] = Form(None)):
    if not url:
        return RedirectResponse(url="/", status_code=307)
    html = f"""<!doctype html><html><body>
      <form id="f" action="/analyze" method="post">
        <input type="hidden" name="url" value="{url}"/>
      </form>
      <script>document.getElementById('f').submit();</script>
    </body></html>"""
    return HTMLResponse(html)

# JSON API variant
class AnalyzeRequest(BaseModel):
    url: str

@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_api(payload: AnalyzeRequest):
    target = _normalize_url(payload.url)
    if not target:
        raise HTTPException(status_code=400, detail="Invalid URL")
    data = await run_in_threadpool(get_pagespeed_data, target)
    return {"url": target, "result": data}

# AMP Compare helper (optional)
@app.get("/amp-compare", name="amp_compare", response_class=HTMLResponse)
async def amp_compare(request: Request, url: str):
    try:
        data = await run_in_threadpool(get_pagespeed_data, _normalize_url(url))
        amp_url = (data or {}).get("amp_url")
    except Exception:
        amp_url = None

    body = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"><title>AMP Compare</title></head>
    <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 16px;">
      <h1 style="margin: 0 0 12px;">AMP vs Canonical</h1>
      <p>Opening pages in new tabs… If nothing opened, use the links below.</p>
      <ul>
        <li><a href="{url}" target="_blank" rel="noopener">Canonical</a></li>
        {("<li><a href='" + amp_url + "' target='_blank' rel='noopener'>AMP</a></li>") if amp_url else ""}
      </ul>
      <script>
        try {{
          window.open("{url}", "_blank");
          {"window.open('" + amp_url + "', '_blank');" if amp_url else ""}
        }} catch (e) {{}}
      </script>
    </body></html>"""
    return HTMLResponse(content=body)

@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    return {"status": "ok"}
