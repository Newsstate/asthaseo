# app/main.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, Request, Form, HTTPException, Query


# Basic logging so we see import/scan errors in Render logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")

# ---------- Paths & App ----------
FILE_DIR = Path(__file__).parent          # /app
PROJECT_ROOT = FILE_DIR.parent            # repo root

app = FastAPI(title="SEO & Performance Dashboard", version="1.2.0")

# ---------- Static ----------
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # avoid mount errors if missing
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
def _fallback_pagespeed(url: str, fast: Optional[bool] = None) -> dict:
    # Safe placeholder if seo.py fails to import; keep signature compatible
    return {
        "ok": False,
        "message": "seo.get_pagespeed_data not found (using fallback)",
        "url": url,
        "fast": fast,
        "errors": ["Analyzer not loaded: check seo.py import errors in logs"],
        "pagespeed": {"enabled": False},
    }

get_pagespeed_data = _fallback_pagespeed
try:
    from .seo import get_pagespeed_data as _real_get_pagespeed_data  # type: ignore
    get_pagespeed_data = _real_get_pagespeed_data
    logger.info("seo.get_pagespeed_data loaded successfully")
except Exception as e:
    logger.error("seo.get_pagespeed_data unavailable, using fallback: %r", e)

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
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "scan_mode": "auto"})

# POST form analyzer (preferred by your UI)
@app.post("/analyze", name="analyze_form", response_class=HTMLResponse)
@app.post("/analyze/", response_class=HTMLResponse)
async def analyze_form(
    request: Request,
    url: Optional[str] = Form(None),
    target_url: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
    fast: Optional[str] = Form(None),   # "1" for fast, "0" for deep, None => default
):
    target = _normalize_url(url or target_url or website or "")
    if not target:
        raise HTTPException(status_code=400, detail="Missing URL (expected 'url', 'target_url', or 'website')")

    fast_bool: Optional[bool] = None if fast is None else (fast == "1")

    try:
        data = await run_in_threadpool(get_pagespeed_data, target, fast_bool)
    except Exception as e:
        logger.exception("Analyze failed")
        data = {"url": target, "errors": [str(e)], "pagespeed": {"enabled": False, "message": str(e)}}

    mode_label = "fast" if fast_bool is True else ("deep" if fast_bool is False else "auto")
    return templates.TemplateResponse("index.html", {"request": request, "result": data, "scan_mode": mode_label})

# Optional GET analyzer: /analyze?url=...&fast=1
@app.get("/analyze", response_class=HTMLResponse)
@app.get("/analyze/", response_class=HTMLResponse)
async def analyze_get(
    request: Request,
    url: Optional[str] = Query(None),
    fast: Optional[int] = Query(None, description="1=fast, 0=deep"),
):
    if not url:
        return RedirectResponse(url="/", status_code=307)

    target = _normalize_url(url)
    fast_bool: Optional[bool] = None if fast is None else (fast == 1)

    try:
        data = await run_in_threadpool(get_pagespeed_data, target, fast_bool)
    except Exception as e:
        logger.exception("Analyze (GET) failed")
        data = {"url": target, "errors": [str(e)], "pagespeed": {"enabled": False, "message": str(e)}}

    mode_label = "fast" if fast_bool is True else ("deep" if fast_bool is False else "auto")
    return templates.TemplateResponse("index.html", {"request": request, "result": data, "scan_mode": mode_label})

# Typo handler
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

# JSON API
class AnalyzeRequest(BaseModel):
    url: str
    fast: Optional[bool] = None

@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_api(payload: AnalyzeRequest):
    target = _normalize_url(payload.url)
    if not target:
        raise HTTPException(status_code=400, detail="Invalid URL")
    data = await run_in_threadpool(get_pagespeed_data, target, payload.fast)
    return {"url": target, "result": data, "fast": payload.fast}

# --- AMP vs Non-AMP compare page (uses template if present, otherwise inline fallback) ---
@app.get("/amp-compare", response_class=HTMLResponse, name="amp_compare")
async def amp_compare(request: Request, url: str = Query(...)):
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, (list, tuple)):
            return " | ".join([str(x) for x in v[:5]])
        return str(v)

    def sget(d, *keys, default=None):
        cur = d
        for k in keys:
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                return default
        return cur if cur is not None else default

    # Fetch canonical (fast mode)
    target = _normalize_url(url)
    try:
        a = await run_in_threadpool(get_pagespeed_data, target, True)
    except Exception as e:
        return HTMLResponse(f"<h1>AMP Compare</h1><p>Fetch failed: {html.escape(str(e))}</p>", status_code=500)

    # Decide roles
    non_amp_url = target
    amp_url = a.get("amp_url")
    if a.get("is_amp"):
        # If input is AMP, swap roles
        canonical = a.get("canonical")
        if canonical:
            non_amp_url = canonical
            amp_url = target

    if not amp_url:
        # Render template if available, otherwise inline
        ctx = {"request": request, "url": non_amp_url, "amp_url": None, "rows": [], "error": 'No AMP version found (no <link rel="amphtml">).'}
        try:
            return templates.TemplateResponse("amp_compare.html", ctx)
        except Exception:
            return HTMLResponse(f"<h1>AMP vs Non-AMP</h1><p>No AMP version found for <a href='{non_amp_url}' target='_blank'>{non_amp_url}</a>.</p>", status_code=200)

    # Fetch AMP (fast mode)
    try:
        b = await run_in_threadpool(get_pagespeed_data, _normalize_url(amp_url), True)
    except Exception as e:
        ctx = {"request": request, "url": non_amp_url, "amp_url": amp_url, "rows": [], "error": f"AMP fetch failed: {e}"}
        try:
            return templates.TemplateResponse("amp_compare.html", ctx)
        except Exception:
            return HTMLResponse(f"<h1>AMP vs Non-AMP</h1><p>AMP fetch failed: {html.escape(str(e))}</p>", status_code=200)

    # Build comparison rows
    rows = []
    def add_row(label, val_a, val_b):
        sa, sb = fmt(val_a), fmt(val_b)
        rows.append({"label": label, "non_amp": sa, "amp": sb, "changed": (sa != sb)})

    add_row("HTTP Status", a.get("status_code"), b.get("status_code"))
    add_row("Title", a.get("title"), b.get("title"))
    add_row("Meta Description", a.get("description"), b.get("description"))
    add_row("Canonical", a.get("canonical"), b.get("canonical"))
    add_row("Robots Meta", a.get("robots_meta"), b.get("robots_meta"))
    add_row("H1", sget(a, "h1", default=[]), sget(b, "h1", default=[]))

    # Performance
    add_row("Load Time (ms)",
            sget(a, "performance", "load_time_ms", default=a.get("load_time_ms")),
            sget(b, "performance", "load_time_ms", default=b.get("load_time_ms")))
    add_row("Page Size (bytes)",
            sget(a, "performance", "page_size_bytes", default=a.get("content_length")),
            sget(b, "performance", "page_size_bytes", default=b.get("content_length")))
    add_row("PSI Mobile Score",
            sget(a, "performance", "mobile_score"),
            sget(b, "performance", "mobile_score"))
    add_row("PSI Desktop Score",
            sget(a, "performance", "desktop_score"),
            sget(b, "performance", "desktop_score"))

    # Social meta
    add_row("OG Image", sget(a, "open_graph", "og:image"), sget(b, "open_graph", "og:image"))
    add_row("Twitter Card", sget(a, "twitter_card", "twitter:card"), sget(b, "twitter_card", "twitter:card"))

    # Indexability
    add_row("Indexable", sget(a, "checks", "indexable", "value"), sget(b, "checks", "indexable", "value"))

    # Try to render template; if missing, inline fallback
    ctx = {"request": request, "url": non_amp_url, "amp_url": amp_url, "rows": rows, "error": None}
    try:
        return templates.TemplateResponse("amp_compare.html", ctx)
    except Exception:
        # Minimal inline UI
        items = "".join(
            f"<tr><td>{html.escape(r['label'])}</td>"
            f"<td>{html.escape(str(r['non_amp']))}</td>"
            f"<td>{html.escape(str(r['amp']))}</td>"
            f"<td>{'Changed' if r['changed'] else 'Same'}</td></tr>"
            for r in rows
        )
        inline = f"""
        <!doctype html><meta charset="utf-8">
        <title>AMP vs Non-AMP</title>
        <h1>AMP vs Non-AMP</h1>
        <p><a href="{non_amp_url}" target="_blank">Open Non-AMP</a> |
           <a href="{amp_url}" target="_blank">Open AMP</a></p>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>Metric</th><th>Non-AMP</th><th>AMP</th><th>Changed</th></tr>
          {items}
        </table>
        """
        return HTMLResponse(inline, status_code=200)


# Health & diagnostics
@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    ok = get_pagespeed_data is not _fallback_pagespeed
    return {"status": "ok", "seo_imported": ok}

@app.get("/_diag", response_class=JSONResponse)
async def diag():
    """Quick tip: open /_diag to confirm seo.py actually loaded."""
    ok = get_pagespeed_data is not _fallback_pagespeed
    return {"seo_imported": ok}

@app.get("/_debug/analyze", response_class=JSONResponse)
async def debug_analyze(url: str = Query(...), fast: Optional[int] = Query(None)):
    """Bypass templates and see raw JSON from the analyzer."""
    target = _normalize_url(url)
    fast_bool = None if fast is None else (fast == 1)
    data = await run_in_threadpool(get_pagespeed_data, target, fast_bool)
    return {"url": target, "fast": fast_bool, "result": data}
