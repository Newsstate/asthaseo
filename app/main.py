# app/main.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Form, HTTPException, Query, Response
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
def _fallback_pagespeed(url: str, fast: bool | None = None) -> dict:
    # Safe placeholder in case seo.py fails to import
    return {"ok": False, "message": "seo.get_pagespeed_data not found", "url": url, "pagespeed": {"enabled": False}}

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

def _to_boolish(v: Optional[str | int | bool]) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return None

# ---------- Routes ----------
@app.head("/")
async def head_root():
    # Silence HEAD / 405 noise in logs
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, fast: Optional[int] = Query(None)):
    # Let query ?fast=1/0 preselect the mode in the UI
    scan_mode = "auto" if fast is None else ("fast" if bool(fast) else "deep")
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "scan_mode": scan_mode})

# Accepts HTML form posts (action points here)
@app.post("/analyze", name="analyze_form", response_class=HTMLResponse)
@app.post("/analyze/", response_class=HTMLResponse)
async def analyze_form(
    request: Request,
    url: Optional[str] = Form(None),
    target_url: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
    fast: Optional[str] = Form(None),  # "1" or "0" (from the select)
):
    target = _normalize_url(url or target_url or website or "")
    if not target:
        raise HTTPException(status_code=400, detail="Missing URL (expected 'url', 'target_url', or 'website')")

    fast_flag = _to_boolish(fast)  # None => use FAST_MODE_DEFAULT in seo.py
    data = await run_in_threadpool(get_pagespeed_data, target, fast_flag)
    scan_mode = "fast" if (fast_flag is True) else ("deep" if (fast_flag is False) else "auto")

    # Render same page with result data + mode for a small badge
    return templates.TemplateResponse("index.html", {"request": request, "result": data, "scan_mode": scan_mode})

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

# JSON API variants
class AnalyzeRequest(BaseModel):
    url: str
    fast: Optional[bool] = None

@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_api(payload: AnalyzeRequest):
    target = _normalize_url(payload.url)
    if not target:
        raise HTTPException(status_code=400, detail="Invalid URL")
    data = await run_in_threadpool(get_pagespeed_data, target, payload.fast)
    return {"url": target, "result": data, "mode": ("fast" if payload.fast else "deep" if payload.fast is False else "auto")}

@app.get("/api/analyze", response_class=JSONResponse)
async def analyze_api_get(url: str = Query(...), fast: Optional[int] = Query(None)):
    target = _normalize_url(url)
    fast_flag = None if fast is None else bool(fast)
    data = await run_in_threadpool(get_pagespeed_data, target, fast_flag)
    return {"url": target, "result": data, "mode": ("fast" if fast_flag else "deep" if fast_flag is False else "auto")}

# --- AMP vs Non-AMP compare page (renders templates/amp_compare.html) ---
@app.get("/amp-compare", response_class=HTMLResponse, name="amp_compare")
async def amp_compare(request: Request, url: str = Query(...)):
    """
    Compare key SEO/performance/meta items between a canonical URL and its AMP variant.
    We use fast=True here so it loads quickly.
    """
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

    try:
        a = await run_in_threadpool(get_pagespeed_data, _normalize_url(url), True)
    except Exception as e:
        return templates.TemplateResponse(
            "amp_compare.html",
            {"request": request, "url": url, "amp_url": None, "rows": [], "error": f"Fetch failed: {e}"}
        )

    non_amp_url = _normalize_url(url)
    amp_url = a.get("amp_url")

    if a.get("is_amp"):
        non_amp_url = a.get("canonical") or non_amp_url
        amp_url = non_amp_url if non_amp_url == url else url

    if not amp_url:
        return templates.TemplateResponse(
            "amp_compare.html",
            {"request": request, "url": non_amp_url, "amp_url": None, "rows": [], "error": "No AMP version found (no <link rel=\"amphtml\">)."}
        )

    try:
        b = await run_in_threadpool(get_pagespeed_data, _normalize_url(amp_url), True)
    except Exception as e:
        return templates.TemplateResponse(
            "amp_compare.html",
            {"request": request, "url": non_amp_url, "amp_url": amp_url, "rows": [], "error": f"AMP fetch failed: {e}"}
        )

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

    add_row("OG Image",
            sget(a, "open_graph", "og:image"),
            sget(b, "open_graph", "og:image"))
    add_row("Twitter Card",
            sget(a, "twitter_card", "twitter:card"),
            sget(b, "twitter_card", "twitter:card"))

    add_row("Indexable",
            sget(a, "checks", "indexable", "value"),
            sget(b, "checks", "indexable", "value"))

    return templates.TemplateResponse(
        "amp_compare.html",
        {"request": request, "url": non_amp_url, "amp_url": amp_url, "rows": rows, "error": None}
    )

@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    return {"status": "ok"}
