# app/main.py
from __future__ import annotations

import os
import html
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Form, HTTPException, Query
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
def _fallback_pagespeed(url: str, fast: Optional[bool] = None) -> dict:
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

def _fmt_cell(v):
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return " | ".join([str(x) for x in v[:5]])
    return str(v)

def _sget(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default

# ---------- Routes ----------
@app.get("/", name="home", response_class=HTMLResponse)
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
    fast: Optional[str] = Form(None),
):
    target = _normalize_url(url or target_url or website or "")
    if not target:
        raise HTTPException(status_code=400, detail="Missing URL (expected 'url', 'target_url', or 'website')")

    # Toggle fast vs deep from form checkbox (values like "on" or "1")
    fast_mode = bool(fast) and fast not in ("0", "false", "False")

    # Run analyzer in a thread so we don't block the event loop
    data = await run_in_threadpool(get_pagespeed_data, target, fast_mode)

    # Render same page with result data
    return templates.TemplateResponse("index.html", {"request": request, "result": data, "fast": fast_mode})

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
    html_doc = f"""<!doctype html><html><body>
      <form id="f" action="/analyze" method="post">
        <input type="hidden" name="url" value="{html.escape(url)}"/>
      </form>
      <script>document.getElementById('f').submit();</script>
    </body></html>"""
    return HTMLResponse(html_doc)

# JSON API variant
class AnalyzeRequest(BaseModel):
    url: str
    fast: Optional[bool] = None

@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_api(payload: AnalyzeRequest):
    target = _normalize_url(payload.url)
    if not target:
        raise HTTPException(status_code=400, detail="Invalid URL")
    data = await run_in_threadpool(get_pagespeed_data, target, payload.fast)
    return {"url": target, "result": data, "fast": bool(payload.fast)}

# --- AMP vs Non-AMP compare page (renders app/templates/amp_compare.html) ---
@app.get("/amp-compare", response_class=HTMLResponse, name="amp_compare")
async def amp_compare(request: Request, url: str = Query(..., description="Canonical or AMP URL to compare")):
    """
    Compare key SEO/performance/meta items between a canonical URL and its AMP variant.
    Uses fast mode to keep it responsive.
    """
    target = _normalize_url(url)

    # Fetch the initial URL (fast mode)
    try:
        a = await run_in_threadpool(get_pagespeed_data, target, True)
    except Exception as e:
        ctx = {"request": request, "url": target, "amp_url": None, "rows": [], "error": f"Fetch failed: {e}"}
        try:
            return templates.TemplateResponse("amp_compare.html", ctx)
        except Exception:
            return HTMLResponse(f"<h1>AMP Compare</h1><p>{html.escape(ctx['error'])}</p>", status_code=200)

    # Decide roles
    non_amp_url = target
    amp_url = a.get("amp_url")
    if a.get("is_amp"):
        non_amp_url = a.get("canonical") or target
        amp_url = target

    if not amp_url:
        ctx = {"request": request, "url": non_amp_url, "amp_url": None, "rows": [], "error": "No AMP version found (no <link rel=\"amphtml\">)."}
        try:
            return templates.TemplateResponse("amp_compare.html", ctx)
        except Exception:
            return HTMLResponse(f"<h1>AMP Compare</h1><p>{html.escape(ctx['error'])}</p>", status_code=200)

    # Fetch AMP page (fast)
    try:
        b = await run_in_threadpool(get_pagespeed_data, _normalize_url(amp_url), True)
    except Exception as e:
        ctx = {"request": request, "url": non_amp_url, "amp_url": amp_url, "rows": [], "error": f"AMP fetch failed: {e}"}
        try:
            return templates.TemplateResponse("amp_compare.html", ctx)
        except Exception:
            return HTMLResponse(f"<h1>AMP Compare</h1><p>{html.escape(ctx['error'])}</p>", status_code=200)

    # Build comparison rows
    rows = []
    def add_row(label, val_a, val_b):
        sa, sb = _fmt_cell(val_a), _fmt_cell(val_b)
        rows.append({"label": label, "non_amp": sa, "amp": sb, "changed": (sa != sb)})

    add_row("HTTP Status", a.get("status_code"), b.get("status_code"))
    add_row("Title", a.get("title"), b.get("title"))
    add_row("Meta Description", a.get("description"), b.get("description"))
    add_row("Canonical", a.get("canonical"), b.get("canonical"))
    add_row("Robots Meta", a.get("robots_meta"), b.get("robots_meta"))
    add_row("H1", _sget(a, "h1", default=[]), _sget(b, "h1", default=[]))

    # Performance
    add_row("Load Time (ms)",
            _sget(a, "performance", "load_time_ms", default=a.get("load_time_ms")),
            _sget(b, "performance", "load_time_ms", default=b.get("load_time_ms")))
    add_row("Page Size (bytes)",
            _sget(a, "performance", "page_size_bytes", default=a.get("content_length")),
            _sget(b, "performance", "page_size_bytes", default=b.get("content_length")))
    add_row("PSI Mobile Score",
            _sget(a, "performance", "mobile_score"),
            _sget(b, "performance", "mobile_score"))
    add_row("PSI Desktop Score",
            _sget(a, "performance", "desktop_score"),
            _sget(b, "performance", "desktop_score"))

    # Social meta
    add_row("OG Image",
            _sget(a, "open_graph", "og:image"),
            _sget(b, "open_graph", "og:image"))
    add_row("Twitter Card",
            _sget(a, "twitter_card", "twitter:card"),
            _sget(b, "twitter_card", "twitter:card"))

    # Indexability
    add_row("Indexable",
            _sget(a, "checks", "indexable", "value"),
            _sget(b, "checks", "indexable", "value"))

    ctx = {"request": request, "url": non_amp_url, "amp_url": amp_url, "rows": rows, "error": None}
    try:
        return templates.TemplateResponse("amp_compare.html", ctx)
    except Exception:
        # Inline fallback if template missing
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
        <p><a href="/" style="text-decoration:underline">← Back to Analyzer</a></p>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>Metric</th><th>Non-AMP</th><th>AMP</th><th>Changed</th></tr>
          {items}
        </table>
        """
        return HTMLResponse(inline, status_code=200)

@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    return {"status": "ok"}
