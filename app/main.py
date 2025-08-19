# app/main.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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

# ---------- SEO analyzer import ----------
def _fallback_pagespeed(url: str) -> dict:
    # Safe placeholder if your seo.py isn't wired yet
    return {"ok": False, "message": "seo.get_pagespeed_data not found", "url": url}

get_pagespeed_data = _fallback_pagespeed
try:
    from .seo import get_pagespeed_data as _real_get_pagespeed_data  # type: ignore
    get_pagespeed_data = _real_get_pagespeed_data
except Exception as e:
    logger.warning("seo.get_pagespeed_data unavailable, using fallback: %r", e)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render the main dashboard with empty result
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Accepts HTML form posts (action="/analyze", method="post")
@app.post("/analyze", response_class=HTMLResponse)
@app.post("/analyze/", response_class=HTMLResponse)  # tolerate trailing slash
async def analyze_form(
    request: Request,
    url: Optional[str] = Form(None),
    target_url: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
):
    target = url or target_url or website
    if not target:
        raise HTTPException(status_code=400, detail="Missing URL (expected 'url', 'target_url', or 'website')")

    # Run analyzer in a thread so we don't block the event loop
    data = await run_in_threadpool(get_pagespeed_data, target)

    # Render the SAME page (index.html) but with `result` populated
    return templates.TemplateResponse("index.html", {"request": request, "result": data})

# JSON API variant: POST /api/analyze with {"url":"https://..."}
class AnalyzeRequest(BaseModel):
    url: str

@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_api(payload: AnalyzeRequest):
    data = await run_in_threadpool(get_pagespeed_data, payload.url)
    return {"url": payload.url, "result": data}

# Small helper page: opens canonical + AMP in new tabs via JS
# The template links to request.url_for('amp_compare')
@app.get("/amp-compare", name="amp_compare", response_class=HTMLResponse)
async def amp_compare(request: Request, url: str):
    # We try to get the AMP url from a quick analyze; if it's not available, we just open the canonical.
    try:
        data = await run_in_threadpool(get_pagespeed_data, url)
        amp_url = (data or {}).get("amp_url")
    except Exception:
        amp_url = None

    # Minimal HTML that pops tabs for quick visual compare
    body = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>AMP Compare</title>
      </head>
      <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 16px;">
        <h1 style="margin: 0 0 12px;">AMP vs Canonical</h1>
        <p>Opening pages in new tabs… If nothing opened, use the links below.</p>
        <ul>
          <li><a href="{url}" target="_blank" rel="noopener">Canonical</a></li>
          {"<li><a href='" + amp_url + "' target='_blank' rel='noopener'>AMP</a></li>" if amp_url else ""}
        </ul>
        <script>
          try {{
            window.open("{url}", "_blank");
            {"window.open('" + amp_url + "', '_blank');" if amp_url else ""}
          }} catch (e) {{}}
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=body)

@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    return {"status": "ok"}
