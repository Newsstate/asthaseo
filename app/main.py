# app/main.py
from pathlib import Path
import os
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Paths ---
FILE_DIR = Path(__file__).parent          # .../app
PROJECT_ROOT = FILE_DIR.parent            # project root (where /static lives)

# --- App FIRST (so mounts/routers can use it) ---
app = FastAPI(title="SEO Insight", version="1.0.0")

# --- Static files ---
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # ensure exists to avoid startup errors
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Templates ---
templates = Jinja2Templates(directory=str(FILE_DIR / "templates"))

# --- Optional: DB init from env (safe if db.py not present) ---
logger = logging.getLogger("uvicorn.error")
try:
    from .db import init_db  # expects a function init_db(db_url: str)
except Exception:
    init_db = None

@app.on_event("startup")
async def on_startup():
    # Initialize database (SQLite path persists on Render if you mounted /var/data)
    db_url = os.getenv("DATABASE_URL", "sqlite:////var/data/seo_insight.db")
    if init_db:
        try:
            init_db(db_url)
            logger.info("Database initialized (%s)", db_url.split("://", 1)[0])
        except Exception:
            logger.exception("Database initialization failed")
            raise

    # Warn if PageSpeed key missing (won't crash the app)
    if not os.getenv("PAGESPEED_API_KEY"):
        logger.warning("PAGESPEED_API_KEY not set; PageSpeed features may not work")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Ensure you have app/templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    return {"status": "ok"}

# --- Optional: include extra routers if they exist ---
try:
    # from .routers import pages, api
    # app.include_router(pages.router)
    # app.include_router(api.router)
    pass
except Exception:
    logger.warning("Optional routers not loaded", exc_info=True)
