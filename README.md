# SEO Insight (Python)

A local FastAPI app that recreates the core features of your original TypeScript project:

- Analyze a URL for SEO basics (title, description, canonical, H1/H2, links, robots, sitemap, structured data, basic performance)
- Detect changes between previous and current analyses
- Schedule recurring scans (daily/weekly/custom cron-ish) using APScheduler
- Export CSV of past analyses
- Minimal web UI (Jinja templates) + JSON API

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000

## Configuration

Environment variables (optional):

- `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `FROM_EMAIL` — to enable email notifications
- `TZ` — timezone name for scheduler (default: system)

## Notes

- For fully-rendered pages (JS-heavy), add Playwright or Selenium later. This version uses plain HTTP fetching.
- Structured data detection is JSON-LD focused; microdata/RDFa can be added later.