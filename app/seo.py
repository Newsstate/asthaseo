# app/seo.py
# --------------------------------------------------------------------------------------
# DEV-ONLY WARNING:
# SSL verification is intentionally disabled (verify=False) for local testing.
# Do NOT use verify=False in production or on untrusted networks.
# --------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import asyncio
import collections
import json
import os
import re
import time
import urllib.robotparser as robotparser

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

UserAgent = "Mozilla/5.0 (compatible; SEO-Analyzer-Py/1.1; +https://example.local)"
STOPWORDS = {
    # minimal english stopwords for keyword density
    "the","and","for","are","but","not","you","your","with","have","this","that","was",
    "from","they","his","her","she","him","has","had","were","will","what","when","where",
    "who","why","how","can","all","any","each","few","more","most","other","some","such",
    "no","nor","too","very","of","to","in","on","by","is","as","at","it","or","be","we",
    "an","a","our","us","if","out","up","so","do","did","does","their","its","than","then"
}

# If you want to hardcode, set PSI_API_KEY = "YOUR_KEY". Otherwise use env var.
PSI_API_KEY = os.getenv("PAGESPEED_API_KEY", "").strip() or None

# ======================================================================================
# HTTP helpers
# ======================================================================================

def _client() -> httpx.AsyncClient:
    """
    Create an AsyncClient for network calls.
    - verify=False for local dev
    - trust_env=True so corporate proxy env vars are respected if present
    """
    return httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": UserAgent},
        verify=False,       # DEV ONLY; DO NOT USE IN PROD
        trust_env=True,
    )

async def fetch(url: str, timeout: float = 25.0) -> Tuple[int, bytes, Dict[str, str], Dict[str, Any]]:
    """
    Fetch raw HTML. Returns: (load_ms, body, headers+status, netinfo)
    """
    async with _client() as client:
        start = time.perf_counter()
        resp = await client.get(
            url,
            timeout=timeout,
            headers={"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"},
        )
        load_ms = int((time.perf_counter() - start) * 1000)
        body = resp.content
        headers = {k.lower(): v for k, v in resp.headers.items()}
        netinfo = {
            "final_url": str(resp.url),
            "http_version": getattr(resp, "http_version", None),
            "redirects": len(resp.history),
        }
        return load_ms, body, headers | {"status": str(resp.status_code)}, netinfo

def _text(el) -> str:
    return (el.get_text(" ", strip=True) if el else "").strip()

def _norm_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u or u in seen:
            continue
        out.append(u)
        seen.add(u)
    return out


# --- Playwright Render Helper --------------------------------------------------
from contextlib import asynccontextmanager

RENDER_TIMEOUT_MS = int(os.getenv("RENDER_TIMEOUT_MS", "20000"))  # 20s default
RENDER_WAIT_STATE = os.getenv("RENDER_WAIT_STATE", "networkidle") # or "load"
RENDER_JS_ENABLED = os.getenv("RENDER_JS_ENABLED", "1") == "1"
RENDER_DEBUG = os.getenv("RENDER_DEBUG", "0") == "1"  # set 1 to log errors

PLAYWRIGHT_ARGS = [
    "--disable-gpu",
    "--no-sandbox",  # harmless on Windows; useful in CI/containers
    "--disable-dev-shm-usage",
    "--disable-setuid-sandbox",
    "--disable-features=SitePerProcess",
    "--disable-blink-features=AutomationControlled",
]

PLAYWRIGHT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

# prefer system browsers to avoid big downloads
PREFERRED_CHANNEL = os.getenv("PW_BROWSER_CHANNEL", "").strip().lower()  # "msedge" or "chrome"
CHROME_PATH = os.getenv("CHROME_PATH", "").strip()  # e.g., r"C:\Program Files\Google\Chrome\Application\chrome.exe"

async def _ensure_playwright():
    try:
        from playwright.async_api import async_playwright  # noqa
        return async_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright not installed. Run: 'pip install playwright'. "
            "For bundled Chromium you can also run: 'python -m playwright install chromium' (optional when using channels)."
        ) from e

@asynccontextmanager
async def _browser_context():
    async_playwright = await _ensure_playwright()
    async with async_playwright().start() as pw:
        browser = None
        last_err: Optional[Exception] = None

        async def _try_channel(ch: str):
            try:
                return await pw.chromium.launch(headless=True, channel=ch, args=PLAYWRIGHT_ARGS)
            except Exception as e:
                return e

        # 1) explicit channel via env
        if PREFERRED_CHANNEL in ("msedge", "chrome"):
            maybe = await _try_channel(PREFERRED_CHANNEL)
            if not isinstance(maybe, Exception):
                browser = maybe

        # 2) Edge (present on Windows)
        if browser is None:
            maybe = await _try_channel("msedge")
            if not isinstance(maybe, Exception):
                browser = maybe
            else:
                last_err = maybe

        # 3) Chrome (if installed)
        if browser is None:
            maybe = await _try_channel("chrome")
            if not isinstance(maybe, Exception):
                browser = maybe
            else:
                last_err = maybe

        # 4) Custom executable path (env)
        if browser is None and CHROME_PATH:
            try:
                browser = await pw.chromium.launch(
                    headless=True, executable_path=CHROME_PATH, args=PLAYWRIGHT_ARGS
                )
            except Exception as e:
                last_err = e

        # 5) Fallback: bundled Chromium (only works if previously installed)
        if browser is None:
            try:
                browser = await pw.chromium.launch(headless=True, args=PLAYWRIGHT_ARGS)
            except Exception as e:
                raise RuntimeError(
                    "No system Chrome/Edge found and bundled Chromium is unavailable. "
                    "Set PW_BROWSER_CHANNEL=msedge (or chrome) or CHROME_PATH to your chrome.exe. "
                    f"Last error: {last_err or e}"
                ) from e

        context = await browser.new_context(
            user_agent=PLAYWRIGHT_UA,
            java_script_enabled=RENDER_JS_ENABLED,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
        )
        try:
            yield context
        finally:
            await context.close()
            await browser.close()

async def render_html_with_playwright(url: str) -> dict:
    """
    Returns: {"html": str, "final_url": str, "timed_out": bool}
    Never raises on common timeouts; caller can decide how to surface it.
    """
    try:
        async with _browser_context() as ctx:
            page = await ctx.new_page()
            # Extra safety: don’t wait forever on flaky resources
            await page.route("**/*", lambda route: route.continue_())
            resp = await page.goto(
                url,
                timeout=RENDER_TIMEOUT_MS,
                wait_until=RENDER_WAIT_STATE,  # "networkidle" waits for quiet network
            )
            # Give late JS a brief chance to settle (tweak via env if needed)
            extra_wait_ms = int(os.getenv("RENDER_EXTRA_WAIT_MS", "600"))
            if extra_wait_ms > 0:
                await page.wait_for_timeout(extra_wait_ms)

            html = await page.content()
            final_url = page.url
            return {
                "html": html or "",
                "final_url": final_url or url,
                "status": resp.status if resp else None,
                "timed_out": False,
            }
    except Exception:
        # Don’t explode the whole request; upstream will show a soft error
        return {"html": "", "final_url": url, "status": None, "timed_out": True}

# ======================================================================================
# Structured data (JSON-LD, Microdata, RDFa)
# ======================================================================================

def _safe_json_loads(s: str) -> Any:
    """
    Try to parse JSON-LD safely. Handle common breakages lightly.
    """
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # Try soft fixes for trailing commas (very light)
        s2 = re.sub(r",(\s*[}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return None

def extract_structured_data_full(html_bytes: bytes, base_url: str) -> Dict[str, Any]:
    """
    Scan entire document (head + body) for:
    - JSON-LD  <script type="application/ld+json">
    - Microdata (itemscope / itemtype / itemprop)
    - RDFa (typeof / about / property)
    """
    html = html_bytes.decode("utf-8", "ignore")
    soup = BeautifulSoup(html, "lxml")

    # JSON-LD (anywhere)
    json_ld: List[Any] = []
    for tag in soup.find_all("script", attrs={"type": lambda v: v and "ld+json" in v.lower()}):
        raw = tag.string or tag.text or ""
        data = _safe_json_loads(raw)
        if data is None:
            continue
        if isinstance(data, list):
            json_ld.extend(data)
        else:
            json_ld.append(data)

    # Microdata (light extraction)
    microdata: List[Dict[str, Any]] = []
    for el in soup.find_all(attrs={"itemscope": True}):
        itemtype = el.get("itemtype")
        props = []
        for prop in el.find_all(attrs={"itemprop": True}):
            props.append({"prop": prop.get("itemprop"), "value": prop.get("content") or _text(prop)})
        microdata.append({"itemtype": itemtype, "properties": props})

    # RDFa (light extraction)
    rdfa: List[Dict[str, Any]] = []
    for el in soup.find_all(attrs={"typeof": True}):
        rdfa.append({
            "typeof": el.get("typeof") or "",
            "about": el.get("about") or el.get("resource") or "",
            "props": [p.get("property") for p in el.find_all(attrs={"property": True})],
        })

    return {"json_ld": json_ld, "microdata": microdata, "rdfa": rdfa}

def _jsonld_items(jsonld_any: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize JSON-LD into a flat list of dict items, handling @graph.
    """
    items: List[Dict[str, Any]] = []

    def push_node(node: Any):
        if isinstance(node, dict):
            items.append(node)
        elif isinstance(node, list):
            for y in node:
                push_node(y)

    for block in (jsonld_any or []):
        if isinstance(block, dict) and "@graph" in block:
            push_node(block.get("@graph"))
        else:
            push_node(block)
    return items

def _sd_required_fields_for(typ: str) -> List[str]:
    t = typ.lower()
    if t in ("article", "newsarticle", "blogposting"):
        return ["headline"]
    if t in ("organization", "localbusiness"):
        return ["name"]
    if t in ("product",):
        return ["name"]
    if t in ("breadcrumblist",):
        return ["itemListElement"]
    if t in ("faqpage",):
        return ["mainEntity"]
    if t in ("event",):
        return ["name", "startDate"]
    return []

def validate_jsonld(jsonld_any: List[Any]) -> Dict[str, Any]:
    items = _jsonld_items(jsonld_any)
    report = []
    for it in items:
        typ = it.get("@type")
        typ_val = (typ[0] if isinstance(typ, list) and typ else (typ or "Unknown"))
        req = _sd_required_fields_for(str(typ_val))
        missing = [f for f in req if f not in it or (isinstance(it.get(f), str) and not it.get(f).strip())]
        report.append({"type": typ_val, "missing": missing, "ok": len(missing) == 0 if req else True})
    summary = {
        "total_items": len(items),
        "ok_count": sum(1 for r in report if r.get("ok")),
        "has_errors": any(r for r in report if not r.get("ok")),
    }
    return {"summary": summary, "items": report}

def _localname(t: str | None) -> str | None:
    if not t:
        return None
    if "#" in t:
        t = t.rsplit("#", 1)[-1]
    if "/" in t:
        t = t.rstrip("/").rsplit("/", 1)[-1]
    t = t.strip()
    return t or None

def structured_types_present(jsonld: List[Any], microdata: List[Any], rdfa: List[Any]) -> Dict[str, Any]:
    types: set[str] = set()

    # JSON-LD types
    for item in _jsonld_items(jsonld):
        t = item.get("@type")
        if isinstance(t, list):
            for x in t:
                ln = _localname(x)
                if ln:
                    types.add(ln)
        elif isinstance(t, str):
            ln = _localname(t)
            if ln:
                types.add(ln)

    # Microdata types
    for md in (microdata or []):
        it = md.get("itemtype")
        if isinstance(it, list):
            for x in it:
                ln = _localname(x)
                if ln:
                    types.add(ln)
        elif isinstance(it, str):
            ln = _localname(it)
            if ln:
                types.add(ln)

    # RDFa types
    for rd in (rdfa or []):
        tf = rd.get("typeof")
        if isinstance(tf, str):
            for tok in tf.split():
                ln = _localname(tok)
                if ln:
                    types.add(ln)

    has_news = any(t.lower() == "newsarticle" for t in types)
    return {"types": sorted(types), "has_newsarticle": has_news}

# ======================================================================================
# Robots meta / X-Robots-Tag helpers
# ======================================================================================

def _parse_robots_meta(val: Optional[str]) -> Dict[str, bool]:
    d = {"noindex": False, "nofollow": False}
    if not val:
        return d
    toks = [t.strip().lower() for t in re.split(r"[,\s]+", val) if t.strip()]
    d["noindex"] = "noindex" in toks
    d["nofollow"] = "nofollow" in toks
    return d

def _parse_x_robots(val: Optional[str]) -> Dict[str, bool]:
    d = {"noindex": False, "nofollow": False}
    if not val:
        return d
    val = val.lower()
    d["noindex"] = "noindex" in val
    d["nofollow"] = "nofollow" in val
    return d

# ======================================================================================
# Keyword density
# ======================================================================================

def _extract_text_for_density(soup: BeautifulSoup) -> str:
    # remove script/style/nav/footer to reduce noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return _text(soup.body or soup)

def keyword_density(text: str, top_n: int = 10) -> List[Dict[str, Any]]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    counter = collections.Counter(words)
    total = sum(counter.values()) or 1
    common = counter.most_common(top_n)
    return [{"word": w, "count": c, "percent": round(c * 100.0 / total, 2)} for w, c in common]

# ======================================================================================
# Phase 1: Static HTML parse & basic checks
# ======================================================================================

def parse_html(url: str, body: bytes, headers: Dict[str, str], load_ms: int) -> Dict[str, Any]:
    soup = BeautifulSoup(body, "lxml")
    head = soup.head or soup

    # --- Meta basics
    title = _text(head.title) if head and head.title else None
    desc = None
    robots = None
    for meta in head.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in ("description", "og:description"):
            desc = desc or meta.get("content")
        if name == "robots":
            robots = meta.get("content")

    # Canonical
    link_canon = head.find("link", rel=lambda v: v and "canonical" in v.lower())
    canon = urljoin(url, link_canon["href"]) if (link_canon and link_canon.get("href")) else None

    # AMP
    amp_link = head.find("link", rel=lambda v: v and "amphtml" in v.lower())
    amp_url = urljoin(url, amp_link["href"]) if (amp_link and amp_link.get("href")) else None
    is_amp = bool(amp_link) or ("amp-boilerplate" in str(body[:5000]).lower())

    # Headings
    h1 = [_text(h) for h in soup.find_all("h1")]
    h2 = [_text(h) for h in soup.find_all("h2")]
    h3 = [_text(h) for h in soup.find_all("h3")]
    h4 = [_text(h) for h in soup.find_all("h4")]
    h5 = [_text(h) for h in soup.find_all("h5")]
    h6 = [_text(h) for h in soup.find_all("h6")]

    # Links
    a_links = [a.get("href") for a in soup.find_all("a")]
    internal_links, external_links, nofollow_links = [], [], []
    parsed = urlparse(url)
    base_host = parsed.netloc.lower()
    for href in a_links:
        if not href:
            continue
        absu = urljoin(url, href)
        host = urlparse(absu).netloc.lower()
        (internal_links if host == base_host else external_links).append(absu)
    for a in soup.find_all("a"):
        if a.get("rel") and "nofollow" in " ".join(a.get("rel")).lower():
            nofollow_links.append(urljoin(url, a.get("href", "")))

    internal_links = _norm_urls(internal_links)[:300]
    external_links = _norm_urls(external_links)[:300]
    nofollow_links = _norm_urls(nofollow_links)[:300]

    # Images: missing alts
    imgs = soup.find_all("img")
    missing_alts = []
    with_alt = 0
    for im in imgs:
        alt = (im.get("alt") or "").strip()
        if alt:
            with_alt += 1
            continue
        src = urljoin(url, im.get("src") or "")
        missing_alts.append({"src": src})

    # Structured data
    sd_all = extract_structured_data_full(body, url)
    jsonld = sd_all.get("json_ld") or []
    microdata_any = sd_all.get("microdata") or []
    rdfa_any = sd_all.get("rdfa") or []
    sd_validation = validate_jsonld(jsonld)
    sd_types = structured_types_present(jsonld, microdata_any, rdfa_any)

    # hreflang
    hreflang_rows = []
    for ln in head.find_all("link", rel=lambda v: v and "alternate" in v.lower()):
        href = ln.get("href")
        hreflang = (ln.get("hreflang") or "").strip().lower()
        if href and hreflang:
            hreflang_rows.append({"hreflang": hreflang, "href": urljoin(url, href)})

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt" if parsed.scheme and parsed.netloc else None

    # --- Quick checks
    checks: Dict[str, Any] = {}

    # Title & meta description length
    title_len = len(title or "")
    desc_len = len((desc or "").strip() or "")
    checks["title_length"] = {"chars": title_len, "ok": 30 <= title_len <= 65}
    checks["meta_description_length"] = {"chars": desc_len, "ok": 70 <= desc_len <= 160}

    # Heading sanity
    checks["h1_count"] = {"count": len(h1), "ok": len(h1) == 1}

    # Viewport
    viewport = head.find("meta", attrs={"name": "viewport"})
    checks["viewport_meta"] = {"present": viewport is not None, "ok": viewport is not None, "value": "present" if viewport else "missing"}

    # Canonical sanity
    checks["canonical"] = {
        "present": canon is not None,
        "absolute": canon.startswith("http") if canon else False,
        "self_ref": canon == url if canon else False,
        "ok": bool(canon and canon.startswith("http")),
        "value": canon or "",
    }

    # IMG alt coverage
    total_imgs = len(imgs)
    checks["alt_coverage"] = {
        "with_alt": with_alt,
        "total_imgs": total_imgs,
        "percent": round((with_alt / total_imgs * 100), 1) if total_imgs else 100.0,
        "ok": (with_alt / total_imgs >= 0.8) if total_imgs else True,
    }

    # Lang & charset
    html_tag = soup.find("html")
    lang_val = html_tag.get("lang") if html_tag else None
    charset = None
    meta_charset = head.find("meta", attrs={"charset": True})
    if meta_charset:
        charset = meta_charset.get("charset")
    else:
        ct = head.find("meta", attrs={"http-equiv": "Content-Type"})
        if ct and ct.get("content"):
            for part in ct["content"].split(";"):
                part = part.strip().lower()
                if part.startswith("charset="):
                    charset = part.split("=", 1)[1]
    checks["lang"] = {"value": lang_val, "present": bool(lang_val), "ok": bool(lang_val)}
    checks["charset"] = {"value": charset, "present": bool(charset), "ok": bool(charset)}

    # Compression (from response headers)
    encoding = headers.get("content-encoding", "").lower()
    compression_value = "gzip" if "gzip" in encoding else ("br" if "br" in encoding else "none")
    checks["compression"] = {"gzip": "gzip" in encoding, "brotli": "br" in encoding, "value": compression_value, "ok": compression_value in ("gzip", "br")}

    # Social cards completeness
    og_required = ["og:title", "og:description", "og:image"]
    og_present = {p: head.find("meta", property=p) is not None for p in og_required}
    tw_required = ["twitter:card", "twitter:title", "twitter:description", "twitter:image"]
    tw_present = {n: head.find("meta", attrs={"name": n}) is not None for n in tw_required}
    checks["social_cards"] = {
        "og_complete": all(og_present.values()),
        "twitter_complete": all(tw_present.values()),
        "ok": all(og_present.values()) and all(tw_present.values()),
        "value": f"OG:{'ok' if all(og_present.values()) else 'miss'} / TW:{'ok' if all(tw_present.values()) else 'miss'}",
    }

    # Robots meta / X-Robots-Tag and indexability
    xr_raw = headers.get("x-robots-tag")
    meta_flags = _parse_robots_meta(robots)
    header_flags = _parse_x_robots(xr_raw)
    noindex_flag = meta_flags["noindex"] or header_flags["noindex"]
    nofollow_flag = meta_flags["nofollow"] or header_flags["nofollow"]
    checks["robots_meta_index"] = {"value": "noindex" if meta_flags["noindex"] else "index", "ok": not meta_flags["noindex"]}
    checks["robots_meta_follow"] = {"value": "nofollow" if meta_flags["nofollow"] else "follow", "ok": not meta_flags["nofollow"]}
    checks["x_robots_tag"] = {"value": xr_raw or "", "ok": not (header_flags["noindex"] or header_flags["nofollow"])}
    checks["indexable"] = {
        "value": ("index,follow" if not (noindex_flag or nofollow_flag)
                  else "noindex,nofollow" if (noindex_flag and nofollow_flag)
                  else ("noindex" if noindex_flag else "nofollow")),
        "ok": not noindex_flag
    }

    # Keyword density (basic)
    density = keyword_density(_extract_text_for_density(soup), top_n=10)

    # Open Graph & Twitter tags (raw snapshot)
    og_tags = {}
    for m in head.find_all("meta", attrs={"property": True}):
        prop = m.get("property")
        if prop and prop.startswith("og:"):
            og_tags[prop] = m.get("content")
    tw_tags = {}
    for m in head.find_all("meta", attrs={"name": True}):
        nm = m.get("name")
        if nm and nm.startswith("twitter:"):
            tw_tags[nm] = m.get("content")

    return {
        "url": url,
        "status_code": int(headers.get("status", "0")),
        "load_time_ms": load_ms,
        "content_length": int(headers.get("content-length") or len(body)),
        "title": title,
        "description": desc,
        "canonical": canon,
        "robots_meta": robots,
        "is_amp": is_amp,
        "amp_url": amp_url,
        "headings": {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "h6": h6},
        "internal_links": _norm_urls(internal_links)[:300],
        "external_links": _norm_urls(external_links)[:300],
        "nofollow_links": _norm_urls(nofollow_links)[:300],
        # Images
        "images_missing_alt": missing_alts[:200],
        # Structured data
        "json_ld": jsonld,
        "json_ld_validation": sd_validation,
        "microdata": microdata_any,
        "microdata_summary": {"count": len(microdata_any)},
        "rdfa": rdfa_any,
        "rdfa_summary": {"count": len(rdfa_any)},
        "sd_types": sd_types,  # includes has_newsarticle
        # Social
        "has_open_graph": any(og_present.values()),
        "has_twitter_card": any(tw_present.values()),
        "open_graph": og_tags,
        "twitter_card": tw_tags,
        # Robots/sitemaps
        "robots_url": robots_url,
        "hreflang": hreflang_rows,
        # Checks
        "checks": checks,
        # Keyword density
        "keyword_density_top": density,
    }

# ======================================================================================
# Phase 2: Link sampling + robots/sitemap reachability
# ======================================================================================

async def _check_one(session: httpx.AsyncClient, u: str) -> Dict[str, Any]:
    item: Dict[str, Any] = {"url": u}
    try:
        r = await session.head(u, timeout=7.0, follow_redirects=True)
        if r.status_code in (405, 501):
            r = await session.get(u, timeout=10.0, follow_redirects=True)
        item["status"] = r.status_code
        item["final_url"] = str(r.url)
        item["redirects"] = len(r.history)
    except Exception as e:
        item["error"] = str(e)
    return item

async def link_audit(data: Dict[str, Any]) -> Dict[str, Any]:
    internal = (data.get("internal_links") or [])[:30]
    external = (data.get("external_links") or [])[:15]
    out: Dict[str, Any] = {"internal": [], "external": []}
    async with _client() as s:
        res_int = await asyncio.gather(*(_check_one(s, u) for u in internal))
        res_ext = await asyncio.gather(*(_check_one(s, u) for u in external))
    out["internal"] = res_int
    out["external"] = res_ext
    return out

def _discover_sitemaps_from_robots(robots_txt: str) -> List[str]:
    sitemaps: List[str] = []
    for line in robots_txt.splitlines():
        if line.strip().lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                sitemaps.append(sm)
    # de-dup & keep order
    seen = set()
    uniq: List[str] = []
    for sm in sitemaps:
        if sm not in seen:
            uniq.append(sm)
            seen.add(sm)
    return uniq

async def robots_sitemap_audit(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Fetch robots.txt (if present)
    - Determine if the scanned URL is blocked for our UA
    - Discover sitemaps **only** from robots.txt (dynamic)
    """
    res: Dict[str, Any] = {"robots_txt": None, "sitemaps": [], "blocked_by_robots": None}
    robots_url = data.get("robots_url")
    target_url = data.get("url") or ""
    async with _client() as s:
        robots_txt = ""
        if robots_url:
            try:
                r = await s.get(robots_url, timeout=8.0)
                robots_txt = r.text if r.status_code == 200 else ""
                res["robots_txt"] = {"url": robots_url, "status": r.status_code, "length": len(r.content)}
            except Exception as e:
                res["robots_txt"] = {"url": robots_url, "error": str(e)}

        # blocked?
        if robots_txt:
            rp = robotparser.RobotFileParser()
            rp.parse(robots_txt.splitlines())
            try:
                res["blocked_by_robots"] = (not rp.can_fetch(UserAgent, target_url))
            except Exception:
                res["blocked_by_robots"] = None

        # sitemaps from robots only (skip scanning default /sitemap_index.xml)
        if robots_txt:
            smaps = _discover_sitemaps_from_robots(robots_txt)
            for sm in smaps:
                if sm.lower().endswith("sitemap_index.xml"):
                    res["sitemaps"].append({"url": sm, "note": "listed in robots; skipped index fetch"})
                    continue
                try:
                    r = await s.head(sm, timeout=8.0, follow_redirects=True)
                    res["sitemaps"].append({"url": sm, "status": r.status_code})
                except Exception as e:
                    res["sitemaps"].append({"url": sm, "error": str(e)})
    return res

# ---- Optional: summarize sitemaps (kept but **skips** sitemap_index.xml fetch) --------

async def summarize_sitemaps(sitemap_urls: List[str]) -> Dict[str, Any]:
    """
    Fetch up to 2 sitemaps (skip *sitemap_index.xml*), count <url> tags (sample).
    """
    out = {"checked": [], "total_url_count_sampled": 0, "is_index": False}
    async with _client() as s:
        checked = 0
        for sm in (sitemap_urls or []):
            if checked >= 2:
                break
            if sm.lower().endswith("sitemap_index.xml"):
                out["checked"].append({"url": sm, "skipped": True, "reason": "index sitemap"})
                continue
            try:
                r = await s.get(sm, timeout=8.0)
                entry = {"url": sm, "status": r.status_code}
                if r.status_code == 200 and "xml" in (r.headers.get("content-type", "").lower()):
                    txt = r.text
                    is_index = "<sitemapindex" in txt[:1000].lower()
                    entry["is_index"] = is_index
                    out["is_index"] = out["is_index"] or is_index
                    url_count = txt.lower().count("<url>")
                    entry["url_tags"] = url_count
                    out["total_url_count_sampled"] += url_count
                out["checked"].append(entry)
            except Exception as e:
                out["checked"].append({"url": sm, "error": str(e)})
            checked += 1
    return out

# ======================================================================================
# Phase 3: Rendered DOM (Windows-safe: Playwright sync API in a thread)
# ======================================================================================

def _render_sync(url: str, user_agent: str, timeout_ms: int) -> dict:
    """
    Returns dict: {"html": Optional[str], "error": Optional[str], "used": str}
    - tries Edge/Chrome channels first, then CHROME_PATH, then bundled Chromium
    - tries wait_until='networkidle' first, then falls back to 'load'
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return {"html": None, "error": f"playwright import failed: {e}", "used": "none"}

    used = []
    try:
        with sync_playwright() as p:
            browser = None

            def try_channel(ch: str):
                nonlocal used
                try:
                    b = p.chromium.launch(headless=True, channel=ch, args=PLAYWRIGHT_ARGS)
                    used.append(f"channel:{ch}")
                    return b
                except Exception as ex:
                    used.append(f"channel:{ch}:fail:{type(ex).__name__}")
                    return None

            # 1) explicit channel
            if PREFERRED_CHANNEL in ("msedge", "chrome"):
                browser = try_channel(PREFERRED_CHANNEL)

            # 2) Edge → 3) Chrome
            if browser is None:
                browser = try_channel("msedge") or try_channel("chrome")

            # 4) executable path
            if browser is None and CHROME_PATH:
                try:
                    browser = p.chromium.launch(
                        headless=True, executable_path=CHROME_PATH, args=PLAYWRIGHT_ARGS
                    )
                    used.append(f"exe:{CHROME_PATH}")
                except Exception as ex:
                    used.append(f"exe:fail:{type(ex).__name__}")
                    browser = None

            # 5) bundled Chromium (only if previously installed)
            if browser is None:
                try:
                    browser = p.chromium.launch(headless=True, args=PLAYWRIGHT_ARGS)
                    used.append("bundled")
                except Exception as ex:
                    used.append(f"bundled:fail:{type(ex).__name__}")
                    return {"html": None, "error": "no browser available (edge/chrome/bundled failed)", "used": " | ".join(used)}

            try:
                context = browser.new_context(user_agent=user_agent, viewport={"width": 1366, "height": 768})
                page = context.new_page()

                # Try networkidle first
                try:
                    page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                except Exception as e1:
                    # Fallback: some sites never go “idle”; try 'load'
                    try:
                        page.goto(url, wait_until="load", timeout=timeout_ms)
                    except Exception as e2:
                        # Return the more informative of the two errors
                        err = f"goto failed (networkidle: {type(e1).__name__}; load: {type(e2).__name__})"
                        return {"html": None, "error": err, "used": " | ".join(used)}

                html = page.content()
                return {"html": html, "error": None, "used": " | ".join(used)}
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
    except Exception as e:
        return {"html": None, "error": f"unexpected render error: {type(e).__name__}: {e}", "used": " | ".join(used)}

async def fetch_rendered(url: str, timeout_ms: int = 30000) -> dict:
    """
    Async wrapper. Returns {"html", "error", "used"}.
    """
    return await asyncio.to_thread(_render_sync, url, UserAgent, timeout_ms)

def _summarize_for_compare(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    head = soup.head or soup

    # meta
    title = _text(head.title) if head and head.title else None
    desc = None
    robots = None
    for meta in head.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in ("description", "og:description"):
            desc = desc or meta.get("content")
        if name == "robots":
            robots = meta.get("content")

    # canonical
    canon = None
    link_canon = head.find("link", rel=lambda v: v and "canonical" in v.lower())
    if link_canon and link_canon.get("href"):
        canon = urljoin(url, link_canon["href"])

    # headings
    h1 = [_text(h) for h in soup.find_all("h1")]

    # social quick flags
    og_ok = head.find("meta", property="og:title") is not None
    tw_ok = head.find("meta", attrs={"name": "twitter:card"}) is not None

    # json-ld / microdata / rdfa counts from rendered HTML
    sd = extract_structured_data_full(html.encode("utf-8", "ignore"), url)
    jsonld_count = len(sd.get("json_ld") or [])
    microdata_count = len(sd.get("microdata") or [])
    rdfa_count = len(sd.get("rdfa") or [])

    # link counts
    a_links = [a.get("href") for a in soup.find_all("a")]
    internal, external = 0, 0
    base_host = urlparse(url).netloc.lower()
    for href in a_links:
        if not href:
            continue
        absu = urljoin(url, href)
        host = urlparse(absu).netloc.lower()
        if host == base_host:
            internal += 1
        else:
            external += 1

    # viewport
    viewport_present = head.find("meta", attrs={"name": "viewport"}) is not None

    return {
        "title": title,
        "description": desc,
        "canonical": canon,
        "robots_meta": robots,
        "h1_count": len(h1),
        "h1_first": h1[0] if h1 else None,
        "has_open_graph": og_ok,
        "has_twitter_card": tw_ok,
        "json_ld_count": jsonld_count,
        "microdata_count": microdata_count,
        "rdfa_count": rdfa_count,
        "internal_links_count": internal,
        "external_links_count": external,
        "viewport_present": viewport_present,
    }

def rendered_compare_matrix(original: Dict[str, Any], rendered_html: Optional[str]) -> Dict[str, Any]:
    if not rendered_html:
        return {"rendered": False}

    before = {
        "title": original.get("title"),
        "description": original.get("description"),
        "canonical": original.get("canonical"),
        "robots_meta": original.get("robots_meta"),
        "h1_count": len((original.get("headings") or {}).get("h1", []) or []),
        "h1_first": ((original.get("headings") or {}).get("h1") or [None])[0],
        "has_open_graph": bool(original.get("has_open_graph")),
        "has_twitter_card": bool(original.get("has_twitter_card")),
        "json_ld_count": len(original.get("json_ld") or []),
        "microdata_count": len(original.get("microdata") or []),
        "rdfa_count": len(original.get("rdfa") or []),
        "internal_links_count": len(original.get("internal_links") or []),
        "external_links_count": len(original.get("external_links") or []),
        "viewport_present": bool(original.get("checks", {}).get("viewport_meta", {}).get("present", False)),
    }
    after = _summarize_for_compare(original.get("url", ""), rendered_html)

    def row(label: str, key: str) -> Dict[str, Any]:
        b = before.get(key)
        a = after.get(key)
        return {"label": label, "key": key, "before": b, "after": a, "changed": (b != a)}

    matrix = [
        row("Title", "title"),
        row("Meta Description", "description"),
        row("Canonical", "canonical"),
        row("Robots Meta", "robots_meta"),
        row("H1 Count", "h1_count"),
        row("First H1", "h1_first"),
        row("Open Graph Present", "has_open_graph"),
        row("Twitter Card Present", "has_twitter_card"),
        row("JSON-LD Count", "json_ld_count"),
        row("Microdata Count", "microdata_count"),
        row("RDFa Count", "rdfa_count"),
        row("Internal Links (count)", "internal_links_count"),
        row("External Links (count)", "external_links_count"),
        row("Viewport Meta Present", "viewport_present"),
    ]

    quick = {
        "rendered": True,
        "title_changed": before["title"] != after["title"],
        "description_changed": before["description"] != after["description"],
        "h1_count_changed": before["h1_count"] != after["h1_count"],
        "render_excerpt": rendered_html[:2000],
    }
    return {**quick, "matrix": matrix, "before": before, "after": after}

# ======================================================================================
# PageSpeed Insights (optional)
# ======================================================================================

async def fetch_pagespeed(url: str, api_key: Optional[str]) -> Dict[str, Any]:
    if not api_key:
        return {"enabled": False, "error": "No API key"}
    base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    async with _client() as s:
        out: Dict[str, Any] = {"enabled": True}
        for strat in ("mobile", "desktop"):
            try:
                r = await s.get(base, params={"url": url, "strategy": strat, "key": api_key}, timeout=35.0)
                if r.status_code != 200:
                    out[strat] = {"error": f"HTTP {r.status_code}"}
                    continue
                data = r.json()
                cat = (data.get("lighthouseResult", {}).get("categories", {}).get("performance") or {})
                score = cat.get("score", None)
                audits = data.get("lighthouseResult", {}).get("audits", {})
                def pick_value(audit_key: str, field: str = "displayValue"):
                    a = audits.get(audit_key) or {}
                    return a.get(field)
                out[strat] = {
                    "score": int(score * 100) if isinstance(score, (int, float)) else None,
                    "metrics": {
                        "first-contentful-paint": pick_value("first-contentful-paint"),
                        "speed-index": pick_value("speed-index"),
                        "largest-contentful-paint": pick_value("largest-contentful-paint"),
                        "total-blocking-time": pick_value("total-blocking-time"),
                        "cumulative-layout-shift": pick_value("cumulative-layout-shift"),
                        "server-response-time": pick_value("server-response-time"),
                    }
                }
            except Exception as e:
                out[strat] = {"error": str(e)}
        return out

# ======================================================================================
# Public entry
# ======================================================================================

async def analyze(url: str, *, do_rendered_check: bool = False) -> Dict[str, Any]:
    """
    Main analyzer used by the app.
    - Phase 1: static parse & checks
    - Phase 2: link sampling, robots/sitemap reachability (+ block check)
    - Phase 3 (optional): rendered DOM comparison via Playwright (sync API in a thread)
    - Performance block + (optional) PageSpeed Insights
    """
    # Phase 1
    load_ms, body, headers, netinfo = await fetch(url)
    data = parse_html(url, body, headers, load_ms)

    # Performance snapshot
    final_url = netinfo.get("final_url") or url
    is_https = urlparse(final_url).scheme.lower() == "https"
    data["performance"] = {
        "load_time_ms": load_ms,
        "page_size_bytes": int(headers.get("content-length") or len(body)),
        "http_version": netinfo.get("http_version"),
        "redirects": netinfo.get("redirects"),
        "final_url": final_url,
        "https": {
            "is_https": is_https,
            "ssl_checked": False,  # can't verify due to verify=False
            "ssl_ok": None,
        },
    }

    # Phase 2: link & robots/sitemap checks
    data["link_checks"] = await link_audit(data)
    data["crawl_checks"] = await robots_sitemap_audit(data)

    # Optionally summarize sitemaps discovered from robots
    sm_urls = [s.get("url") for s in (data.get("crawl_checks", {}).get("sitemaps") or []) if s.get("url")]
    if sm_urls:
        data["sitemap_summary"] = await summarize_sitemaps(sm_urls)
    else:
        data["sitemap_summary"] = {"checked": [], "total_url_count_sampled": 0, "is_index": False}

    # Phase 3: Rendered compare
    if do_rendered_check:
        rres = await fetch_rendered(final_url)
        html2 = rres.get("html")
        used = rres.get("used")
        err = rres.get("error")
        if html2:
            data["rendered_diff"] = rendered_compare_matrix(data, html2)
            data["rendered_diff"]["engine"] = used
        else:
            msg = "Playwright render skipped/failed"
            if err:
                msg = f"{msg}: {err} | engine={used}"
            data["rendered_diff"] = {"rendered": False, "error": msg}
            if RENDER_DEBUG:
                # log to console to help diagnose
                print("[RENDER_DEBUG]", msg)

    # PageSpeed Insights (optional)
    if PSI_API_KEY:
        data["pagespeed"] = await fetch_pagespeed(final_url, PSI_API_KEY)
        # Also surface scores at top-level performance block (handy for UI)
        try:
            data["performance"]["mobile_score"] = data["pagespeed"].get("mobile", {}).get("score")
            data["performance"]["desktop_score"] = data["pagespeed"].get("desktop", {}).get("score")
        except Exception:
            pass
    else:
        data["pagespeed"] = {"enabled": False}

    return data
