# app/seo.py
from __future__ import annotations

import os
import re
import time
import json
import html
import string
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# ---------- Tunables via env ----------
# Overall "fast mode"
FAST_MODE_DEFAULT = os.getenv("FAST_MODE_DEFAULT", "1") == "1"

# Page fetch timeouts (seconds)
HTTP_TIMEOUT_MAIN = float(os.getenv("HTTP_TIMEOUT_MAIN", "10"))   # main page GET
HTTP_TIMEOUT_HEAD = float(os.getenv("HTTP_TIMEOUT_HEAD", "5"))    # link/asset HEADs
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "5"))

# Concurrency & sampling
HEAD_MAX_WORKERS = int(os.getenv("HEAD_MAX_WORKERS", "16"))
HEAD_SAMPLE_INTERNAL = int(os.getenv("HEAD_SAMPLE_INTERNAL", "4"))
HEAD_SAMPLE_EXTERNAL = int(os.getenv("HEAD_SAMPLE_EXTERNAL", "4"))
ASSET_SAMPLE_PER_TYPE = int(os.getenv("ASSET_SAMPLE_PER_TYPE", "10"))

# PSI (PageSpeed) control
ENABLE_PSI = os.getenv("ENABLE_PSI", "1") == "1"
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY")
# Include PSI inside fast mode and keep it quick
PSI_IN_FAST = os.getenv("PSI_IN_FAST", "1") == "1"             # run PSI even in fast mode
PSI_STRATEGY_FAST = os.getenv("PSI_STRATEGY_FAST", "mobile")   # mobile|desktop|both
PSI_TIMEOUT = int(os.getenv("PSI_TIMEOUT", "20"))              # per strategy seconds

# Parser: try lxml (faster) if installed
USE_LXML = os.getenv("USE_LXML", "1") == "1"

# ---------- Constants ----------
USER_AGENT = (
    "SEO-Inspector/1.1 (+https://example.com) "
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

STOPWORDS = set("""
a an the and or but if then else for to of in on at by with from as this that those these is are be was were been being
you your we our they their he she it its not no yes do does did done have has had having i me my mine ourselves himself
herself itself themselves them can will would should could may might must about above across after again against all almost
also am among amongst around because before behind below beneath beside besides between beyond both during each either few
more most much many nor off over under up down out into than too very via per just so only such own same very
""".split())

_LANG_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z]{4})?(-[a-zA-Z]{2}|\d{3})?$")  # loose BCP-47-ish

# ---------- HTTP Session (pooled) ----------
_SESSION: requests.Session | None = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        retry = Retry(
            total=1,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
        )
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate, br",
        })
        _SESSION = s
    return _SESSION

# ---------- Helpers ----------
def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    parsed = urlparse(u)
    if not parsed.scheme:
        u = "https://" + u
    return u

def _soup_parse(body: bytes, _base_url: str) -> BeautifulSoup:
    if USE_LXML:
        try:
            return BeautifulSoup(body, "lxml")
        except Exception:
            pass
    return BeautifulSoup(body, "html.parser")

def _fetch(url: str) -> Tuple[requests.Response, float]:
    s = _get_session()
    start = time.perf_counter()
    resp = s.get(
        url,
        allow_redirects=True,
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_MAIN),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return resp, elapsed_ms

def _get_text_density(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    text = html.unescape(text).lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    words = [w for w in text.split() if w and w not in STOPWORDS and len(w) > 2]
    if not words:
        return []
    total = len(words)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return [{"word": w, "count": c, "percent": round(c * 100.0 / total, 2)} for w, c in top]

def _collect_metas(soup: BeautifulSoup) -> Tuple[Dict[str, str], Dict[str, str]]:
    og: Dict[str, str] = {}
    tw: Dict[str, str] = {}
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not name or not content:
            continue
        if name.startswith("og:"):
            og[name] = content
        if name.startswith("twitter:"):
            tw[name] = content
    return og, tw

def _bool_badge(ok: bool | None) -> bool | None:
    return True if ok is True else (None if ok is None else False)

def _safe_len(s: str | None) -> int:
    return len(s or "")

def _check_indexability(robots_meta: str | None, x_robots: str | None) -> Tuple[bool | None, str]:
    meta = (robots_meta or "").lower()
    x = (x_robots or "").lower()
    tokens = set(re.split(r"[,\s]+", (meta + " " + x).strip()))
    if not tokens or tokens == {""}:
        return (None, "unknown")
    if "noindex" in tokens or "none" in tokens:
        return (False, "noindex")
    return (True, "index")

def _detect_amp(soup: BeautifulSoup, base_url: str) -> Tuple[bool, str | None]:
    html_tag = soup.find("html")
    is_amp = bool(html_tag and (html_tag.has_attr("amp") or html_tag.has_attr("⚡")))
    amp_link = soup.find("link", rel=lambda v: v and "amphtml" in v.lower())
    amp_url = urljoin(base_url, amp_link["href"]) if amp_link and amp_link.get("href") else None
    return is_amp, amp_url

def _hreflang_links(soup: BeautifulSoup, base: str) -> List[Dict[str, str]]:
    out = []
    for ln in soup.find_all("link", rel=lambda v: v and "alternate" in v.lower()):
        if (ln.get("hreflang") or ln.get("href")):
            out.append({
                "hreflang": (ln.get("hreflang") or "").strip(),
                "href": urljoin(base, (ln.get("href") or "").strip())
            })
    return out

def _validate_hreflang(hreflang: List[Dict[str, str]], base_host: str) -> Dict[str, Any]:
    errors = []
    seen = {}
    x_default_count = 0
    for row in hreflang:
        hl = (row.get("hreflang") or "").strip()
        href = (row.get("href") or "").strip()
        if not hl or not href:
            errors.append({"type": "missing_values", "item": row}); continue
        if hl.lower() == "x-default":
            x_default_count += 1
        elif not _LANG_RE.match(hl):
            errors.append({"type": "invalid_code", "code": hl, "href": href})
        host = urlparse(href).netloc.lower()
        if host and base_host and host != base_host:
            errors.append({"type": "cross_host", "code": hl, "href": href, "host": host})
        key = (hl.lower(), href)
        if key in seen:
            errors.append({"type": "duplicate", "code": hl, "href": href})
        seen[key] = True
    if x_default_count > 1:
        errors.append({"type": "multiple_x_default", "count": x_default_count})
    return {"errors": errors, "ok": len(errors) == 0}

def _extract_links(soup: BeautifulSoup, base: str) -> Tuple[List[str], List[str], List[str]]:
    host = urlparse(base).netloc.lower()
    internal, external, nofollow = [], [], []
    for a in soup.find_all("a", href=True):
        href = urljoin(base, a["href"].strip())
        netloc = urlparse(href).netloc.lower()
        rel = (a.get("rel") or [])
        reltxt = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        (internal if (host and netloc == host) else external).append(href)
        if "nofollow" in reltxt:
            nofollow.append(href)

    def _dedup(lst: List[str]) -> List[str]:
        seen = set(); out = []
        for u in lst:
            if u not in seen:
                out.append(u); seen.add(u)
        return out

    return _dedup(internal), _dedup(external), _dedup(nofollow)

def _head_one(url: str) -> Dict[str, Any]:
    s = _get_session()
    try:
        r = s.head(url, allow_redirects=True, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_HEAD))
        return {
            "url": url,
            "final_url": r.url,
            "status": r.status_code,
            "redirects": len(r.history),
            "headers": dict(r.headers),
        }
    except Exception as e:
        return {"url": url, "status": None, "redirects": 0, "error": str(e)}

def _sample_status(internal: List[str], external: List[str], fast: bool) -> Dict[str, List[Dict[str, Any]]]:
    n_int = max(0, HEAD_SAMPLE_INTERNAL // (2 if fast else 1))
    n_ext = max(0, HEAD_SAMPLE_EXTERNAL // (2 if fast else 1))
    ints = internal[:n_int or HEAD_SAMPLE_INTERNAL]
    exts = external[:n_ext or HEAD_SAMPLE_EXTERNAL]

    rows_int: List[Dict[str, Any]] = []
    rows_ext: List[Dict[str, Any]] = []
    tasks = {u: ("int", u) for u in ints} | {u: ("ext", u) for u in exts}

    if not tasks:
        return {"internal": [], "external": []}

    with ThreadPoolExecutor(max_workers=min(HEAD_MAX_WORKERS, len(tasks))) as tp:
        futs = {tp.submit(_head_one, u): (kind, u) for u, (kind, _) in tasks.items()}
        for f in as_completed(futs):
            kind, _u = futs[f]
            row = f.result()
            (rows_int if kind == "int" else rows_ext).append(row)

    return {"internal": rows_int, "external": rows_ext}

def _robots_and_sitemaps(base: str) -> Dict[str, Any]:
    robots_url = f"{urlparse(base).scheme}://{urlparse(base).netloc}/robots.txt"
    s = _get_session()
    sitemaps: List[Dict[str, Any]] = []
    try:
        r = s.get(robots_url, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_HEAD))
        if r.status_code == 200 and r.text:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    try:
                        h = s.head(sm_url, allow_redirects=True, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_HEAD))
                        sitemaps.append({"url": sm_url, "status": h.status_code})
                    except Exception as e:
                        sitemaps.append({"url": sm_url, "error": str(e)})
    except Exception as e:
        sitemaps = [{"url": None, "error": f"robots fetch failed: {e}"}]
    return {"robots_url": robots_url, "sitemaps": sitemaps, "blocked_by_robots": None}

def _fetch_xml(url: str) -> str | None:
    try:
        r = _get_session().get(url, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_HEAD))
        if r.status_code == 200 and "xml" in (r.headers.get("Content-Type","").lower()):
            return r.text
    except Exception:
        pass
    return None

def _parse_sitemap_urls(xml_text: str) -> List[str]:
    out = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for u in root.findall(".//sm:url/sm:loc", ns):
            if u.text: out.append(u.text.strip())
    except Exception:
        pass
    return out

def _parse_sitemap_index(xml_text: str) -> List[str]:
    idx = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for u in root.findall(".//sm:sitemap/sm:loc", ns):
            if u.text: idx.append(u.text.strip())
    except Exception:
        pass
    return idx

def _collect_sitemap_sample(sitemap_urls: List[str], cap: int = 500) -> List[str]:
    collected: List[str] = []
    for sm in sitemap_urls:
        xml = _fetch_xml(sm)
        if not xml: continue
        urls = _parse_sitemap_urls(xml)
        if urls:
            for u in urls:
                collected.append(u)
                if len(collected) >= cap: return collected
        else:
            # maybe a sitemap index
            idx = _parse_sitemap_index(xml)
            for child in idx:
                x = _fetch_xml(child)
                if not x: continue
                for u in _parse_sitemap_urls(x):
                    collected.append(u)
                    if len(collected) >= cap: return collected
    return collected

def _extract_assets(soup: BeautifulSoup, base: str, max_per_type: int = 10) -> Dict[str, List[str]]:
    css = [urljoin(base, l["href"].strip())
           for l in soup.find_all("link", rel=lambda v: v and "stylesheet" in v.lower(), href=True)]
    js = [urljoin(base, s["src"].strip())
          for s in soup.find_all("script", src=True)]
    imgs = [urljoin(base, im["src"].strip())
            for im in soup.find_all("img", src=True)]
    return {
        "css": css[:max_per_type],
        "js": js[:max_per_type],
        "img": imgs[:max_per_type],
    }

def _head_many(urls: List[str]) -> List[Dict[str, Any]]:
    if not urls: return []
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(HEAD_MAX_WORKERS, len(urls))) as tp:
        futs = [tp.submit(_head_one, u) for u in urls]
        for f in as_completed(futs):
            rows.append(f.result())
    return rows

def _audit_assets(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    compressed = 0
    cacheable = 0
    big_assets: List[Dict[str, Any]] = []
    legacy_imgs: List[str] = []
    for r in rows:
        if r.get("status") is None:
            continue
        h = {k.lower(): v for k, v in (r.get("headers") or {}).items()}
        cenc = (h.get("content-encoding") or "").lower()
        cctl = (h.get("cache-control") or "").lower()
        clen = h.get("content-length")
        ctype = (h.get("content-type") or "").lower()

        if cenc in ("gzip", "br", "deflate"):
            compressed += 1
        if any(tok in cctl for tok in ("max-age", "s-maxage")) and "no-store" not in cctl:
            cacheable += 1

        try:
            size = int(clen) if clen is not None else None
        except Exception:
            size = None
        if size and size >= 300_000:  # 300 KB
            big_assets.append({"url": r.get("final_url") or r.get("url"), "bytes": size})

        if ctype.startswith("image/") and not any(fmt in ctype for fmt in ("webp", "avif")):
            legacy_imgs.append(r.get("final_url") or r.get("url"))

    return {
        "total_sampled": total,
        "compressed_percent": round(100.0 * compressed / total, 1) if total else 0.0,
        "cacheable_percent": round(100.0 * cacheable / total, 1) if total else 0.0,
        "big_assets": sorted(big_assets, key=lambda x: x["bytes"], reverse=True)[:10],
        "legacy_images": legacy_imgs[:10],
    }

def _mixed_content(assets: Dict[str, List[str]], page_url: str) -> Dict[str, Any]:
    if not page_url.startswith("https://"):
        return {"affected": 0, "items": []}
    flat = assets.get("css", []) + assets.get("js", []) + assets.get("img", [])
    bad = [u for u in flat if u.startswith("http://")]
    return {"affected": len(bad), "items": bad[:20]}

def _audit_security_headers(headers: dict) -> Dict[str, Any]:
    hv = {k.lower(): v for k, v in (headers or {}).items()}
    report = {
        "hsts": hv.get("strict-transport-security"),
        "csp": hv.get("content-security-policy"),
        "xcto": hv.get("x-content-type-options"),
        "xfo": hv.get("x-frame-options"),
        "referrer_policy": hv.get("referrer-policy"),
        "permissions_policy": hv.get("permissions-policy") or hv.get("feature-policy"),
    }
    score = 0
    score += 1 if report["hsts"] and "max-age" in report["hsts"] else 0
    score += 1 if report["csp"] else 0
    score += 1 if report["xcto"] == "nosniff" else 0
    score += 1 if report["xfo"] in ("DENY", "SAMEORIGIN") else 0
    score += 1 if report["referrer_policy"] else 0
    score += 1 if report["permissions_policy"] else 0
    report["score_6"] = score
    return report

# ---------- PageSpeed (with CrUX) ----------
def _psi_call(url: str, strategy: str, timeout_sec: int) -> Dict[str, Any]:
    base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    s = _get_session()
    params = {"url": url, "strategy": strategy, "key": PAGESPEED_API_KEY}
    r = s.get(base, params=params, timeout=(HTTP_CONNECT_TIMEOUT, timeout_sec))
    r.raise_for_status()
    data = r.json()
    lr = data.get("lighthouseResult", {})
    audits = lr.get("audits", {})
    cat = lr.get("categories", {}).get("performance", {})
    score = cat.get("score")

    def _crux(src: dict | None) -> dict:
        if not isinstance(src, dict):
            return {}
        def p75(key: str):
            try:
                return round(src.get("metrics", {}).get(key, {}).get("percentiles", {}).get("p75"), 2)
            except Exception:
                return None
        return {
            "LCP_p75_ms": p75("LARGEST_CONTENTFUL_PAINT_MS"),
            "CLS_p75": p75("CUMULATIVE_LAYOUT_SHIFT"),
            "INP_p75_ms": p75("INTERACTION_TO_NEXT_PAINT"),
            "source": src.get("overall_category")
        }

    field_data = _crux(data.get("loadingExperience"))
    origin_field_data = _crux(data.get("originLoadingExperience"))

    def metr(audit_key: str):
        v = audits.get(audit_key, {}).get("numericValue")
        return v if v is not None else None

    metrics = {
        "First Contentful Paint (ms)": metr("first-contentful-paint"),
        "Largest Contentful Paint (ms)": metr("largest-contentful-paint"),
        "Cumulative Layout Shift": audits.get("cumulative-layout-shift", {}).get("numericValue"),
        "Total Blocking Time (ms)": metr("total-blocking-time"),
        "Speed Index (ms)": metr("speed-index"),
        "Time To Interactive (ms)": metr("interactive"),
    }
    return {
        "score": score and round(score * 100),
        "metrics": {k: (round(v, 0) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        "field_data": field_data,
        "origin_field_data": origin_field_data,
    }

def _pagespeed_full(url: str) -> Dict[str, Any]:
    if not (ENABLE_PSI and PAGESPEED_API_KEY):
        return {"enabled": False, "message": "PageSpeed disabled or missing API key"}
    out = {"enabled": True, "mobile": {"metrics": {}}, "desktop": {"metrics": {}}}
    try:
        out["mobile"].update(_psi_call(url, "mobile", timeout_sec=30))
    except Exception as e:
        out["mobile"]["error"] = str(e)
    try:
        out["desktop"].update(_psi_call(url, "desktop", timeout_sec=30))
    except Exception as e:
        out["desktop"]["error"] = str(e)
    return out

def _pagespeed_fast(url: str) -> Dict[str, Any]:
    """Faster PSI: strategy via PSI_STRATEGY_FAST and shorter timeout."""
    if not (ENABLE_PSI and PAGESPEED_API_KEY):
        return {"enabled": False, "message": "PageSpeed disabled or missing API key"}
    strat = PSI_STRATEGY_FAST.lower()
    out = {"enabled": True, "mobile": {"metrics": {}}, "desktop": {"metrics": {}}}

    if strat in ("mobile", "desktop"):
        try:
            data = _psi_call(url, strat, timeout_sec=PSI_TIMEOUT)
            out[strat].update(data)
        except Exception as e:
            out[strat]["error"] = str(e)
        return out

    # both
    try:
        out["mobile"].update(_psi_call(url, "mobile", timeout_sec=PSI_TIMEOUT))
    except Exception as e:
        out["mobile"]["error"] = str(e)
    try:
        out["desktop"].update(_psi_call(url, "desktop", timeout_sec=PSI_TIMEOUT))
    except Exception as e:
        out["desktop"]["error"] = str(e)
    return out

# ---------- Public entry ----------
def get_pagespeed_data(target_url: str, fast: bool | None = None) -> Dict[str, Any]:
    fast = FAST_MODE_DEFAULT if fast is None else fast
    url = _normalize_url(target_url)

    result: Dict[str, Any] = {
        "url": url,
        "status_code": None,
        "load_time_ms": None,
        "content_length": None,
        "title": None,
        "description": None,
        "canonical": None,
        "robots_meta": None,
        "robots_url": None,
        "is_amp": False,
        "amp_url": None,
        "has_open_graph": False,
        "has_twitter_card": False,
        "open_graph": {},
        "twitter_card": {},
        "h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": [],
        "keyword_density_top": [],
        "hreflang": [],
        "hreflang_validation": {"ok": True, "errors": []},
        "images_missing_alt": [],
        "internal_links": [],
        "external_links": [],
        "nofollow_links": [],
        "link_checks": {"internal": [], "external": []},
        "checks": {},
        "performance": {},
        "pagespeed": {"enabled": False},
        "crawl_checks": {},
        "sitemap_summary": {},
        "assets": {"css": [], "js": [], "img": []},
        "assets_audit": {},
        "mixed_content": {},
        "security_headers": {},
        "json_ld": [],
        "microdata": [],
        "rdfa": [],
        "sd_types": {"types": []},
    }

    # --- Fetch page ---
    try:
        resp, elapsed_ms = _fetch(url)
    except Exception as e:
        result["performance"] = {
            "final_url": url,
            "http_version": "HTTP/1.1",
            "redirects": 0,
            "page_size_bytes": 0,
            "load_time_ms": None,
            "https": {"is_https": url.startswith("https"), "ssl_checked": False, "ssl_ok": None},
            "mobile_score": None,
            "desktop_score": None,
        }
        result["pagespeed"] = {"enabled": False, "message": f"Fetch failed: {e}"}
        return result

    final_url = resp.url
    status = resp.status_code
    body = resp.content or b""
    content_len = len(body)
    soup = _soup_parse(body, final_url)

    # --- Security headers (from main response) ---
    security = _audit_security_headers(resp.headers)

    # --- Meta basics ---
    title = (soup.title.string.strip() if (soup.title and soup.title.string) else None)
    m_desc = soup.find("meta", attrs={"name": "description"})
    meta_desc = m_desc.get("content").strip() if m_desc and m_desc.get("content") else None
    m_robots = soup.find("meta", attrs={"name": "robots"})
    robots_meta = (m_robots.get("content") or "").strip().lower() if m_robots else None
    viewport = soup.find("meta", attrs={"name": "viewport"})
    viewport_val = (viewport.get("content") or "").strip() if viewport else None
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    canonical = urljoin(final_url, canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None

    x_robots = resp.headers.get("X-Robots-Tag")
    enc = (resp.headers.get("Content-Encoding") or "").lower() or "none"
    charset = None
    ctype = resp.headers.get("Content-Type") or ""
    m = re.search(r"charset=([\w\-]+)", ctype, flags=re.I)
    if m:
        charset = m.group(1)

    redirects = len(resp.history)
    http_version = "HTTP/1.1"
    is_https = final_url.startswith("https")

    # --- OG / Twitter ---
    og, tw = _collect_metas(soup)
    has_og = bool(og)
    has_twitter = bool(tw)

    # --- AMP ---
    is_amp, amp_url = _detect_amp(soup, final_url)

    # --- Headings ---
    heads = {lvl: [h.get_text(strip=True) for h in soup.find_all(lvl)] for lvl in ["h1","h2","h3","h4","h5","h6"]}

    # --- Links & images ---
    internal, external, nofollow = _extract_links(soup, final_url)
    images = soup.find_all("img")
    imgs_missing = [{"src": urljoin(final_url, (im.get("src") or ""))} for im in images if not (im.get("alt") or "").strip()]
    total_imgs = len(images)
    miss_count = len(imgs_missing)
    alt_percent = round(100.0 * (total_imgs - miss_count) / total_imgs, 2) if total_imgs else 100.0

    # --- Keyword density ---
    kd = _get_text_density(soup)

    # --- robots & sitemaps ---
    crawl = _robots_and_sitemaps(final_url)
    sitemap_urls = [sm.get("url") for sm in crawl.get("sitemaps", []) if sm.get("url")]
    sitemap_sample = _collect_sitemap_sample(sitemap_urls, cap=300)
    # quick orphan candidates = present in sitemap but not linked on this page
    orphans = [u for u in sitemap_sample if u not in set(internal)]

    # --- Link status (concurrent & sampled) ---
    link_checks = _sample_status(internal, external, fast=fast)

    # --- Assets & audits ---
    assets = _extract_assets(soup, final_url, max_per_type=ASSET_SAMPLE_PER_TYPE)
    asset_rows = _head_many(assets["css"] + assets["js"] + assets["img"])
    assets_audit = _audit_assets(asset_rows)
    mixed = _mixed_content(assets, final_url)

    # --- Hreflang validation ---
    base_host = urlparse(final_url).netloc.lower()
    hreflang_list = _hreflang_links(soup, final_url)
    hreflang_check = _validate_hreflang(hreflang_list, base_host)

    # --- Indexability / checks ---
    indexable_ok, indexable_val = _check_indexability(robots_meta, x_robots)
    checks = {
        "canonical": {"ok": bool(canonical), "value": canonical},
        "viewport_meta": {"ok": bool(viewport_val and "width=device-width" in viewport_val.lower()), "value": viewport_val},
        "h1_count": {"ok": (len(heads.get("h1", [])) == 1), "value": len(heads.get("h1", []))},
        "alt_coverage": {"ok": (alt_percent >= 80), "percent": alt_percent, "total_imgs": total_imgs},
        "indexable": {"ok": _bool_badge(indexable_ok), "value": indexable_val},
        "title_length": {"ok": (10 <= _safe_len(title) <= 60), "chars": _safe_len(title)},
        "meta_description_length": {"ok": (50 <= _safe_len(meta_desc) <= 160), "chars": _safe_len(meta_desc)},
        "robots_meta_index": {"ok": (indexable_val != "noindex"), "value": robots_meta or ""},
        "robots_meta_follow": {"ok": ("nofollow" not in (robots_meta or "")), "value": robots_meta or ""},
        "x_robots_tag": {"ok": ("noindex" not in (x_robots or "").lower())},
        "lang": {"ok": bool(soup.find("html") and (soup.find("html").get("lang") or "").strip())},
        "charset": {"ok": bool(charset)},
        "compression": {"ok": (enc in ["gzip", "br", "deflate"]), "value": enc},
    }

    # --- Performance summary ---
    perf = {
        "final_url": final_url,
        "http_version": http_version,
        "redirects": redirects,
        "page_size_bytes": content_len,
        "load_time_ms": round(elapsed_ms),
        "https": {"is_https": is_https, "ssl_checked": False, "ssl_ok": None},
        "mobile_score": None,
        "desktop_score": None,
    }

    # --- PageSpeed (with CrUX) ---
    if fast:
        ps = _pagespeed_fast(final_url) if PSI_IN_FAST else {"enabled": False, "message": "Skipped (FAST_MODE_DEFAULT)"}
    else:
        ps = _pagespeed_full(final_url)
    if ps.get("enabled"):
        perf["mobile_score"] = ps.get("mobile", {}).get("score")
        perf["desktop_score"] = ps.get("desktop", {}).get("score")

    # --- JSON-LD quick scan ---
    json_ld = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            json_ld.append(json.loads(tag.string or "{}"))
        except Exception:
            continue

    result.update({
        "status_code": status,
        "load_time_ms": round(elapsed_ms),
        "content_length": content_len,
        "title": title,
        "description": meta_desc,
        "canonical": canonical,
        "robots_meta": robots_meta,
        "robots_url": crawl.get("robots_url"),
        "is_amp": is_amp,
        "amp_url": amp_url,
        "has_open_graph": has_og,
        "has_twitter_card": has_twitter,
        "open_graph": og,
        "twitter_card": tw,
        "h1": heads["h1"], "h2": heads["h2"], "h3": heads["h3"], "h4": heads["h4"], "h5": heads["h5"], "h6": heads["h6"],
        "keyword_density_top": kd,
        "hreflang": hreflang_list,
        "hreflang_validation": hreflang_check,
        "images_missing_alt": imgs_missing,
        "internal_links": internal,
        "external_links": external,
        "nofollow_links": nofollow,
        "link_checks": link_checks,
        "checks": checks,
        "performance": perf,
        "pagespeed": ps,
        "crawl_checks": {"sitemaps": crawl.get("sitemaps", []), "blocked_by_robots": crawl.get("blocked_by_robots")},
        "sitemap_summary": {
            "sitemaps_found": len(sitemap_urls),
            "sampled_url_count": len(sitemap_sample),
            "possible_orphans_sampled": orphans[:50],
        },
        "assets": assets,
        "assets_audit": assets_audit,
        "mixed_content": mixed,
        "security_headers": security,
        "json_ld": json_ld,
        "microdata": [],
        "rdfa": [],
        "sd_types": {"types": list({item.get("@type") for item in json_ld if isinstance(item, dict) and item.get("@type")}) if json_ld else []},
    })
    return result
