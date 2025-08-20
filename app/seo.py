# app/seo.py
from __future__ import annotations

import os
import re
import time
import json
import html
import string
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# ---------------- Tunables via env ----------------
FAST_MODE_DEFAULT = os.getenv("FAST_MODE_DEFAULT", "1") == "1"

HTTP_TIMEOUT_MAIN = float(os.getenv("HTTP_TIMEOUT_MAIN", "10"))
HTTP_TIMEOUT_HEAD = float(os.getenv("HTTP_TIMEOUT_HEAD", "5"))
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "5"))

HEAD_MAX_WORKERS = int(os.getenv("HEAD_MAX_WORKERS", "12"))
HEAD_SAMPLE_INTERNAL = int(os.getenv("HEAD_SAMPLE_INTERNAL", "6"))
HEAD_SAMPLE_EXTERNAL = int(os.getenv("HEAD_SAMPLE_EXTERNAL", "6"))
ASSET_SAMPLE_PER_TYPE = int(os.getenv("ASSET_SAMPLE_PER_TYPE", "8"))

ENABLE_PSI = os.getenv("ENABLE_PSI", "1") == "1"
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY")
PSI_IN_FAST = os.getenv("PSI_IN_FAST", "0") == "1"  # default skip PSI in fast mode
PSI_STRATEGY_FAST = os.getenv("PSI_STRATEGY_FAST", "mobile")
PSI_TIMEOUT = int(os.getenv("PSI_TIMEOUT", "15"))

USE_LXML = os.getenv("USE_LXML", "1") == "1"

# ---------------- Constants & headers ----------------
_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
USER_AGENT = _CHROME_UA

_BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

STOPWORDS = set("""
a an the and or but if then else for to of in on at by with from as this that those these is are be was were been being
you your we our they their he she it its not no yes do does did done have has had having i me my mine ourselves himself
herself itself themselves them can will would should could may might must about above across after again against all almost
also am among amongst around because before behind below beneath beside besides between beyond both during each either few
more most much many nor off over under up down out into than too very via per just so only such own same very
""".split())

_LANG_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z]{4})?(-[a-zA-Z]{2}|\d{3})?$")

# ---------------- Session ----------------
_SESSION: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.trust_env = False
        retry = Retry(
            total=1,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
        )
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update(_BASE_HEADERS)
        _SESSION = s
    return _SESSION

# ---------------- Helpers ----------------
def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    p = urlparse(u)
    if not p.scheme:
        u = "https://" + u
    return u

def _soup_parse(body: bytes) -> BeautifulSoup:
    if USE_LXML:
        try:
            return BeautifulSoup(body, "lxml")
        except Exception:
            pass
    return BeautifulSoup(body, "html.parser")

def _fetch(url: str) -> Tuple[requests.Response, float]:
    s = _get_session()
    start = time.perf_counter()
    r = s.get(url, allow_redirects=True, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_MAIN))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return r, elapsed_ms

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

def _detect_amp(soup: BeautifulSoup, base_url: str) -> Tuple[bool, Optional[str]]:
    html_tag = soup.find("html")
    is_amp = bool(html_tag and (html_tag.has_attr("amp") or html_tag.has_attr("⚡")))
    amp_link = soup.find("link", rel=lambda v: v and "amphtml" in v.lower())
    amp_url = urljoin(base_url, amp_link["href"]) if amp_link and amp_link.get("href") else None
    return is_amp, amp_url

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
    # dedup
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
    n_int = HEAD_SAMPLE_INTERNAL if not fast else max(1, HEAD_SAMPLE_INTERNAL // 2)
    n_ext = HEAD_SAMPLE_EXTERNAL if not fast else max(1, HEAD_SAMPLE_EXTERNAL // 2)
    ints = internal[:n_int]
    exts = external[:n_ext]
    rows_int: List[Dict[str, Any]] = []
    rows_ext: List[Dict[str, Any]] = []
    tasks = {u: ("int", u) for u in ints} | {u: ("ext", u) for u in exts}
    if not tasks:
        return {"internal": [], "external": []}
    with ThreadPoolExecutor(max_workers=min(HEAD_MAX_WORKERS, len(tasks))) as tp:
        futs = {tp.submit(_head_one, u): kind for u, (kind, _) in tasks.items()}
        for f in as_completed(futs):
            kind = futs[f]
            row = f.result()
            (rows_int if kind == "int" else rows_ext).append(row)
    return {"internal": rows_int, "external": rows_ext}

def _robots_and_sitemaps(base: str) -> Dict[str, Any]:
    try:
        parts = urlparse(base)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    except Exception:
        robots_url = None
    sitemaps: List[Dict[str, Any]] = []
    if robots_url:
        try:
            r = _get_session().get(robots_url, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT_HEAD))
            if r.status_code == 200 and r.text:
                for line in r.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemaps.append({"url": line.split(":", 1)[1].strip(), "status": None})
        except Exception as e:
            sitemaps = [{"url": None, "error": f"robots fetch failed: {e}"}]
    return {"robots_url": robots_url, "sitemaps": sitemaps, "blocked_by_robots": None}

def _extract_assets(soup: BeautifulSoup, base: str, max_per_type: int = 8) -> Dict[str, List[str]]:
    css = [urljoin(base, l["href"].strip())
           for l in soup.find_all("link", rel=lambda v: v and "stylesheet" in v.lower(), href=True)]
    js = [urljoin(base, s["src"].strip())
          for s in soup.find_all("script", src=True)]
    imgs = [urljoin(base, im["src"].strip())
            for im in soup.find_all("img", src=True)]
    return {"css": css[:max_per_type], "js": js[:max_per_type], "img": imgs[:max_per_type]}

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
        if size and size >= 300_000:
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

def _check_indexability(robots_meta: Optional[str], x_robots: Optional[str]) -> Tuple[Optional[bool], str]:
    meta = (robots_meta or "").lower()
    x = (x_robots or "").lower()
    tokens = set(re.split(r"[,\s]+", (meta + " " + x).strip()))
    if not tokens or tokens == {""}:
        return (None, "unknown")
    if "noindex" in tokens or "none" in tokens:
        return (False, "noindex")
    return (True, "index")

# ---------------- PageSpeed (optional) ----------------
def _psi_call(url: str, strategy: str, timeout_sec: int) -> Dict[str, Any]:
    base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    r = _get_session().get(base, params={"url": url, "strategy": strategy, "key": PAGESPEED_API_KEY},
                           timeout=(HTTP_CONNECT_TIMEOUT, timeout_sec))
    r.raise_for_status()
    data = r.json()
    lr = data.get("lighthouseResult", {})
    audits = lr.get("audits", {})
    score = lr.get("categories", {}).get("performance", {}).get("score")
    def metr(k): 
        v = audits.get(k, {}).get("numericValue")
        return v if isinstance(v, (int, float)) else None
    metrics = {
        "First Contentful Paint (ms)": metr("first-contentful-paint"),
        "Largest Contentful Paint (ms)": metr("largest-contentful-paint"),
        "Cumulative Layout Shift": audits.get("cumulative-layout-shift", {}).get("numericValue"),
        "Total Blocking Time (ms)": metr("total-blocking-time"),
        "Speed Index (ms)": metr("speed-index"),
        "Time To Interactive (ms)": metr("interactive"),
    }
    return {"score": score and round(score * 100), "metrics": metrics}

def _pagespeed(url: str, fast: bool) -> Dict[str, Any]:
    if not (ENABLE_PSI and PAGESPEED_API_KEY):
        return {"enabled": False, "message": "PageSpeed disabled or missing API key"}
    if fast and not PSI_IN_FAST:
        return {"enabled": False, "message": "Skipped in fast mode"}
    out = {"enabled": True, "mobile": {"metrics": {}}, "desktop": {"metrics": {}}}
    try:
        if fast and PSI_STRATEGY_FAST in ("mobile", "desktop"):
            out[PSI_STRATEGY_FAST].update(_psi_call(url, PSI_STRATEGY_FAST, PSI_TIMEOUT))
        else:
            out["mobile"].update(_psi_call(url, "mobile", PSI_TIMEOUT))
            out["desktop"].update(_psi_call(url, "desktop", PSI_TIMEOUT))
    except Exception as e:
        out["error"] = str(e)
    return out

# ---------------- Public entry ----------------
def get_pagespeed_data(target_url: str, fast: Optional[bool] = None) -> Dict[str, Any]:
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
        "mixed_content": {"affected": 0, "items": []},
        "security_headers": {},
        "json_ld": [],
        "microdata": [],
        "rdfa": [],
        "sd_types": {"types": []},
        "rendered_diff": {"matrix": [], "error": "Rendered DOM check disabled"},
        "errors": [],
        "notes": {},
    }

    try:
        resp, elapsed_ms = _fetch(url)
    except Exception as e:
        result["errors"].append(f"Fetch failed: {e}")
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
        return result

    final_url = str(resp.url)
    status = resp.status_code
    body_bytes = (resp.text or "").encode(resp.encoding or "utf-8", errors="ignore")
    content_len = len(body_bytes)
    soup = _soup_parse(body_bytes)

    # Security headers (from main response)
    hv = {k.lower(): v for k, v in resp.headers.items()}
    security = {
        "hsts": hv.get("strict-transport-security"),
        "csp": hv.get("content-security-policy"),
        "xcto": hv.get("x-content-type-options"),
        "xfo": hv.get("x-frame-options"),
        "referrer_policy": hv.get("referrer-policy"),
        "permissions_policy": hv.get("permissions-policy") or hv.get("feature-policy"),
        "score_6": 0
    }
    if security["hsts"] and "max-age" in security["hsts"]: security["score_6"] += 1
    if security["csp"]: security["score_6"] += 1
    if (hv.get("x-content-type-options") or "").lower() == "nosniff": security["score_6"] += 1
    if (hv.get("x-frame-options") or "") in ("DENY", "SAMEORIGIN"): security["score_6"] += 1
    if security["referrer_policy"]: security["score_6"] += 1
    if security["permissions_policy"]: security["score_6"] += 1

    # Meta basics
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
    redirects = len(resp.history)
    http_version = "HTTP/1.1"
    is_https = final_url.startswith("https")

    # OG / Twitter
    og, tw = _collect_metas(soup)
    has_og = bool(og)
    has_twitter = bool(tw)

    # AMP
    is_amp, amp_url = _detect_amp(soup, final_url)

    # Headings
    heads = {lvl: [h.get_text(strip=True) for h in soup.find_all(lvl)] for lvl in ["h1","h2","h3","h4","h5","h6"]}

    # Links & images
    internal, external, nofollow = _extract_links(soup, final_url)
    images = soup.find_all("img")
    imgs_missing = [{"src": urljoin(final_url, (im.get("src") or ""))} for im in images if not (im.get("alt") or "").strip()]
    total_imgs = len(images)
    miss_count = len(imgs_missing)
    alt_percent = round(100.0 * (total_imgs - miss_count) / total_imgs, 2) if total_imgs else 100.0

    # Keyword density
    kd = _get_text_density(soup)

    # robots & sitemaps (lightweight)
    crawl = _robots_and_sitemaps(final_url)
    sitemap_urls = [sm.get("url") for sm in crawl.get("sitemaps", []) if sm.get("url")]

    # Link status (sampled)
    link_checks = _sample_status(internal, external, fast=fast)

    # Assets & audits
    assets = _extract_assets(soup, final_url, max_per_type=ASSET_SAMPLE_PER_TYPE if not fast else max(1, ASSET_SAMPLE_PER_TYPE // 2))
    asset_rows = []
    try:
        flat = assets["css"] + assets["js"] + assets["img"]
        with ThreadPoolExecutor(max_workers=min(HEAD_MAX_WORKERS, len(flat))) as tp:
            futs = [tp.submit(_head_one, u) for u in flat]
            for f in as_completed(futs):
                asset_rows.append(f.result())
    except Exception:
        pass
    assets_audit = _audit_assets(asset_rows)
    mixed = _mixed_content(assets, final_url)

    # Hreflang validation
    base_host = urlparse(final_url).netloc.lower()
    hreflang_list = []
    for ln in soup.find_all("link", rel=lambda v: v and "alternate" in v.lower()):
        if (ln.get("hreflang") or ln.get("href")):
            hreflang_list.append({"hreflang": (ln.get("hreflang") or "").strip(),
                                   "href": urljoin(final_url, (ln.get("href") or "").strip())})
    errors_hl = []
    x_default_count = sum(1 for r in hreflang_list if (r.get("hreflang") or "").lower() == "x-default")
    if x_default_count > 1:
        errors_hl.append({"type": "multiple_x_default", "count": x_default_count})
    hreflang_check = {"ok": len(errors_hl) == 0, "errors": errors_hl}

    # Indexability / checks
    indexable_ok, indexable_val = _check_indexability(robots_meta, x_robots)
    checks = {
        "canonical": {"ok": bool(canonical), "value": canonical},
        "viewport_meta": {"ok": bool(viewport_val and "width=device-width" in (viewport_val or "").lower()), "value": viewport_val},
        "h1_count": {"ok": (len(heads.get("h1", [])) == 1), "value": len(heads.get("h1", []))},
        "alt_coverage": {"ok": (alt_percent >= 80), "percent": alt_percent, "total_imgs": total_imgs},
        "indexable": {"ok": True if indexable_ok is True else (None if indexable_ok is None else False), "value": indexable_val},
        "title_length": {"ok": (10 <= len(title or "") <= 60), "chars": len(title or "")},
        "meta_description_length": {"ok": (50 <= len(meta_desc or "") <= 160), "chars": len(meta_desc or "")},
        "robots_meta_index": {"ok": (indexable_val != "noindex"), "value": robots_meta or ""},
        "robots_meta_follow": {"ok": ("nofollow" not in (robots_meta or "")), "value": robots_meta or ""},
        "x_robots_tag": {"ok": ("noindex" not in (x_robots or "").lower())},
        "lang": {"ok": bool(soup.find("html") and (soup.find("html").get("lang") or "").strip())},
        "charset": {"ok": bool(re.search(r"charset=([\w\-]+)", resp.headers.get("Content-Type",""), flags=re.I))},
        "compression": {"ok": ((resp.headers.get("Content-Encoding") or "").lower() in ["gzip", "br", "deflate"]),
                        "value": (resp.headers.get("Content-Encoding") or "none").lower()},
    }

    # Performance summary
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

    # PageSpeed (optional)
    ps = _pagespeed(final_url, fast)
    if ps.get("enabled"):
        perf["mobile_score"] = ps.get("mobile", {}).get("score")
        perf["desktop_score"] = ps.get("desktop", {}).get("score")

    # JSON-LD (simple)
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
            "sampled_url_count": 0,
            "possible_orphans_sampled": [],
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
