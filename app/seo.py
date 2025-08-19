# app/seo.py
from __future__ import annotations

import os
import re
import time
import json
import math
import html
import string
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

# ---- Config / constants ----
USER_AGENT = (
    "SEO-Inspector/1.0 (+https://example.com) "
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
)
TIMEOUT = 15  # seconds for HTTP requests
HEAD_SAMPLE = 8  # how many links to sample for status checks

PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY")  # set in Render dashboard

# Basic stopwords for crude keyword density
STOPWORDS = set("""
a an the and or but if then else for to of in on at by with from as this that those these is are be was were been being
you your we our they their he she it its not no yes do does did done have has had having i me my mine ourselves himself
herself itself themselves them can will would should could may might must about above across after again against all almost
also am among amongs around because before behind below beneath beside besides between beyond both during each either few
more most much many nor off over under up down out into than too very via per just so only such own same very
""".split())

# ---- Helpers ----
def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    parsed = urlparse(u)
    if not parsed.scheme:
        u = "https://" + u
    return u

def _fetch(url: str) -> Tuple[requests.Response, float]:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate, br"}
    start = time.perf_counter()
    resp = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return resp, elapsed_ms

def _get_text_density(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    # crude keyword density on visible text
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    words = [w for w in text.split() if w and w not in STOPWORDS and len(w) > 2]
    if not words:
        return []
    total = len(words)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
    out: List[Dict[str, Any]] = []
    for w, c in top:
        out.append({"word": w, "count": c, "percent": round(c * 100.0 / total, 2)})
    return out

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
    # Return (ok, value) where ok=True means indexable
    meta = (robots_meta or "").lower()
    x = (x_robots or "").lower()
    tokens = set(re.split(r"[,\s]+", meta + " " + x))
    if not tokens or tokens == {""}:
        return (None, "unknown")
    if "noindex" in tokens or "none" in tokens:
        return (False, "noindex")
    return (True, "index")

def _detect_amp(soup: BeautifulSoup, base_url: str) -> Tuple[bool, str | None]:
    html_tag = soup.find("html")
    is_amp = False
    if html_tag:
        is_amp = bool(html_tag.has_attr("amp") or html_tag.has_attr("⚡"))
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

def _extract_links(soup: BeautifulSoup, base: str) -> Tuple[List[str], List[str], List[str]]:
    parsed = urlparse(base)
    host = parsed.netloc.lower()
    internal, external, nofollow = [], [], []
    for a in soup.find_all("a", href=True):
        href = urljoin(base, a["href"].strip())
        netloc = urlparse(href).netloc.lower()
        rel = (a.get("rel") or [])
        reltxt = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        if host and netloc == host:
            internal.append(href)
        else:
            external.append(href)
        if "nofollow" in reltxt:
            nofollow.append(href)
    # de-dup while preserving order
    def _dedup(lst: List[str]) -> List[str]:
        seen = set()
        out = []
        for u in lst:
            if u not in seen:
                out.append(u)
                seen.add(u)
        return out
    return _dedup(internal), _dedup(external), _dedup(nofollow)

def _sample_status(urls: List[str]) -> List[Dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT}
    out = []
    for url in urls[:HEAD_SAMPLE]:
        try:
            r = requests.head(url, headers=headers, allow_redirects=True, timeout=TIMEOUT)
            out.append({
                "url": url,
                "final_url": r.url,
                "status": r.status_code,
                "redirects": len(r.history)
            })
        except Exception as e:
            out.append({"url": url, "status": None, "redirects": 0, "error": str(e)})
    return out

def _robots_and_sitemaps(base: str) -> Dict[str, Any]:
    parsed = urlparse(base)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    headers = {"User-Agent": USER_AGENT}
    sitemaps: List[Dict[str, Any]] = []
    blocked_by_robots: bool | None = None
    try:
        r = requests.get(robots_url, headers=headers, timeout=TIMEOUT)
        if r.status_code == 200 and r.text:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    # quick HEAD to see if reachable
                    try:
                        h = requests.head(sm_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
                        sitemaps.append({"url": sm_url, "status": h.status_code})
                    except Exception as e:
                        sitemaps.append({"url": sm_url, "error": str(e)})
        else:
            sitemaps = []
    except Exception as e:
        sitemaps = [{"url": None, "error": f"robots fetch failed: {e}"}]
    return {"robots_url": robots_url, "sitemaps": sitemaps, "blocked_by_robots": blocked_by_robots}

def _pagespeed(url: str) -> Dict[str, Any]:
    # If no key, return disabled
    if not PAGESPEED_API_KEY:
        return {"enabled": False, "message": "Set PAGESPEED_API_KEY to enable PageSpeed"}
    base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    headers = {"User-Agent": USER_AGENT}
    out = {"enabled": True, "mobile": {"metrics": {}}, "desktop": {"metrics": {}}}

    def call(strategy: str) -> Dict[str, Any]:
        params = {"url": url, "strategy": strategy, "key": PAGESPEED_API_KEY}
        r = requests.get(base, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        lr = data.get("lighthouseResult", {})
        audits = lr.get("audits", {})
        cat = lr.get("categories", {}).get("performance", {})
        score = cat.get("score")
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
        return {"score": score and round(score * 100), "metrics": {k: (round(v, 0) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}}

    try:
        out["mobile"].update(call("mobile"))
    except Exception as e:
        out["mobile"]["error"] = str(e)
    try:
        out["desktop"].update(call("desktop"))
    except Exception as e:
        out["desktop"]["error"] = str(e)

    return out

# ---- Public entry point (sync) ----
def get_pagespeed_data(target_url: str) -> Dict[str, Any]:
    """
    Main analyzer. Returns a dict that your template reads.
    """
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
        "images_missing_alt": [],
        "internal_links": [],
        "external_links": [],
        "nofollow_links": [],
        "link_checks": {"internal": [], "external": []},
        "checks": {},
        "performance": {},
        "pagespeed": {"enabled": False},
        "crawl_checks": {},
        "rendered_diff": {"matrix": [], "error": "Rendered DOM check skipped on this host"},
        "json_ld": [],
        "microdata": [],
        "rdfa": [],
        "sd_types": {"types": []},
    }

    try:
        resp, elapsed_ms = _fetch(url)
    except Exception as e:
        # network error — return minimal info
        result["load_time_ms"] = None
        result["status_code"] = None
        result["content_length"] = None
        result["checks"] = {}
        result["performance"] = {
            "final_url": url,
            "http_version": "HTTP/1.1",
            "redirects": 0,
            "page_size_bytes": 0,
            "load_time_ms": None,
            "https": {"is_https": url.startswith("https"), "ssl_checked": False, "ssl_ok": None},
        }
        result["pagespeed"] = {
            "enabled": False,
            "message": f"Fetch failed: {e}",
        }
        return result

    # Basics
    final_url = resp.url
    status = resp.status_code
    body = resp.content or b""
    content_len = len(body)

    # Parse
    soup = BeautifulSoup(body, "html.parser")

    # Title & meta
    title = (soup.title.string.strip() if (soup.title and soup.title.string) else None)
    m_desc = soup.find("meta", attrs={"name": "description"})
    meta_desc = m_desc.get("content").strip() if m_desc and m_desc.get("content") else None
    m_robots = soup.find("meta", attrs={"name": "robots"})
    robots_meta = m_robots.get("content").strip().lower() if m_robots and m_robots.get("content") else None
    viewport = soup.find("meta", attrs={"name": "viewport"})
    viewport_val = viewport.get("content").strip() if viewport and viewport.get("content") else None
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    canonical = urljoin(final_url, canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None

    # Headers checks
    x_robots = resp.headers.get("X-Robots-Tag")
    enc = (resp.headers.get("Content-Encoding") or "").lower() or "none"
    charset = None
    ctype = resp.headers.get("Content-Type") or ""
    m = re.search(r"charset=([\w\-]+)", ctype, flags=re.I)
    if m:
        charset = m.group(1)

    # HTTP-ish info
    redirects = len(resp.history)
    http_version = "HTTP/1.1"  # requests doesn't expose version; OK for display
    is_https = final_url.startswith("https")

    # OG / Twitter
    og, tw = _collect_metas(soup)
    has_og = bool(og)
    has_twitter = bool(tw)

    # AMP
    is_amp, amp_url = _detect_amp(soup, final_url)

    # Headings
    heads = {}
    for lvl in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        heads[lvl] = [h.get_text(strip=True) for h in soup.find_all(lvl)]
    result.update(heads)

    # Links, images
    internal, external, nofollow = _extract_links(soup, final_url)
    images = soup.find_all("img")
    imgs_missing = [{"src": urljoin(final_url, (im.get("src") or ""))} for im in images if not (im.get("alt") or "").strip()]
    total_imgs = len(images)
    miss_count = len(imgs_missing)
    alt_percent = round(100.0 * (total_imgs - miss_count) / total_imgs, 2) if total_imgs else 100.0

    # Keyword density
    kd = _get_text_density(soup)

    # robots & sitemaps
    crawl = _robots_and_sitemaps(final_url)

    # Link checks (sample)
    link_checks = {
        "internal": _sample_status(internal),
        "external": _sample_status(external),
    }

    # Indexability / checks
    indexable_ok, indexable_val = _check_indexability(robots_meta, x_robots)
    checks = {
        "canonical": {"ok": bool(canonical), "value": canonical},
        "viewport_meta": {"ok": bool(viewport_val and "width=device-width" in viewport_val.lower()), "value": viewport_val},
        "h1_count": {"ok": (len(result["h1"]) == 1), "value": len(result["h1"])},
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

    # Performance summary
    perf = {
        "final_url": final_url,
        "http_version": http_version,
        "redirects": redirects,
        "page_size_bytes": content_len,
        "load_time_ms": round(elapsed_ms),
        "https": {"is_https": is_https, "ssl_checked": False, "ssl_ok": None},
        # scores filled from PSI below if available
        "mobile_score": None,
        "desktop_score": None,
    }

    # PageSpeed (optional)
    ps = _pagespeed(final_url)
    if ps.get("enabled"):
        try:
            perf["mobile_score"] = ps.get("mobile", {}).get("score")
            perf["desktop_score"] = ps.get("desktop", {}).get("score")
        except Exception:
            pass

    # JSON-LD / microdata / RDFa (lightweight)
    json_ld = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            json_ld.append(json.loads(tag.string or "{}"))
        except Exception:
            continue

    # Assemble result
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
        "keyword_density_top": kd,
        "hreflang": _hreflang_links(soup, final_url),
        "images_missing_alt": imgs_missing,
        "internal_links": internal,
        "external_links": external,
        "nofollow_links": nofollow,
        "link_checks": link_checks,
        "checks": checks,
        "performance": perf,
        "pagespeed": ps,
        "crawl_checks": {"sitemaps": crawl.get("sitemaps", []), "blocked_by_robots": crawl.get("blocked_by_robots")},
        "json_ld": json_ld,
        "microdata": [],   # placeholder
        "rdfa": [],        # placeholder
        "sd_types": {"types": list({item.get("@type") for item in json_ld if isinstance(item, dict) and item.get("@type")}) if json_ld else []},
    })

    return result
