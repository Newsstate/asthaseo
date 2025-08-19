# app/seo.py
from __future__ import annotations

import os
import re
import time
import json
import html
import math
import string
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

# ====== Config / constants ======
USER_AGENT = (
    "SEO-Inspector/1.1 (+https://example.com) "
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = 15               # seconds for HTTP requests
HEAD_SAMPLE = 8            # how many links to sample for status checks

# PageSpeed key (optional)
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY")

# Rendered vs Static flags
ENABLE_RENDERED = os.getenv("ENABLE_RENDERED", "0") == "1"
RENDER_TIMEOUT_MS = int(os.getenv("RENDER_TIMEOUT_MS", "45000"))      # 45s default
RENDER_WAIT_UNTIL = os.getenv("RENDER_WAIT_UNTIL", "networkidle")     # 'load' | 'domcontentloaded' | 'networkidle'

# Basic stopwords for crude keyword density
STOPWORDS = set("""
a an the and or but if then else for to of in on at by with from as this that those these is are be was were been being
you your we our they their he she it its not no yes do does did done have has had having i me my mine ourselves himself
herself itself themselves them can will would should could may might must about above across after again against all almost
also am among amongst around because before behind below beneath beside besides between beyond both during each either few
more most much many nor off over under up down out into than too very via per just so only such own same very
""".split())


# ====== Helpers ======
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
    # Return (ok, value) where ok=True means indexable
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
                    try:
                        h = requests.head(sm_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
                        sitemaps.append({"url": sm_url, "status": h.status_code})
                    except Exception as e:
                        sitemaps.append({"url": sm_url, "error": str(e)})
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
        # round ms-ish values
        clean = {k: (round(v, 0) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        return {"score": score and round(score * 100), "metrics": clean}

    try:
        out["mobile"].update(call("mobile"))
    except Exception as e:
        out["mobile"]["error"] = str(e)
    try:
        out["desktop"].update(call("desktop"))
    except Exception as e:
        out["desktop"]["error"] = str(e)

    return out


# ====== Rendered DOM (Playwright) ======
def _extract_render_summary(soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
    title = (soup.title.string.strip() if (soup.title and soup.title.string) else None)
    m_desc = soup.find("meta", attrs={"name": "description"})
    meta_desc = m_desc.get("content").strip() if m_desc and m_desc.get("content") else None
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    canonical = urljoin(base_url, canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None
    og_count = 0
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").strip().lower()
        if name.startswith("og:"):
            og_count += 1
    h1_count = len([h for h in soup.find_all("h1")])
    text_len = len((soup.get_text(" ", strip=True) or "").strip())
    return {
        "title": title,
        "description": meta_desc,
        "canonical": canonical,
        "og_count": og_count,
        "h1_count": h1_count,
        "text_len": text_len,
    }


def _rendered_compare(url: str, soup_static: BeautifulSoup, timeout_ms: int = 30000, wait_until: str = "networkidle") -> Dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return {"matrix": [], "error": f"Playwright not available: {e}"}

    static = _extract_render_summary(soup_static, url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            page = browser.new_page(user_agent=USER_AGENT, viewport={"width": 1366, "height": 768})
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            html_r = page.content()
            browser.close()
    except Exception as e:
        return {"matrix": [], "error": f"Playwright render skipped/failed: {e}"}

    soup_r = BeautifulSoup(html_r, "html.parser")
    rendered = _extract_render_summary(soup_r, url)

    def row(label: str, key: str):
        before = static.get(key)
        after = rendered.get(key)
        return {
            "label": label,
            "before": "—" if before in (None, "") else before,
            "after": "—" if after in (None, "") else after,
            "changed": before != after,
        }

    matrix = [
        row("Title", "title"),
        row("Meta Description", "description"),
        row("Canonical", "canonical"),
        row("H1 count", "h1_count"),
        row("Text length", "text_len"),
        row("OG tags (count)", "og_count"),
    ]
    return {"matrix": matrix}


# ====== Public entry point (sync) ======
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

    # --- Fetch ---
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

    # --- Basics ---
    final_url = resp.url
    status = resp.status_code
    body = resp.content or b""
    content_len = len(body)
    soup = BeautifulSoup(body, "html.parser")

    # --- Title & meta ---
    title = (soup.title.string.strip() if (soup.title and soup.title.string) else None)
    m_desc = soup.find("meta", attrs={"name": "description"})
    meta_desc = m_desc.get("content").strip() if m_desc and m_desc.get("content") else None
    m_robots = soup.find("meta", attrs={"name": "robots"})
    robots_meta = (m_robots.get("content") or "").strip().lower() if m_robots else None
    viewport = soup.find("meta", attrs={"name": "viewport"})
    viewport_val = (viewport.get("content") or "").strip() if viewport else None
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    canonical = urljoin(final_url, canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None

    # --- Headers / HTTP-ish info ---
    x_robots = resp.headers.get("X-Robots-Tag")
    enc = (resp.headers.get("Content-Encoding") or "").lower() or "none"
    charset = None
    ctype = resp.headers.get("Content-Type") or ""
    m = re.search(r"charset=([\w\-]+)", ctype, flags=re.I)
    if m:
        charset = m.group(1)
    redirects = len(resp.history)
    http_version = "HTTP/1.1"     # requests doesn't expose version; OK for display
    is_https = final_url.startswith("https")

    # --- OG / Twitter ---
    og, tw = _collect_metas(soup)
    has_og = bool(og)
    has_twitter = bool(tw)

    # --- AMP ---
    is_amp, amp_url = _detect_amp(soup, final_url)

    # --- Headings ---
    heads = {}
    for lvl in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        heads[lvl] = [h.get_text(strip=True) for h in soup.find_all(lvl)]

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

    # --- Link checks (sample) ---
    link_checks = {
        "internal": _sample_status(internal),
        "external": _sample_status(external),
    }

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

    # --- PageSpeed (optional) ---
    ps = _pagespeed(final_url)
    if ps.get("enabled"):
        perf["mobile_score"] = ps.get("mobile", {}).get("score")
        perf["desktop_score"] = ps.get("desktop", {}).get("score")

    # --- Rendered vs Static ---
    rendered_diff = {"matrix": [], "error": "Rendered DOM check disabled (set ENABLE_RENDERED=1)"}
    if ENABLE_RENDERED:
        try:
            rendered_diff = _rendered_compare(final_url, soup, timeout_ms=RENDER_TIMEOUT_MS, wait_until=RENDER_WAIT_UNTIL)
            if not rendered_diff.get("matrix"):
                rendered_diff.setdefault("error", "Rendered check produced no differences (or extractor returned empty).")
        except Exception as e:
            rendered_diff = {"matrix": [], "error": f"Playwright render skipped/failed: {e}"}

    # --- JSON-LD (lightweight) ---
    json_ld = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            json_ld.append(json.loads(tag.string or "{}"))
        except Exception:
            continue

    # --- Assemble result ---
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
        "h1": heads["h1"],
        "h2": heads["h2"],
        "h3": heads["h3"],
        "h4": heads["h4"],
        "h5": heads["h5"],
        "h6": heads["h6"],
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
        "rendered_diff": rendered_diff,
        "json_ld": json_ld,
        "microdata": [],   # placeholder
        "rdfa": [],        # placeholder
        "sd_types": {"types": list({item.get("@type") for item in json_ld if isinstance(item, dict) and item.get("@type")}) if json_ld else []},
    })

    return result
