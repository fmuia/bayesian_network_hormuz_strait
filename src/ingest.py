"""URL / title ingestion for the news translator (post-P5 improvements round).

Resolves the analyst's single input box into an ``Article``-ready
``(headline, body, source, source_type)`` before the translator runs:

- a **body** filled in the expander ⇒ the top field is a **headline** (manual path);
- otherwise a **URL** from a curated news outlet ⇒ fetch the page + split into
  headline/body, with the outlet's credibility ``source_type``;
- a **URL** from an **unlisted / non-news** domain ⇒ rejected with a clear message;
- anything else ⇒ a bare **title** (the existing headline-only path).

Pure and import-safe (no Streamlit; ``requests`` / ``bs4`` are imported lazily).
Network egress is confined to allow-listed outlets and goes through an injectable
``fetcher``, so tests and the offline/fake mode need no network (this mirrors the
fake translator in :mod:`src.translator`).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple
from urllib.parse import urlparse

from src.translator import SourceType

Fetcher = Callable[[str], str]


@dataclass(frozen=True)
class SiteInfo:
    """A registry entry: the human outlet name + its credibility ``source_type``."""

    name: str
    source_type: SourceType


# Curated allow-list of news outlets. Match is on the registrable domain (``www.``
# stripped, subdomains accepted via suffix match). ``source_type`` drives the
# credibility weight w (see ``SOURCE_TYPE_CREDIBILITY`` in src/translator.py).
# One line per outlet — extend freely.
NEWS_SITES: Dict[str, SiteInfo] = {
    "reuters.com": SiteInfo("Reuters", "wire_service"),
    "apnews.com": SiteInfo("Associated Press", "wire_service"),
    "afp.com": SiteInfo("AFP", "wire_service"),
    "bbc.com": SiteInfo("BBC News", "commercial_press"),
    "bbc.co.uk": SiteInfo("BBC News", "commercial_press"),
    "theguardian.com": SiteInfo("The Guardian", "commercial_press"),
    "nytimes.com": SiteInfo("The New York Times", "commercial_press"),
    "wsj.com": SiteInfo("The Wall Street Journal", "commercial_press"),
    "ft.com": SiteInfo("Financial Times", "commercial_press"),
    "washingtonpost.com": SiteInfo("The Washington Post", "commercial_press"),
    "cnn.com": SiteInfo("CNN", "commercial_press"),
    "bloomberg.com": SiteInfo("Bloomberg", "commercial_press"),
    "economist.com": SiteInfo("The Economist", "commercial_press"),
    "aljazeera.com": SiteInfo("Al Jazeera", "commercial_press"),
    "timesofisrael.com": SiteInfo("The Times of Israel", "commercial_press"),
    "al-monitor.com": SiteInfo("Al-Monitor", "commercial_press"),
    "middleeasteye.net": SiteInfo("Middle East Eye", "commercial_press"),
    "thenationalnews.com": SiteInfo("The National", "commercial_press"),
    "lloydslist.com": SiteInfo("Lloyd's List", "commercial_press"),
    "tradewindsnews.com": SiteInfo("TradeWinds", "commercial_press"),
    "gcaptain.com": SiteInfo("gCaptain", "commercial_press"),
    "presstv.ir": SiteInfo("Press TV", "state_media"),
    "irna.ir": SiteInfo("IRNA", "state_media"),
    "tasnimnews.com": SiteInfo("Tasnim News", "state_media"),
    "rt.com": SiteInfo("RT", "state_media"),
    "xinhuanet.com": SiteInfo("Xinhua", "state_media"),
}

_SUPPORTED_HINT = "Reuters, AP, BBC, The Guardian, Al Jazeera, Bloomberg, …"
_MAX_BODY_CHARS = 12_000
_MAX_HTML_BYTES = 5_000_000

# A bare host like "reuters.com/world/x" or "www.bbc.co.uk/news": label(.label)+
# optionally followed by a path/query, and (checked separately) no whitespace.
_BARE_HOST_RE = re.compile(r"^(?:[\w-]+\.)+[a-z]{2,}(?:[/?#]\S*)?$", re.IGNORECASE)

_ARTICLE_FIXTURES_DIR = Path(
    os.environ.get(
        "INGEST_FAKE_DIR",
        str(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "articles"),
    )
)


# --- input classification --------------------------------------------------

def classify_input(text: str) -> str:
    """``"url"`` or ``"title"``. URLs have no internal whitespace; a title
    (headline) typically does, and never looks like a bare host."""
    t = (text or "").strip()
    if not t or any(ch.isspace() for ch in t):
        return "title"
    if t.lower().startswith(("http://", "https://")):
        return "url"
    if _BARE_HOST_RE.match(t):
        return "url"
    return "title"


def _ensure_scheme(url: str) -> str:
    u = url.strip()
    return u if u.lower().startswith(("http://", "https://")) else "https://" + u


def normalize_host(url: str) -> str:
    """Lower-cased registrable host: userinfo/port/``www.`` stripped."""
    host = urlparse(_ensure_scheme(url)).netloc.lower()
    if "@" in host:
        host = host.split("@", 1)[1]
    host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def identify_site(url: str) -> Optional[SiteInfo]:
    """The registry entry for a URL's outlet, or ``None`` if not allow-listed."""
    host = normalize_host(url)
    if host in NEWS_SITES:
        return NEWS_SITES[host]
    for domain, info in NEWS_SITES.items():       # subdomain: edition.cnn.com
        if host.endswith("." + domain):
            return info
    return None


# --- fetch + extract -------------------------------------------------------

def fetch_html(url: str, *, timeout: float = 8.0,
               fetcher: Optional[Fetcher] = None) -> str:
    """Fetch a page's HTML. ``fetcher`` (tests / offline) overrides live HTTP."""
    if fetcher is not None:
        return fetcher(url)
    import requests  # lazy: keep the network dep off the pure path
    resp = requests.get(
        _ensure_scheme(url), timeout=timeout, stream=True,
        headers={
            # TODO(pack-separation): "Hormuz" in the bot UA is scenario-specific —
            # make it generic (e.g. ScenarioBNBot) or pack-derived.
            "User-Agent": "Mozilla/5.0 (compatible; HormuzScenarioBot/1.0)",
            # Identity encoding: keep Content-Length and the read cap below in
            # terms of *actual* HTML bytes, so a compressed "zip bomb" can't
            # decompress past the cap in memory.
            "Accept-Encoding": "identity",
        },
    )
    resp.raise_for_status()
    # Cap memory *while* downloading: a non-stream `.text[:cap]` buffers the whole
    # body first, so a giant/hostile response would be fully read before trimming.
    # Reject early on a declared oversize, then read at most _MAX_HTML_BYTES.
    declared = resp.headers.get("Content-Length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_HTML_BYTES:
        resp.close()
        raise ValueError(f"page too large ({declared} bytes > {_MAX_HTML_BYTES})")
    with resp:
        raw = resp.raw.read(_MAX_HTML_BYTES + 1, decode_content=True)
    if len(raw) > _MAX_HTML_BYTES:
        raise ValueError(f"page exceeds {_MAX_HTML_BYTES} bytes")
    # `resp.encoding` comes from the Content-Type header; the stream is already
    # consumed so we can't sniff the body. BeautifulSoup re-detects from the
    # <meta charset> during parsing, so a header-less utf-8 default is fine.
    return raw.decode(resp.encoding or "utf-8", errors="replace")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _iter_jsonld(data) -> Iterator[dict]:
    if isinstance(data, list):
        for item in data:
            yield from _iter_jsonld(item)
    elif isinstance(data, dict):
        yield data
        if isinstance(data.get("@graph"), list):
            yield from _iter_jsonld(data["@graph"])


def _jsonld_body(soup) -> str:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        for obj in _iter_jsonld(data):
            if obj.get("articleBody"):
                return _clean(str(obj["articleBody"]))
    return ""


def extract_article(html: str, url: str = "") -> Tuple[str, str]:
    """Best-effort (headline, body) from raw HTML.

    headline: ``og:title`` → ``<title>`` → first ``<h1>``.
    body: JSON-LD ``articleBody`` → ``<article>`` paragraphs → all ``<p>``.
    """
    from bs4 import BeautifulSoup  # lazy
    soup = BeautifulSoup(html, "html.parser")

    headline = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        headline = _clean(og["content"])
    if not headline and soup.title and soup.title.string:
        headline = _clean(soup.title.string)
    if not headline:
        h1 = soup.find("h1")
        if h1:
            headline = _clean(h1.get_text())

    body = _jsonld_body(soup)
    if not body:
        article = soup.find("article")
        if article:
            body = _clean(" ".join(p.get_text(" ") for p in article.find_all("p")))
    if not body:
        body = _clean(" ".join(p.get_text(" ") for p in soup.find_all("p")))

    return headline, body[:_MAX_BODY_CHARS]


def fake_fetcher(url: str) -> str:
    """Offline fetcher: load saved HTML for a URL's host from the fixtures dir
    (filename stem must appear in the host, e.g. ``reuters.html`` ↔ reuters.com).
    Used in fake/offline mode and tests so no network is needed."""
    host = normalize_host(url)
    if not _ARTICLE_FIXTURES_DIR.is_dir():
        raise FileNotFoundError(f"no article-fixtures dir: {_ARTICLE_FIXTURES_DIR}")
    for path in sorted(_ARTICLE_FIXTURES_DIR.glob("*.html")):
        if path.stem.lower() in host:
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no article fixture for host {host!r}")


# --- orchestrator ----------------------------------------------------------

@dataclass
class IngestResult:
    """Resolved input, ready to build an :class:`~src.translator.Article` (or a
    rejection when ``ok`` is False)."""

    kind: str                      # "url" | "title" | "manual"
    ok: bool
    message: str = ""              # rejection / info text (shown to the user)
    headline: str = ""
    body: str = ""
    source: str = ""               # outlet name (url) or "" (title/manual)
    source_type: SourceType = "unknown"
    url: str = ""
    site_name: str = ""


def ingest(top_text: str, body_text: str = "", *,
           fetcher: Optional[Fetcher] = None, timeout: float = 8.0) -> IngestResult:
    """Resolve the analyst's top input (+ optional body) into an IngestResult."""
    top = (top_text or "").strip()
    body = (body_text or "").strip()

    # A body in the expander forces the headline interpretation (manual path).
    if body:
        return IngestResult(kind="manual", ok=True, headline=top, body=body)

    if classify_input(top) == "url":
        site = identify_site(top)
        if site is None:
            host = normalize_host(top) or top
            return IngestResult(
                kind="url", ok=False, url=top,
                message=(
                    f"“{host}” isn’t a recognised news outlet, so the article "
                    f"can’t be fetched. Paste the headline directly, or use a link "
                    f"from a supported outlet ({_SUPPORTED_HINT})."
                ),
            )
        try:
            html = fetch_html(top, timeout=timeout, fetcher=fetcher)
            headline, art_body = extract_article(html, top)
        except Exception as exc:        # network / parse failure -> graceful reject
            return IngestResult(
                kind="url", ok=False, url=top, site_name=site.name,
                message=(
                    f"Couldn’t read the article from {site.name} "
                    f"({exc.__class__.__name__}). Paste the headline directly instead."
                ),
            )
        if not headline:
            return IngestResult(
                kind="url", ok=False, url=top, site_name=site.name,
                message=(
                    f"Fetched {site.name} but couldn’t find an article headline. "
                    f"Paste the headline directly instead."
                ),
            )
        return IngestResult(
            kind="url", ok=True, headline=headline, body=art_body,
            source=site.name, source_type=site.source_type, url=top,
            site_name=site.name,
        )

    return IngestResult(kind="title", ok=True, headline=top)
