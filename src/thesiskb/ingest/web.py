"""Web pages as sources (Milestone 17, ADR 0018): polite fetching, snapshots, whole-site crawl.

  fetch      one request at a time per host, robots.txt obeyed (urllib.robotparser), a
             descriptive User-Agent, a delay between requests (robots Crawl-delay if larger),
             back-off on 429/503 (Retry-After)
  render     pages whose HTML has almost no text (JavaScript apps, e.g. SvelteKit) are
             rendered with headless Chromium (Playwright, `uv sync --extra web`)
  snapshot   data/raw/web/<host>/<page>/: snapshot.html.gz (the HTML used), meta.json (URL,
             final URL, retrieval time, rendered?), main.html = the main content only
             (trafilatura: no navigation, footer or "more projects" cards; repeated carousel
             paragraphs dropped), media/ = the page's own images and same-site video/audio
             files (direct links only, no platform downloaders)
  site       `--site`: same host only; sitemap.xml (or robots.txt's Sitemap:) first, else the
             links of each fetched page; at most `max_pages`

main.html and the media files are ordinary inputs for the pipeline (HTML → section locators,
images → vision, video → transcript + keyframes); `origin` = the URL, `retrieved_at` = time.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import BaseModel, Field

from thesiskb import __version__

log = logging.getLogger(__name__)
logging.getLogger("trafilatura").setLevel(logging.ERROR)  # "discarding data" on every JS shell

USER_AGENT = (
    f"ThesisKB/{__version__} (private research knowledge base; "
    "+https://github.com/HenryCai-st/thesiskb)"
)
ROBOTS_AGENT = "ThesisKB"
DELAY_S = 1.0
RETRIES = 3
MIN_TEXT = 300  # characters of main text below which a page is rendered with JavaScript
MAX_IMAGES = 20
MIN_IMAGE_BYTES = 10_000  # icons, spacers
MAX_MEDIA_BYTES = 500_000_000
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_EXT = {".mp4", ".webm", ".mov", ".m4v", ".mp3", ".m4a", ".wav", ".ogg"}
SKIP_LINK_EXT = IMAGE_EXT | MEDIA_EXT | {".pdf", ".zip", ".gif", ".svg", ".css", ".js", ".xml"}


class WebError(RuntimeError):
    """A page that cannot or may not be fetched (robots.txt, HTTP error, no readable text)."""


class Fetcher:
    """Polite HTTP: robots.txt, one request at a time per host with a delay, back-off."""

    def __init__(self, delay: float = DELAY_S, client: httpx.Client | None = None) -> None:
        self.delay = delay
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30
        )
        self._robots: dict[str, RobotFileParser] = {}
        self._last: dict[str, float] = {}

    def robots(self, url: str) -> RobotFileParser:
        u = urlparse(url)
        base = f"{u.scheme}://{u.netloc}"
        if base not in self._robots:
            rp = RobotFileParser(base + "/robots.txt")
            try:
                r = self._request("GET", base + "/robots.txt", check=False)
                if r.status_code in (401, 403):
                    rp.disallow_all = True
                elif r.status_code >= 400:
                    rp.allow_all = True
                else:
                    rp.parse(r.text.splitlines())
            except httpx.HTTPError:
                rp.allow_all = True  # no robots.txt reachable: the site set no rules
            self._robots[base] = rp
        return self._robots[base]

    def allowed(self, url: str) -> bool:
        return self.robots(url).can_fetch(ROBOTS_AGENT, url)

    def _wait(self, url: str) -> None:
        host = urlparse(url).netloc
        rp = self._robots.get(f"{urlparse(url).scheme}://{host}")
        delay = max(self.delay, float((rp and rp.crawl_delay(ROBOTS_AGENT)) or 0))
        wait = self._last.get(host, 0) + delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def _request(self, method: str, url: str, check: bool = True, **kw) -> httpx.Response:
        if check and not self.allowed(url):
            raise WebError(f"robots.txt does not allow fetching {url}")
        for attempt in range(RETRIES + 1):
            self._wait(url)
            r = self.client.request(method, url, **kw)
            if r.status_code not in (429, 503) or attempt == RETRIES:
                return r
            retry = r.headers.get("Retry-After", "")
            pause = float(retry) if retry.isdigit() else 2.0 * 2**attempt
            log.info("%s answered %d: waiting %.0f s", urlparse(url).netloc, r.status_code, pause)
            time.sleep(min(pause, 120))
        return r

    def get(self, url: str) -> httpx.Response:
        r = self._request("GET", url)
        if r.status_code >= 400:
            raise WebError(f"{url}: HTTP {r.status_code}")
        return r

    def download(self, url: str, dest: Path, max_bytes: int = MAX_MEDIA_BYTES) -> bool:
        """Stream a file to dest; False (and nothing written) if not allowed, too large or
        unreachable after one retry: a missing image must not fail its page."""
        if not self.allowed(url):
            return False
        for attempt in (1, 2):
            try:
                return self._download(url, dest, max_bytes)
            except httpx.HTTPError as e:
                dest.unlink(missing_ok=True)
                log.warning("download %s failed (%s, attempt %d)", url, type(e).__name__, attempt)
        return False

    def _download(self, url: str, dest: Path, max_bytes: int) -> bool:
        self._wait(url)
        with self.client.stream("GET", url) as r:
            if r.status_code >= 400:
                return False
            size = int(r.headers.get("Content-Length") or 0)
            if size > max_bytes:
                log.info("skipping %s: %d MB", url, size // 1_000_000)
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            done = 0
            with dest.open("wb") as fh:
                for block in r.iter_bytes(1 << 20):
                    done += len(block)
                    if done > max_bytes:
                        break
                    fh.write(block)
        if done > max_bytes:
            dest.unlink()
            return False
        return True


def render(url: str) -> str:
    """The page's HTML after JavaScript ran (headless Chromium), lazy content scrolled in."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise WebError(
            f"{url} needs JavaScript to show its text: install the web extra "
            "(`uv sync --extra web` and `uv run playwright install chromium`)"
        ) from e
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=60_000)
            size = -1
            for _ in range(10):  # until the DOM stops growing (client-side rendering)
                page.wait_for_timeout(1000)
                html = page.content()
                if len(html) == size:
                    break
                size = len(html)
            page.mouse.wheel(0, 50_000)  # lazy-loaded images and sections
            page.wait_for_timeout(1500)
            return page.content()
        finally:
            browser.close()


# --- main content ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main_html(html: str, url: str, title: str) -> tuple[str, int]:
    """Clean HTML of the main content (headings, paragraphs, lists) and its text length."""
    import trafilatura
    from lxml import etree

    # no deduplicate=True: its cache is global, so a second page (or a re-fetch) lost its text
    xml = trafilatura.extract(
        _without_clutter(html, url), url=url, output_format="xml", include_tables=True,
        include_images=False, include_links=False, include_comments=False,
    )  # fmt: skip
    if not xml:
        return "", 0
    root = etree.fromstring(xml.encode())
    body = root.find("main") if root.find("main") is not None else root
    # trafilatura often returns headings as plain paragraphs: restore them from the page
    headings = _page_headings(html)
    seen: set[str] = set()
    out: list[str] = []
    chars = 0

    def text_of(el) -> str:
        return _norm("".join(el.itertext()))

    def emit(tag: str, text: str) -> None:
        nonlocal chars
        key = text.lower()
        if not text or (len(text) >= 20 and key in seen):  # carousel clones, repeated cards
            return
        seen.add(key)
        chars += len(text)
        out.append(f"<{tag}>{escape(text)}</{tag}>")

    for el in body:
        if el.tag == "head":
            level = {"h1": 2, "h2": 2, "h3": 3, "h4": 4}.get(el.get("rend", "h2"), 3)
            emit(f"h{level}", text_of(el))
        elif el.tag == "list":
            items = [text_of(i) for i in el.iter("item")]
            items = [i for i in items if i and i.lower() not in seen]
            if items:
                seen.update(i.lower() for i in items)
                chars += sum(map(len, items))
                out.append("<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>")
        elif el.tag == "quote":
            emit("blockquote", text_of(el))
        else:  # p, table rows, anything else with text; a page heading's text → heading
            text = text_of(el)
            level = headings.get(text.lower())
            emit(f"h{max(level, 2)}" if level else "p", text)
    out = _drop_empty_headings(out)
    doc = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        f'<meta name="source-url" content="{escape(url)}"></head>\n<body>\n'
        f"<h1>{escape(title)}</h1>\n" + "\n".join(out) + "\n</body></html>\n"
    )
    return doc, chars


def _without_clutter(html: str, url: str) -> str:
    """Remove navigation, header, footer, aside and link cards (a link to another page that
    wraps an image, e.g. "more projects") before main-content extraction."""
    import lxml.html

    doc = lxml.html.fromstring(html)
    page = urldefrag(url)[0]
    for el in list(doc.iter("nav", "header", "footer", "aside")):
        el.drop_tree()
    for a in list(doc.iter("a")):
        target = urldefrag(urljoin(page, a.get("href") or ""))[0]
        if target != page and a.find(".//img") is not None and a.getparent() is not None:
            a.drop_tree()
    return lxml.html.tostring(doc, encoding="unicode")


def _drop_empty_headings(blocks: list[str]) -> list[str]:
    """Drop a heading with no text under it: the next block is a heading of the same or a
    higher rank, or there is none (e.g. "More projects" once its cards are gone)."""

    def rank(block: str) -> int | None:
        m = re.match(r"<h([1-6])>", block)
        return int(m[1]) if m else None

    keep: list[str] = []
    for i, block in enumerate(blocks):
        r = rank(block)
        nxt = rank(blocks[i + 1]) if i + 1 < len(blocks) else 0
        if r is not None and (i + 1 == len(blocks) or (nxt is not None and nxt <= r)):
            continue
        keep.append(block)
    return keep


def _page_headings(html: str) -> dict[str, int]:
    import lxml.html

    out: dict[str, int] = {}
    for el in lxml.html.fromstring(html).iter("h1", "h2", "h3", "h4"):
        text = _norm(el.text_content()).lower()
        if text:
            out.setdefault(text, int(el.tag[1]))
    return out


def page_title(html: str, url: str) -> str:
    import trafilatura

    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else "") or urlparse(url).path.strip("/")
    return _norm(title) or urlparse(url).netloc


# --- links, images, media -------------------------------------------------------------------


def _same_site(a: str, b: str) -> bool:
    ha, hb = urlparse(a).netloc.lower(), urlparse(b).netloc.lower()
    base = hb.removeprefix("www.")
    return ha in (hb, base) or ha.endswith("." + base)


def _ext(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


class PageParts(BaseModel):
    links: list[str] = Field(default_factory=list)  # same host, html-looking
    images: list[str] = Field(default_factory=list)
    media: list[str] = Field(default_factory=list)


def page_parts(html: str, url: str) -> PageParts:
    """Links to follow, the page's own images (inside <main> if the page has one; not in
    navigation/header/footer, not cards linking to other pages, not hidden) and direct
    same-site video/audio files."""
    import lxml.html

    doc = lxml.html.fromstring(html)
    mains = doc.xpath("//main")
    content = mains[0] if mains else doc
    page = urldefrag(url)[0]
    host = urlparse(url).netloc
    parts = PageParts()

    def boxed(el) -> bool:
        hidden = re.compile(r"display\s*:\s*none")
        if el.get("hidden") is not None or hidden.search(el.get("style") or ""):
            return True  # e.g. a site's UI sound effect
        for a in el.iterancestors():
            if a.tag in ("nav", "header", "footer", "aside"):
                return True
            if a.tag == "a" and a.get("href"):
                target = urldefrag(urljoin(page, a.get("href")))[0]
                if target != page:
                    return True  # a card linking somewhere else
        return False

    for a in doc.iter("a"):
        href = urldefrag(urljoin(page, a.get("href") or ""))[0]
        if (
            urlparse(href).scheme in ("http", "https")
            and urlparse(href).netloc == host
            and _ext(href) not in SKIP_LINK_EXT
            and href not in parts.links
        ):
            parts.links.append(href)
        elif (
            _ext(href) in MEDIA_EXT
            and _same_site(href, url)
            and href not in parts.media
            and not boxed(a)
            and (content is doc or content in a.iterancestors())
        ):
            parts.media.append(href)
    for el in content.iter("video", "audio", "source"):
        src = el.get("src")
        if src and not boxed(el if el.tag != "source" else el.getparent()):
            full = urljoin(page, src)
            if _ext(full) in MEDIA_EXT and _same_site(full, url) and full not in parts.media:
                parts.media.append(full)
    for img in content.iter("img"):
        src = img.get("src") or ""
        if not src or src.startswith("data:") or boxed(img):
            continue
        full = urljoin(page, src)
        if _ext(full) in IMAGE_EXT and full not in parts.images:
            parts.images.append(full)
    return parts


# --- snapshots ------------------------------------------------------------------------------


class Snapshot(BaseModel):
    url: str
    final_url: str
    title: str
    folder: Path
    main: Path
    files: list[Path]  # main.html + media to ingest
    origins: dict[str, str]  # file path → URL it came from
    links: list[str]
    rendered: bool
    retrieved_at: datetime


def page_folder(root: Path, url: str) -> Path:
    u = urlparse(url)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (u.path.strip("/") or "index"))[:80].strip("-.")
    if u.query:
        slug += "-" + hashlib.sha1(u.query.encode()).hexdigest()[:8]
    return root / re.sub(r"[^A-Za-z0-9.-]", "_", u.netloc) / (slug or "index")


def snapshot_page(
    fetcher: Fetcher, url: str, root: Path, images: bool = True, media: bool = True
) -> Snapshot:
    r = fetcher.get(url)
    final = str(r.url)
    ctype = r.headers.get("Content-Type", "")
    if "html" not in ctype:
        raise WebError(f"{url} is not a web page ({ctype or 'unknown type'})")
    html, rendered = r.text, False
    title = page_title(html, final)
    doc, chars = main_html(html, final, title)
    if chars < MIN_TEXT:
        html, rendered = render(final), True
        title = page_title(html, final)
        doc, chars = main_html(html, final, title)
    if chars < MIN_TEXT:
        raise WebError(f"{url}: no readable main text ({chars} characters)")

    folder = page_folder(root, final)
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    with gzip.open(folder / "snapshot.html.gz", "wt", encoding="utf-8") as fh:
        fh.write(html)
    main = folder / "main.html"
    main.write_text(doc, encoding="utf-8")
    parts = page_parts(html, final)
    files, origins = [main], {str(main): final}

    if images:
        n = 0
        for src in parts.images:
            if n >= MAX_IMAGES:
                break
            dest = folder / "media" / f"img-{n + 1:02d}{_ext(src)}"
            if fetcher.download(src, dest, max_bytes=50_000_000):
                if dest.stat().st_size < MIN_IMAGE_BYTES:
                    dest.unlink()
                    continue
                files.append(dest)
                origins[str(dest)] = src
                n += 1
    if media:
        for i, src in enumerate(parts.media, 1):
            name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(urlparse(src).path).name) or f"media-{i}"
            dest = folder / "media" / name
            if fetcher.download(src, dest):
                files.append(dest)
                origins[str(dest)] = src
    meta = {
        "url": url,
        "final_url": final,
        "title": title,
        "retrieved_at": now.isoformat(),
        "rendered": rendered,
        "user_agent": USER_AGENT,
        "files": {Path(k).name: v for k, v in origins.items()},
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Snapshot(
        url=url, final_url=final, title=title, folder=folder, main=main, files=files,
        origins=origins, links=parts.links, rendered=rendered, retrieved_at=now,
    )  # fmt: skip


def sitemap_urls(fetcher: Fetcher, start: str, limit: int) -> list[str]:
    """Page URLs on the start URL's host from robots.txt Sitemap: lines or /sitemap.xml."""
    u = urlparse(start)
    base = f"{u.scheme}://{u.netloc}"
    maps = list(fetcher.robots(start).site_maps() or []) or [base + "/sitemap.xml"]
    urls: list[str] = []
    seen_maps: set[str] = set()
    while maps and len(urls) < limit:
        sm = maps.pop(0)
        if sm in seen_maps or urlparse(sm).netloc != u.netloc:
            continue
        seen_maps.add(sm)
        try:
            r = fetcher.get(sm)
        except (WebError, httpx.HTTPError):
            continue
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
        if "<sitemapindex" in r.text:
            maps += locs
            continue
        for loc in locs:
            if urlparse(loc).netloc == u.netloc and loc not in urls:
                urls.append(loc)
    return urls[:limit]


def _canonical(url: str) -> str:
    """No fragment; "https://host" is "https://host/"."""
    url = urldefrag(url)[0]
    return url + "/" if not urlparse(url).path else url


def crawl(
    urls: list[str],
    root: Path,
    site: bool = False,
    max_pages: int = 50,
    fetcher: Fetcher | None = None,
    on_page=None,
    images: bool = True,
    media: bool = True,
) -> tuple[list[Snapshot], list[tuple[str, str]]]:
    """Snapshot the URLs (and with `site`, the rest of each start URL's host). Returns the
    snapshots and (url, reason) for pages that failed or were not allowed."""
    fetcher = fetcher or Fetcher()
    urls = [_canonical(u) for u in urls]
    todo = list(urls)
    if site:
        for start in urls:
            todo += [x for x in sitemap_urls(fetcher, start, max_pages) if x not in todo]
    hosts = {urlparse(x).netloc for x in urls}
    done: set[str] = set()
    snaps: list[Snapshot] = []
    failed: list[tuple[str, str]] = []
    while todo and len(snaps) < (max_pages if site else len(urls)):
        url = _canonical(todo.pop(0))
        if url in done:
            continue
        done.add(url)
        if on_page:
            on_page(len(snaps), url)
        try:
            snap = snapshot_page(fetcher, url, root, images=images, media=media)
        except (WebError, httpx.HTTPError) as e:
            log.warning("web: %s", e)
            failed.append((url, str(e)))
            continue
        done.add(_canonical(snap.final_url))
        snaps.append(snap)
        if site:  # follow the page's links on the same host (after the sitemap's pages)
            todo += [x for x in snap.links if urlparse(x).netloc in hosts and x not in done]
    return snaps, failed


# --- into the knowledge base ----------------------------------------------------------------


def ingest_web(
    session,
    urls: list[str],
    cfg,
    gateway,
    site: bool = False,
    max_pages: int = 50,
    progress=None,
    fetcher: Fetcher | None = None,
    extract: bool = True,
    summarize: bool = True,
    images: bool = True,
    media: bool = True,
    person_id=None,
):
    """Snapshot the URLs (or sites) and ingest them. Returns (report, sources, failed)."""
    from sqlalchemy import select

    from thesiskb.core.models import Source
    from thesiskb.ingest.detect import detect
    from thesiskb.ingest.pipeline import ingest

    def on_page(i: int, url: str) -> None:
        if progress:
            progress("fetch", url, i, max_pages if site else len(urls))

    snaps, failed = crawl(
        urls, cfg.raw_dir / "web", site, max_pages, fetcher, on_page, images, media
    )

    def key(p) -> str:  # detect() resolves paths; snapshots may hold relative ones
        return str(Path(p).resolve())

    origin = {key(p): u for s in snaps for p, u in s.origins.items()}
    when = {key(p): s.retrieved_at for s in snaps for p in s.files}
    titles = {key(p): s.title for s in snaps for p in s.files}
    files = detect([p for s in snaps for p in s.files])
    report = ingest(
        session, files, cfg, gateway, source_type="web", extract=extract, progress=progress,
        summarize=summarize,
    )  # fmt: skip
    sources = []
    for f in files:
        src = session.scalar(select(Source).where(Source.sha256 == f.sha256))
        if src is None:
            continue
        k = key(f.path)
        src.origin, src.retrieved_at = origin.get(k), when.get(k)
        if f.path.name != "main.html":  # an image or video of the page
            src.title = f"{titles.get(k, '')} – {f.path.name}"  # named after its page
        if person_id is not None:
            src.person_id = person_id
        sources.append(src)
    session.commit()
    report["failed"] = list(report.get("failed", [])) + failed
    return report, sources, failed
