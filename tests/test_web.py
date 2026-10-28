"""Web pages (Milestone 17): polite fetching, main content, images/media, site crawl, web path.
No network: every request goes to an httpx MockTransport."""

import gzip
import json

import httpx
import pytest

from thesiskb.ingest import web

FEAR = "<p>Fear of judgment, of not being understood, or of what therapy might bring up.</p>"
ARTICLE = " ".join(["Therapy can feel overwhelming at the start for many people."] * 8)
PAGE = f"""<!doctype html><html><head><title>SideBy | Portfolio</title></head><body>
<header><nav><a href="/">Home</a><a href="/about">About</a><img src="/logo.png"></nav></header>
<main>
  <h1>SideBy</h1>
  <h2>Problem</h2><p>{ARTICLE}</p>
  <h2>Our Approach</h2>
  <div class="carousel">
    <div><h3>Fear is a major barrier</h3>{FEAR}</div>
    <div><h3>Fear is a major barrier</h3>{FEAR}</div>
  </div>
  <figure><img src="/images/sideby/cover.png"></figure>
  <img src="https://cdn.example.org/shot.jpg">
  <video src="/videos/sideby/demo.mp4"></video>
  <video src="https://youtube.example/v.mp4"></video>
  <section><h2>More projects</h2>
    <a href="/projects/other"><img src="/images/other/card.png"><p>Other</p></a></section>
</main>
<audio src="/sounds/swish.mp3" style="display: none;"></audio>
<footer><p>Impressum · Privacy · © Vivien Cai</p></footer>
</body></html>"""
SITEMAP = """<?xml version="1.0"?><urlset>
<url><loc>https://site.test/</loc></url><url><loc>https://site.test/projects/sideby</loc></url>
<url><loc>https://other.test/x</loc></url></urlset>"""


def transport(routes: dict, log: list | None = None, robots: str = "User-agent: *\nAllow: /\n"):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if log is not None:
            log.append(url)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text=robots)
        if url in routes:
            body, ctype = routes[url]
            if isinstance(body, int):
                return httpx.Response(body)
            return httpx.Response(200, content=body if isinstance(body, bytes) else body.encode(),
                                  headers={"Content-Type": ctype})  # fmt: skip
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def fetcher(routes, log=None, robots="User-agent: *\nAllow: /\n") -> web.Fetcher:
    client = httpx.Client(transport=transport(routes, log, robots), follow_redirects=True)
    return web.Fetcher(delay=0, client=client)


def test_robots_txt_is_obeyed():
    f = fetcher({"https://site.test/private/x": (PAGE, "text/html")},
                robots="User-agent: *\nDisallow: /private/\n")  # fmt: skip
    with pytest.raises(web.WebError, match="robots.txt"):
        f.get("https://site.test/private/x")
    assert f.allowed("https://site.test/public")


def test_backs_off_on_429(monkeypatch):
    calls = []
    slept = []
    monkeypatch.setattr(web.time, "sleep", slept.append)

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, text="ok")

    f = web.Fetcher(delay=0, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert f.get("https://site.test/a").text == "ok"
    assert len(calls) == 3 and slept.count(7.0) == 2


def test_main_content_keeps_headings_drops_navigation_footer_cards_and_repeats():
    doc, chars = web.main_html(PAGE, "https://site.test/projects/sideby", "SideBy")
    assert chars > web.MIN_TEXT
    assert "<h2>Problem</h2>" in doc and "<h3>Fear is a major barrier</h3>" in doc
    assert doc.count("Fear of judgment") == 1  # carousel clone dropped
    for clutter in ("Impressum", "About", "More projects"):
        assert clutter not in doc, clutter
    assert 'name="source-url" content="https://site.test/projects/sideby"' in doc


def test_page_parts_takes_the_pages_own_images_and_same_site_media():
    p = web.page_parts(PAGE, "https://site.test/projects/sideby")
    assert p.images == [
        "https://site.test/images/sideby/cover.png",
        "https://cdn.example.org/shot.jpg",
    ]
    assert p.media == ["https://site.test/videos/sideby/demo.mp4"]  # no other host, no UI sound
    assert "https://site.test/about" in p.links and "https://site.test/projects/other" in p.links


def test_snapshot_stores_html_main_text_media_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "render", lambda url: pytest.fail("no JavaScript needed here"))
    big = b"\x89PNG" + b"0" * 20_000
    f = fetcher({
        "https://site.test/projects/sideby": (PAGE, "text/html; charset=utf-8"),
        "https://site.test/images/sideby/cover.png": (big, "image/png"),
        "https://cdn.example.org/shot.jpg": (b"tiny", "image/jpeg"),  # < 10 kB: an icon
        "https://site.test/videos/sideby/demo.mp4": (b"\x00" * 1000, "video/mp4"),
    })  # fmt: skip
    s = web.snapshot_page(f, "https://site.test/projects/sideby", tmp_path)
    assert s.folder == tmp_path / "site.test" / "projects-sideby"
    assert [p.name for p in s.files] == ["main.html", "img-01.png", "demo.mp4"]
    assert s.origins[str(s.folder / "media" / "demo.mp4")].endswith("/videos/sideby/demo.mp4")
    meta = json.loads((s.folder / "meta.json").read_text("utf-8"))
    assert meta["url"] == "https://site.test/projects/sideby" and meta["rendered"] is False
    with gzip.open(s.folder / "snapshot.html.gz", "rt") as fh:
        assert "Impressum" in fh.read()  # the full page as fetched


def test_javascript_page_is_rendered(tmp_path, monkeypatch):
    shell = "<html><head><title>App</title></head><body><div id=app></div></body></html>"
    monkeypatch.setattr(web, "render", lambda url: PAGE)
    f = fetcher({"https://site.test/app": (shell, "text/html")})
    s = web.snapshot_page(f, "https://site.test/app", tmp_path, images=False, media=False)
    assert s.rendered and "Therapy can feel overwhelming" in s.main.read_text("utf-8")


def test_site_crawl_uses_the_sitemap_same_host_only_and_stops_at_max_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "render", lambda url: pytest.fail("not needed"))
    seen = []
    routes = {
        "https://site.test/sitemap.xml": (SITEMAP, "application/xml"),
        "https://site.test/": (PAGE, "text/html"),
        "https://site.test/projects/sideby": (PAGE, "text/html"),
        "https://site.test/about": (PAGE, "text/html"),
        "https://site.test/projects/other": (PAGE, "text/html"),
    }
    snaps, failed = web.crawl(
        ["https://site.test/"], tmp_path, site=True, max_pages=3, fetcher=fetcher(routes, seen),
        images=False, media=False,
    )  # fmt: skip
    assert [s.url for s in snaps] == [
        "https://site.test/", "https://site.test/projects/sideby", "https://site.test/about"
    ]  # fmt: skip
    assert not any("other.test" in u for u in seen)  # never another host
    assert failed == []


def test_web_route_only_when_asked_and_query_without_the_request():
    from thesiskb.answer.web import search_query
    from thesiskb.retrieval.router import WEB_WORDS

    assert WEB_WORDS.search("Search the web for SideBy mental health")
    assert WEB_WORDS.search("Such im Internet nach dem DGX Spark Preis")
    assert not WEB_WORDS.search("Which online tools did the team use?")
    assert not WEB_WORDS.search("What does the survey say about web data?")
    assert search_query("Look up Vivien Cai online") == "Vivien Cai"
    assert search_query("Search the web for the pyannote licence") == "the pyannote licence"


def test_web_search_says_how_to_start_searxng(monkeypatch):
    from thesiskb.answer import web as answer_web

    def down(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(answer_web.httpx, "get", down)
    with pytest.raises(answer_web.WebSearchError, match="docker compose --profile web"):
        answer_web.web_search("x")
