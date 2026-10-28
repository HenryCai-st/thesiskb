# ADR 0018 — Web pages as sources; web search with self-hosted SearXNG

Date: 2026-10-28 · Status: accepted

Context: users want web pages and whole sites in the KB, and web search when they ask for it
(feedback points 5 and 7, BUILD_GUIDE_v2 Milestone 17). The legal check is answered
(section 4: private use outside the EU); the technical safeguards stay.

## Part A — URL and site ingest
- **Snapshot, then the normal pipeline.** `tkb ingest <url>` / "Add URL" fetch once into
  `data/raw/web/<host>/<page>/`: `snapshot.html.gz` (the HTML used), `meta.json` (URL, final
  URL, retrieval time, rendered or not), `main.html` (main content only) and `media/`. The
  pipeline ingests `main.html` like any HTML file (section locators; `/view` + text fragment
  opens the saved copy at the cited passage) and the media like any image/video. The source's
  `origin` is the URL and `retrieved_at` the time (migration 0007); citations also link to the
  live page with the same text fragment. A re-fetch of a changed page replaces the old version
  (same path, M13).
- **Main content: trafilatura** (new dependency, Apache-2.0) on HTML from which navigation,
  header, footer, aside and link cards (links to other pages wrapping an image: "more
  projects") were removed. Its paragraph output loses headings, so they are restored from the
  page's `<h1>`–`<h4>`; repeated paragraphs (carousel clones) are dropped; headings left empty
  are dropped. trafilatura's own `deduplicate` is off: its cache is process-wide and blanked
  the second page of a crawl.
- **JavaScript: Playwright directly, not Crawl4AI.** A page whose main text is under 300
  characters is rendered in headless Chromium (wait until the DOM stops growing, scroll for
  lazy content). The reference page (SvelteKit) needs this. Crawl4AI is built on Playwright;
  we only need "render, return HTML", without its crawler and LLM stack. Optional extra:
  `uv sync --extra web` + `uv run playwright install chromium`; without it such pages fail
  with that instruction.
- **Media**: the page's own images (inside `<main>` if present; not in navigation, cards or
  hidden elements; ≥ 10 kB; at most 20) and direct same-site video/audio files (≤ 500 MB). No
  platform downloaders (YouTube etc.).
- **Politeness**: robots.txt via `urllib.robotparser` (unreachable → allowed, 401/403 → all
  disallowed), User-Agent `ThesisKB/<version> (private research knowledge base; +repo URL)`,
  one request at a time per host with ≥ 1 s delay (robots Crawl-delay if larger), back-off on
  429/503 honouring Retry-After (3 retries). Rendering loads the page's own scripts once.
- **Whole site** (`--site`, "whole site" box): same host only; robots.txt `Sitemap:` or
  `/sitemap.xml` first (sitemap indexes followed on the same host), then links found on
  fetched pages; at most `--max-pages` (default 50). Each page is its own source.
- In the web UI a URL is a queue item (`kind = url`), so fetching and ingesting run one at a
  time with everything else.

## Part B — web search
- **Backend: SearXNG, self-hosted** (user's choice over Brave/Tavily): no account or key, and
  queries are not tied to the user by a search provider. A second container behind the compose
  profile `web`, bound to 127.0.0.1:8888, JSON format enabled, limiter off (one local user).
  Upstream engines can rate-limit (seen: Brave, DuckDuckGo CAPTCHA); the others still answer.
- **Only when asked**: the router sends a question to the `web` path only if it asks
  explicitly ("search the web for …", "look up … online", "im Internet suchen") or the user
  picks the path; the LLM router never chooses it. The query is the question without those
  words.
- **Results become sources**: the top 4 results that robots.txt allows are fetched and
  ingested (Part A) as `source_type = web`, `unverified`, without extraction, summaries,
  images or video, so an answer takes about a minute; then hybrid search + rerank runs over
  those pages only and the answer cites them like any other source. If an ingest is running,
  the question is refused with a message instead of waiting (the queue worker may itself be
  waiting for the question to finish).
- **People**: if the question names a known person, the fetched pages get `person_id`;
  `tkb forget-person` deletes those sources, their processed files and the snapshots.
