# ADR 0011 — EPUB and HTML: section locators instead of pages

Date: 2026-10-27 · Status: accepted

Context: EPUB ingest "failed" for users (BUILD_GUIDE_v2 M11). Docling 2.x converts EPUB and
HTML without extra configuration, but its items for these formats carry no page provenance,
and `extract_blocks()` skipped every item without one → 0 blocks → 0 chunks.

## Decisions
- **Locator for page-less documents**: `{"part": "<section path>", "order": n}` per block;
  chunks merge to `{"part", "order", "order_end"}`. Citations show the section ("Chapter 2",
  "Problem") instead of a page. PDFs are unchanged (same code path when the document has pages).
- **Title/authors** from the EPUB package (`dc:title`, `dc:creator`) or the HTML `<title>`;
  headings in page-less documents are always sections (EPUB chapters use `<h1>`, which Docling
  labels as a title).
- EPUB navigation pages ("Table of Contents") are skipped.
- **Images**: Docling exposes `<img>` as picture items with their alt text as caption but no
  image data; each becomes an `Image: <caption>` block. No vision model runs on web/EPUB images
  (not needed for the gold questions; revisit if captions prove too thin).
- **Videos** in saved pages point at the live site (`/videos/...mp4`) and are not in the saved
  copy; they are left for URL ingest (M17).
- **Saved pages**: files inside `<name>_files/` next to `<name>.html` are page assets, not
  separate sources (otherwise every screenshot was ingested as a photo).
- **Viewer**: `/view/{source_id}` serves Docling's HTML export of the parsed document
  (cached as `view.html`, no scripts). Citation links add a URL text fragment
  (`#:~:text=<first words>`), so Chromium-based browsers and Firefox scroll to and highlight
  the passage. Chosen over mapping blocks back to EPUB XHTML files/anchors, which Docling does
  not record.
- Default source type: EPUB → `book` (literature path), HTML → `web`.
- The ebooklib fallback from the guide was not needed: both test files convert with Docling.
