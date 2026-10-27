# ADR 0012 — Whole-document questions: part summaries at ingest + a summary path

Date: 2026-10-27 · Status: accepted

Context: "Summarize the book" covered chapters 1 and 3 but skipped 2 (feedback point 2). The
answer model only ever sees the top 8–12 retrieved chunks, chosen by similarity to the
question; nothing in the pipeline looked at a whole source. A bigger model would get the same
chunks.

## Decisions
- **New stage `summarize`** between `chunk` and `embed` (so summaries are embedded with the
  rest). Per source: split into parts — documents by top-level section (EPUB chapters, numbered
  paper sections), recordings by ~2,500-token time windows — summarize each part (role
  `extraction`: thinking off; parts over 5,000 tokens are summarized in pieces, then combined),
  then summarize the whole source from the part summaries. A structure-based form of
  [RAPTOR](https://huggingface.co/papers/2401.18059): the document's own headings replace
  clustering. Cached in `data/processed/<sha>/summaries.json` by text hash, so re-runs are free.
- Short sections (< 300 tokens, e.g. "Tools", "Team" on a web page) absorb the following
  section; sources under 400 tokens and images get no summary (all their chunks fit anyway).
- **Storage without a migration**: summaries are chunks whose locator has
  `"summary": "section" | "document"`, the part label, an index and the covered page / time /
  block range. Citations read "summary of 2 Methods, pp. 3–9".
- **Kept out of normal search** (`SearchFilters.summaries`), extraction and person lookups.
  The guide said "searchable"; mixing paraphrases into top-k search changed which passages
  ordinary questions cite, and quote/timestamp checks need original text. Only the summary path
  searches them.
- **Summary path**: picks the source whose summaries best match the question (hybrid search +
  rerank over summary chunks), then shows the document summary and every part summary in
  order. `merge()` never truncates it (`Ctx.keep_all`), so long books are not cut at 12
  passages. The router sends overview questions there by rule ("summarize / overview / main
  points … book / paper / meeting", "what is the paper about", German equivalents) and drops
  general/literature for them; the LLM router may also choose it.
- **Coverage**: after answering, the parts the answer did not cite are recorded in the trace
  and shown under the answer ("parts covered: 9/10; not covered: 9 CONCLUSION").
- Not changed (guide: out of scope): chunk size, top-k, the answer model.
