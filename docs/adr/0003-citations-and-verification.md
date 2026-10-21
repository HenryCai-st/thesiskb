# ADR 0003 — Citation labels, per-claim support check, test database

Date: 2026-10-21 · Status: accepted

## Passage labels `[S1]..[Sn]` instead of chunk UUIDs or plain numbers
The LLM sees reranked chunks as `[S1]`…`[S8]` and cites those; `generate.py` maps them back to
`[chunk:<uuid>]` (`Answer.text_with_ids`). Small models copy short labels far more reliably than
UUIDs. Plain `[1]` clashed with papers' own bibliography numbers (`[377]`), which were then
flagged as invalid citations.

## One judge call per claim
Judging all claims in one call against all passages (≈4k tokens) made qwen3:14b report names as
missing that were present (support rate 0.62 on lit-02). Checking each claim against only its
cited passages raised it to 0.86. Costs ~1–2 s per claim. The judge is its own role in
`models.yaml` (same weights as `answer`, thinking off — with thinking on it spent the whole token
budget and returned no JSON).

## Reranker in fp16
With the answer LLM resident (~12.9 GB), the fp32 reranker (2.3 GB + activations) exceeded 16 GB
and Windows silently spilled into shared system memory. fp16 halves it to ~1.1 GB.
Remaining CLI rerank latency (~18 s) is process start-up (importing sentence-transformers ~11 s,
loading weights ~6 s); scoring itself is ~1 s. The API server (Milestone 7) loads it once.

## Tests use `<db>_test`
The migration test downgrades to base. It once ran against the real KB and wiped it, so DB tests
now derive a `*_test` database from `TKB_DATABASE_URL` and refuse any other name. The parse stage
also treats `data/processed/<sha>/docling.json` as a cache, so re-ingesting after a DB reset needs
no GPU work.
