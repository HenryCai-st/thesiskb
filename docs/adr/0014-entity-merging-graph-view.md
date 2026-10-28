# ADR 0014 — Entity keys, automatic merging of near-duplicates, graph view

Date: 2026-10-28 · Status: accepted

Context: extraction matched entities by exact lowercase name, so spelling variants became
separate nodes ("Monte-Carlo sampling" / "Monte Carlo sampling", "GAN" / "GANs"). Users want to
see what is in the KB as a graph; they do not want to edit entities (feedback point 4,
BUILD_GUIDE_v2 Milestone 14).

## Decisions
- **Entity key** (`entities.key`, unique): NFKC, casefold, hyphens/underscores (incl. Unicode
  dashes) → spaces, collapsed whitespace. Extraction finds entities by key or by an alias's key.
  Migration 0005 merged the existing duplicates by key (2 in the thesis KB).
- **Near-duplicates**: after an ingest, entity names without a vector are embedded with the
  chunk embedding model (bge-m3, always the local role) and compared with their 8 nearest
  names (pgvector, cosine). Rule, from the 607 pairs ≥ 0.82 in the thesis KB:
  - names with different numbers never pair ("Song et al. [287]" / "[282]", 0.97);
  - same words up to plural endings and spaces ("GAN" / "GANs" 0.92, "Auto-Encoder" /
    "Autoencoders") → merge from 0.80;
  - otherwise merge from 0.95 (all such pairs were spelling, plural or translation variants:
    "Bachelor Project" / "Bachelor Projekt"), or from 0.93 with a shared relation neighbour;
  - 0.90–0.95 → suggestion only ("possible" / "probable futures" 0.93 are different terms).
  Result on the KB: 61 automatic merges, 49 suggestions, no wrong merge found on review.
  Shared *sources* are not used as evidence: one paper names distinct look-alike concepts
  side by side. DEG-RAG-style neighbour overlap is used only as a tie-breaker (few entity
  pairs share relation neighbours here).
- **Merge**: mentions and relations move to the more-mentioned entity, self-loops are
  dropped, the other name becomes an alias, and the merge is logged in `entity_merges`
  (also suggestions). No undo: a wrong merge is reported from `tkb entities merges`, and
  the guide's manual merge/rename UI stays dropped.
- **Eval**: `eval/questions.yaml` `entity_variants` lists known variants (`same`) and
  look-alikes (`different`); `tkb eval` reports `entity_duplicates` and `entity_wrong_merges`.
- **Graph view**: Cytoscape.js 3.34.2 (MIT), vendored under `api/static/vendor/` because the
  UI must work offline and without a CDN seeing the user's traffic. `/api/entities?q=`,
  `/api/entities/{id}` (sources = chunks that mention it or state one of its relations),
  `/api/entities/{id}/graph?hops=1|2` (at most 150 nodes, the most-mentioned kept). Clicking
  an edge opens the relation's chunk in the existing viewer.
