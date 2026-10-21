# ADR 0007 — Router, answer paths, and the graph decision

Date: 2026-10-22 · Status: accepted (answers BUILD_GUIDE §11.3 "Graph layer needed at all?")

## Paths as context providers
`answer/paths.py`: each path returns citable passages; one generation + the usual citation and
support checks run over the merged, renumbered passages. Multi-path questions ("What did Prof. X
say about my deadline?" → person + task) need no extra code.

| path | passages |
|---|---|
| general | hybrid search + rerank |
| literature | 3 LLM sub-queries + the question → hybrid search each (papers/books) → RRF → rerank → per-chunk evidence summary with relevance 0-10 (PaperQA2-style), keep the best 8 |
| person | the person's extracted statements (with their transcript line) + search restricted to chunks where they speak (speaker labels) or papers they authored |
| task | open tasks from SQL, sorted by due date |
| entity | graph-lite lookup (below) |

## Router
Rules first (person name/alias in the question → person; schedule/deadline words → task;
"which … appear in my sources" → entity), then one LLM call (role `router`, thinking off,
~0.3 s) that may add paths. Paths without data fall back to `general`; literature is dropped for
a person who authored no paper; `entity` is never used without a search path. Router accuracy on
the 11 gold questions: 1.0 (also German and multi-path probes). Known limitation: a person alias
that is a common word ("Example") triggers the person path anywhere.

## Graph decision: SQL over the extracted tables, no LightRAG / Neo4j
Gold relation questions, answered with plain retrieval first:
- rel-01 "How are diffusion models connected to VAEs, GANs, normalizing flows?" — passes with
  the literature path (one paper, section 6).
- rel-02 "Which NVIDIA products appear in my sources, and what is each one for?" — **failed**:
  refused, nothing above the rerank threshold. No passage looks like a product list.

The entities/relations tables already held the answer (DGX Spark, HGX, Jetson variants, typed
`product`, with relations like "DGX Spark — enables running — LLMs"). `retrieval/graph.py`
answers it with SQL: seed entities named in the question, the asked type ("products"),
entities co-occurring with the seed (same source, or seed in the source title) and 2-hop
neighbours via a recursive CTE; evidence is grouped per mentioning chunk. rel-02 now passes
(9/9 supported), lookup < 0.1 s.

LightRAG or Neo4j would add a second store to keep in sync for no gain at this size (1k entities).
Revisit if multi-hop questions over many papers fail, or entity counts reach ~100k.

## Costs and open issues
- Literature path is the slowest (~55-70 s per question: sub-queries + 10 evidence summaries).
  `literature.evidence_summaries: false` halves it.
- Entity name variants are not merged ("Jetson Nano Super", "Jetson Orin Nano", "Jetson Orin
  Nano Developer Kit"); the answer LLM is asked to merge them and does so only partly.
- Support rate varies between runs (0.89-0.96 here): with more passages the model writes more
  claims, some going beyond the cited text; the judge flags them.

## Milestone 6 acceptance (eval run 2026-10-22)
retrieval/citation hit 1.0, router accuracy 1.0, quote hit 1.0, keyframe hit 1.0, task recall
1.0, tasks without evidence 0, invalid citations 0 — no regression vs Milestone 5.
