"""Example plug-in extractor: decisions taken in meetings and notes.

Enable it in config/pipeline.yaml:

    plugins:
      - examples.plugins.decisions

Every meeting/interview/note chunk is then also asked for "decisions" in the same LLM call as
tasks and entities. Items need a verbatim evidence quote (or are dropped, like built-ins) and
are written to data/processed/<sha>/decisions.json — a plug-in can store its results anywhere;
a file keeps this example free of database migrations.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from thesiskb.core.config import get_pipeline_config
from thesiskb.ingest.extract import Extractor, register_extractor


class Decision(BaseModel):
    decision: str
    decided_by: str | None = None
    evidence_quote: str


class DecisionsExtractor(Extractor):
    key = "decisions"
    doc = (
        '"decisions": decisions that were taken or agreed (not tasks, not opinions): '
        "{decision (one sentence), decided_by (as written, or null), evidence_quote}"
    )
    item = Decision

    def applies(self, src):
        return (src.source_type or "") in {"meeting", "interview", "note"}

    def write(self, session, src, items):
        rows = [
            {"chunk_id": str(chunk.id), **d.model_dump()}
            for chunk, decisions in items
            for d in decisions
        ]
        out = Path(get_pipeline_config().processed_dir) / src.sha256 / "decisions.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(rows)


register_extractor(DecisionsExtractor())
