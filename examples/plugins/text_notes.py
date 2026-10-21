"""Example plug-in: plain-text notes (.txt), e.g. meeting minutes typed during a meeting.

Enable it in config/pipeline.yaml:

    plugins:
      - examples.plugins.text_notes

(run tkb from the repo root, or install your plug-in package and use its module path / a
`thesiskb.plugins` entry point). Then `tkb ingest notes/` picks up .txt files: headings
("# Title", "Title:" on its own line) become sections, paragraphs become blocks, and citations
point at line numbers.
"""

from __future__ import annotations

import json
import re

from thesiskb.ingest.chunking import Block
from thesiskb.ingest.parsers import Parser, register_parser
from thesiskb.ingest.pipeline import write_blocks

_HEADING = re.compile(r"^(#{1,6}\s+.+|[^\s].{0,60}:)\s*$")


def parse_text(text: str) -> list[dict]:
    """Paragraphs with their section path and line range (1-based)."""
    blocks: list[dict] = []
    section: list[str] = []
    para: list[str] = []
    start = 0
    md_depth = 0  # depth of the last "#" heading

    def flush(end: int) -> None:
        if para:
            blocks.append(
                {"text": " ".join(para), "section": list(section), "line": start, "line_end": end}
            )
            para.clear()

    lines = text.splitlines()
    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            flush(n - 1)
            continue
        if _HEADING.match(line) and not para:
            heading = line.lstrip("#").strip().rstrip(":")
            if line.startswith("#"):
                md_depth = len(line) - len(line.lstrip("#"))
                section = [*section[: md_depth - 1], heading]
            else:  # "Decisions:" — a sub-heading under the last "#" heading
                section = [*section[:md_depth], heading]
            continue
        if not para:
            start = n
        para.append(line)
    flush(len(lines))
    return blocks


class TextNotesParser(Parser):
    name = "text_notes"
    modalities = frozenset({"text"})
    suffixes = {".txt": "text"}
    output = "text.json"

    def parse(self, ctx, items):
        # cheap, no model: still a "parse" stage so outputs are cached like every other parser
        for src, f in items:

            def work(src=src, f=f):
                out = ctx.out_dir(src)
                out.mkdir(parents=True, exist_ok=True)
                text = f.path.read_text(encoding="utf-8", errors="replace")
                (out / self.output).write_text(json.dumps(parse_text(text)), encoding="utf-8")

            ctx.mark(src, f, work)

    def chunk(self, ctx, src, f, count):
        raw = json.loads((ctx.out_dir(src) / self.output).read_text(encoding="utf-8"))
        blocks = [
            Block(
                text=b["text"],
                section=b["section"],
                locator={"line": b["line"], "line_end": b["line_end"]},
            )
            for b in raw
        ]
        if src.title == f.path.stem and raw:
            src.title = f.path.stem.replace("_", " ")
        write_blocks(ctx, src, blocks, count)


register_parser(TextNotesParser())
