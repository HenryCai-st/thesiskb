"""Parser plug-in point: one Parser per modality (BUILD_GUIDE §8 "Parser interface").

A parser turns files of its modalities into stage outputs (`parse`, run once per batch so a GPU
model is loaded once) and turns those outputs into segments + chunks (`chunk`, cheap, re-run by
`tkb reindex`). Built-ins: documents (Docling), images (VLM + OCR), audio/video
(Whisper + pyannote + keyframes), registered by `thesiskb.ingest.pipeline`.

Adding a file type without touching the core:

    from thesiskb.ingest.parsers import Parser, register_parser

    class EmailParser(Parser):
        name = "email"
        modalities = frozenset({"email"})
        suffixes = {".eml": "email"}
        output = "email.json"

        def parse(self, ctx, items): ...        # write processed/<sha>/email.json per file
        def chunk(self, ctx, src, f, count): ... # Blocks → ctx.write_chunks(src, blocks)

    register_parser(EmailParser())

and list the module under `plugins:` in config/pipeline.yaml (or expose it as a
`thesiskb.plugins` entry point in your own package).
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from thesiskb.core.config import PipelineConfig
    from thesiskb.core.models import Source
    from thesiskb.ingest.detect import DetectedFile
    from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)


@dataclass
class ParseContext:
    """What a parser gets from the pipeline."""

    session: Session
    cfg: PipelineConfig
    gateway: LLMGateway
    force: bool = False
    metadata_lookup: bool = True
    # mark(src, f, fn) runs fn as the source's "parse" stage (job row, error capture)
    mark: Callable[[Source, DetectedFile, Callable[[], object]], bool] = field(
        default=lambda src, f, fn: bool(fn())
    )

    def out_dir(self, src: Source) -> Path:
        return Path(self.cfg.processed_dir) / src.sha256

    @property
    def options(self) -> dict[str, Any]:
        """Free-form sections of pipeline.yaml (ocr, audio, video, literature, ...)."""
        return self.cfg.model_extra or {}


class Parser(ABC):
    name: str = ""
    modalities: frozenset[str] = frozenset()
    suffixes: dict[str, str] = {}  # extra file suffix → modality this parser adds
    output: str = ""  # file in processed/<sha>/ whose existence means "parse done"

    @abstractmethod
    def parse(self, ctx: ParseContext, items: list[tuple[Source, DetectedFile]]) -> None:
        """Produce stage outputs for a batch; call ctx.mark(src, f, fn) per file."""

    @abstractmethod
    def chunk(self, ctx: ParseContext, src: Source, f: DetectedFile, count) -> None:
        """Replace the source's segments and chunks from its stage outputs."""

    def output_for(self, src: Source) -> str:
        return self.output


_PARSERS: dict[str, Parser] = {}  # modality → parser


def register_parser(parser: Parser) -> Parser:
    from thesiskb.ingest import detect

    for m in parser.modalities:
        if m in _PARSERS and _PARSERS[m] is not parser:
            log.info("parser %s replaces %s for %s", parser.name, _PARSERS[m].name, m)
        _PARSERS[m] = parser
    for suffix, modality in parser.suffixes.items():
        detect.MODALITY_BY_SUFFIX[suffix.lower()] = modality
    return parser


def parser_for(modality: str) -> Parser | None:
    return _PARSERS.get(modality)


def supported_modalities() -> set[str]:
    return set(_PARSERS)


_loaded: set[str] = set()


def load_plugins(modules: list[str] | None = None) -> list[str]:
    """Import plug-in modules (from pipeline.yaml `plugins:` and `thesiskb.plugins` entry
    points). Importing a module is what registers its parsers/backends/extractors."""
    names = list(modules or [])
    with contextlib.suppress(Exception):  # broken package metadata must not stop ingestion
        names += [ep.value for ep in entry_points(group="thesiskb.plugins")]
    loaded = []
    for name in names:
        module = name.split(":", 1)[0]
        if module in _loaded:
            continue
        importlib.import_module(module)
        _loaded.add(module)
        loaded.append(module)
    return loaded
