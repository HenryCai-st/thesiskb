"""Logging: a rotating JSON-lines file in data/logs/ for bug reports, plus the console.

The file gets INFO and above from every module; the console keeps showing warnings only
(ingest swaps in its rich handler at INFO via `set_console_handler`).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_NAME = "thesiskb.log"
CONTEXT_FIELDS = ("sha256", "stage", "source", "trace_id")  # pass via `extra={...}`
NOISY = ("LiteLLM", "httpx", "httpcore", "docling", "transformers", "urllib3", "filelock")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        d |= {k: getattr(record, k) for k in CONTEXT_FIELDS if hasattr(record, k)}
        if record.exc_info:
            d["traceback"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False, default=str)


def log_dir() -> Path:
    from thesiskb.core.config import get_pipeline_config

    return get_pipeline_config().data_dir / "logs"


def setup_logging(directory: Path | None = None) -> Path:
    """Attach the file and console handlers once per process; returns the log file path."""
    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, "_tkb_file", False):
            return Path(h.baseFilename)
    directory = directory or log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        directory / LOG_NAME, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonFormatter())
    fh._tkb_file = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console._tkb_console = True  # type: ignore[attr-defined]
    root.addHandler(console)
    root.setLevel(logging.INFO)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return Path(fh.baseFilename)


def set_console_handler(handler: logging.Handler, level: int = logging.INFO) -> None:
    """Replace the console handler (e.g. rich output above progress bars)."""
    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, "_tkb_console", False)]:
        root.removeHandler(h)
    handler.setLevel(level)
    handler._tkb_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
