"""Central rotating-file logger for the phosphoribotrap app.

Streamlit reruns the entire script on every interaction, so naive
``logging.getLogger().addHandler(...)`` would stack a new handler on every
rerun. We tag each handler we install with ``handler._phosphotrap = True``
and skip reinstallation if one is already present. We also disable
propagation to the root logger so Streamlit's own root handler does not
double-print everything we emit.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

LOGGER_NAME = "phosphotrap"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "phosphotrap.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def _has_phosphotrap_handler(logger: logging.Logger) -> bool:
    for h in logger.handlers:
        if getattr(h, "_phosphotrap", False):
            return True
    return False


def get_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """Return the package logger, installing a rotating file handler once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if _has_phosphotrap_handler(logger):
        return logger

    directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / DEFAULT_LOG_FILE

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler._phosphotrap = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler._phosphotrap = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    return logger


def log_path(log_dir: Optional[Path] = None) -> Path:
    directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    return directory / DEFAULT_LOG_FILE


def tail_log(
    log_dir: Optional[Path] = None,
    max_lines: int = 500,
    filter_substr: str = "",
) -> str:
    """Return the last ``max_lines`` of the rotating log file.

    Stitches rolled-over backups (.log.1 … .log.N) together so runs that
    straddle a rotation boundary don't lose recent lines. Filtering is a
    plain substring match, case-insensitive; an empty filter returns
    everything.
    """
    base = log_path(log_dir)
    candidates: list[Path] = []
    # Oldest → newest so the tail ends with the most recent content.
    for i in range(BACKUP_COUNT, 0, -1):
        p = base.with_name(base.name + f".{i}")
        if p.exists():
            candidates.append(p)
    if base.exists():
        candidates.append(base)

    if not candidates:
        return ""

    lines: list[str] = []
    for p in candidates:
        try:
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                lines.extend(fh.readlines())
        except OSError:
            continue

    if filter_substr:
        needle = filter_substr.lower()
        lines = [ln for ln in lines if needle in ln.lower()]

    return "".join(lines[-max_lines:])
