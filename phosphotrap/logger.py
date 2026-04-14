"""Central rotating-file logger for the phosphoribotrap app.

Two-step initialisation:

* :func:`get_logger` is cheap and has **no filesystem side effects**.
  It attaches a stream handler on first call so modules that
  ``logger = get_logger()`` at import time don't accidentally create a
  ``logs/`` directory in whatever cwd the user happened to launch
  streamlit from. Tests import these modules without spamming the repo.

* :func:`attach_file_handler` is called once by the Streamlit app at
  startup with the user-configured log directory. It creates the
  directory, installs a :class:`RotatingFileHandler`, and is idempotent
  — a second call with the same directory is a no-op, a call with a
  different directory rotates the handler onto the new location.

Streamlit reruns the script on every interaction, so both handler
installers tag their handlers with ``handler._phosphotrap = True`` and
skip reinstallation if one is already present. Propagation to the root
logger is disabled so Streamlit's own handlers don't double-print.
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

_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# The effective directory for the file handler. Populated by
# :func:`attach_file_handler`; read by :func:`tail_log` so the Logs tab
# always reads from the same place the app is writing to.
_active_log_dir: Optional[Path] = None


def _has_tag(handler: logging.Handler) -> bool:
    return getattr(handler, "_phosphotrap", False)


def _has_stream_handler(logger: logging.Logger) -> bool:
    return any(
        _has_tag(h) and isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def get_logger() -> logging.Logger:
    """Return the package logger. Side-effect-free on the filesystem.

    Installs a stream handler once. No mkdir, no file creation — use
    :func:`attach_file_handler` for that.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not _has_stream_handler(logger):
        sh = logging.StreamHandler()
        sh.setFormatter(_FORMATTER)
        sh._phosphotrap = True  # type: ignore[attr-defined]
        logger.addHandler(sh)

    return logger


def attach_file_handler(log_dir: Optional[Path] = None) -> Path:
    """Install the rotating file handler at ``log_dir``.

    Idempotent — if a phosphotrap file handler is already attached at
    the same resolved path we return immediately. If it's attached at a
    different path we rotate it onto the new one.
    """
    global _active_log_dir
    directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = (directory / DEFAULT_LOG_FILE).resolve()

    logger = get_logger()  # ensures stream handler exists

    for h in list(logger.handlers):
        if _has_tag(h) and isinstance(h, logging.handlers.RotatingFileHandler):
            if Path(h.baseFilename).resolve() == target:
                _active_log_dir = directory
                return target
            logger.removeHandler(h)
            h.close()

    fh = logging.handlers.RotatingFileHandler(
        target, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    fh.setFormatter(_FORMATTER)
    fh._phosphotrap = True  # type: ignore[attr-defined]
    logger.addHandler(fh)
    _active_log_dir = directory
    return target


def active_log_dir() -> Path:
    """Return the directory the file handler is currently writing to."""
    return _active_log_dir if _active_log_dir is not None else DEFAULT_LOG_DIR


def log_path(log_dir: Optional[Path] = None) -> Path:
    directory = Path(log_dir) if log_dir else active_log_dir()
    return directory / DEFAULT_LOG_FILE


def tail_log(
    log_dir: Optional[Path] = None,
    max_lines: int = 500,
    filter_substr: str = "",
) -> str:
    """Return the last ``max_lines`` of the rotating log file.

    Stitches rolled-over backups (.log.1 … .log.N) together so runs that
    straddle a rotation boundary don't lose recent lines. If
    ``log_dir`` is omitted, reads from whichever directory the file
    handler is currently attached to, falling back to ``DEFAULT_LOG_DIR``
    if none has been attached yet. Filtering is a plain substring
    match, case-insensitive; an empty filter returns everything.
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
