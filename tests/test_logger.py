"""Tests for the lazy logger and ``attach_file_handler`` fallbacks."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from phosphotrap import logger as logger_mod
from phosphotrap.logger import (
    LOGGER_NAME,
    attach_file_handler,
    get_logger,
)


def _drop_phosphotrap_file_handlers() -> None:
    """Detach any file handlers a previous test left attached."""
    log = logging.getLogger(LOGGER_NAME)
    for h in list(log.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler) and \
                getattr(h, "_phosphotrap", False):
            log.removeHandler(h)
            h.close()


def test_get_logger_is_filesystem_side_effect_free(tmp_path: Path, monkeypatch):
    """get_logger() must not create a logs/ directory on import.

    We point DEFAULT_LOG_DIR at a sentinel path and call get_logger;
    the path must not exist afterwards.
    """
    sentinel = tmp_path / "should_not_exist"
    monkeypatch.setattr(logger_mod, "DEFAULT_LOG_DIR", sentinel)

    log = get_logger()
    assert log.name == LOGGER_NAME
    assert not sentinel.exists()


def test_attach_file_handler_falls_back_for_unwriteable_path(
    tmp_path: Path, monkeypatch
):
    """A directory path that can't be created (it's actually a file
    on disk) must not raise. attach_file_handler should warn and
    fall back to the default log dir.
    """
    _drop_phosphotrap_file_handlers()

    # Create a regular file, then ask the logger to use a path under it.
    file_on_disk = tmp_path / "not_a_dir"
    file_on_disk.write_text("blocker")
    bad_path = file_on_disk / "logs"  # creating this would require
                                       # not_a_dir to be a directory.

    fallback = tmp_path / "fallback_logs"
    monkeypatch.setattr(logger_mod, "DEFAULT_LOG_DIR", fallback)

    # Must not raise.
    result = attach_file_handler(bad_path)

    # Result points into the fallback dir, not the bad one.
    assert fallback in result.parents
    assert fallback.exists()
    # A rotating file handler was attached (or at least the dir was
    # created — in worst case we fall back to a tmp dir).
    log = logging.getLogger(LOGGER_NAME)
    has_file_handler = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "_phosphotrap", False)
        for h in log.handlers
    )
    assert has_file_handler


def test_attach_file_handler_empty_path_falls_back(
    tmp_path: Path, monkeypatch
):
    """An empty / whitespace ``log_dir`` must fall back, not create
    ``logs/`` in cwd.
    """
    _drop_phosphotrap_file_handlers()
    fallback = tmp_path / "fallback_logs"
    monkeypatch.setattr(logger_mod, "DEFAULT_LOG_DIR", fallback)

    result = attach_file_handler(Path(""))
    assert fallback in result.parents
    assert fallback.exists()


def test_attach_file_handler_idempotent_on_same_path(
    tmp_path: Path, monkeypatch
):
    """Calling attach_file_handler twice with the same path should
    leave exactly one phosphotrap-tagged RotatingFileHandler attached.
    """
    _drop_phosphotrap_file_handlers()
    log_dir = tmp_path / "logs"

    attach_file_handler(log_dir)
    attach_file_handler(log_dir)

    log = logging.getLogger(LOGGER_NAME)
    tagged_file_handlers = [
        h for h in log.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "_phosphotrap", False)
    ]
    assert len(tagged_file_handlers) == 1
