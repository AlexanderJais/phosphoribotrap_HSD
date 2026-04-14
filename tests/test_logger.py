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
    list_per_sample_logs,
    read_log_file,
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


def test_list_per_sample_logs_empty_when_dir_missing(tmp_path: Path):
    assert list_per_sample_logs(tmp_path) == []
    assert list_per_sample_logs(tmp_path / "does_not_exist") == []


def test_list_per_sample_logs_returns_sorted_files(tmp_path: Path):
    per_sample = tmp_path / "logs" / "per-sample"
    per_sample.mkdir(parents=True)
    (per_sample / "NCD_IP1.log").write_text("a")
    (per_sample / "HSD1_IP5.log").write_text("b")
    (per_sample / "NCD_INPUT1.log").write_text("c")
    # A non-log file shouldn't be included.
    (per_sample / "notes.txt").write_text("ignored")

    found = list_per_sample_logs(tmp_path)
    names = [p.name for p in found]
    assert names == sorted(names)
    assert "notes.txt" not in names
    assert "NCD_IP1.log" in names
    assert "HSD1_IP5.log" in names
    assert "NCD_INPUT1.log" in names


def test_read_log_file_returns_full_content_when_small(tmp_path: Path):
    p = tmp_path / "small.log"
    p.write_text("hello\nworld\n")
    assert read_log_file(p) == "hello\nworld\n"


def test_read_log_file_truncates_large_files_to_tail(tmp_path: Path):
    """A 2 MB tail limit must drop the front of a much larger file
    and not raise.
    """
    p = tmp_path / "big.log"
    # 3 MB of predictable content; tail should start well past the
    # halfway mark.
    chunk = ("x" * 99 + "\n")  # 100 chars per line incl. newline
    p.write_text(chunk * 30_000)  # ~3 MB

    content = read_log_file(p, max_bytes=2 * 1024 * 1024)
    assert len(content.encode("utf-8")) <= 2 * 1024 * 1024
    # The tail ends with a full line (we didn't leave a ragged
    # partial line at the end).
    assert content.endswith("\n")


def test_read_log_file_missing_returns_empty(tmp_path: Path):
    assert read_log_file(tmp_path / "nope.log") == ""
