"""Tests for config.validate_reference_paths.

Covers the footgun that broke the first live pipeline run: a Config-tab
text_input ended up with a DIRECTORY in ``tx2gene_tsv`` instead of the
``tx2gene.tsv`` file inside it, which caused salmon quant to fail
silently and anota2seq to crash with a German R error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phosphotrap.config import validate_reference_paths


def _make_index(tmp_path: Path, name: str = "salmon_index") -> Path:
    idx = tmp_path / name
    idx.mkdir()
    (idx / "info.json").write_text("{}")
    return idx


def test_happy_path(tmp_path: Path):
    idx = _make_index(tmp_path)
    tx = tmp_path / "tx2gene.tsv"
    tx.write_text("ENSMUST1\tENSMUSG1\tGene1\n")

    errs = validate_reference_paths(str(idx), str(tx))

    assert errs == []


def test_empty_salmon_index_is_an_error(tmp_path: Path):
    tx = tmp_path / "tx2gene.tsv"
    tx.write_text("ENSMUST1\tENSMUSG1\tGene1\n")

    errs = validate_reference_paths("", str(tx))

    assert any("index" in e.lower() and "empty" in e.lower() for e in errs)


def test_empty_tx2gene_is_an_error(tmp_path: Path):
    idx = _make_index(tmp_path)

    errs = validate_reference_paths(str(idx), "")

    assert any("tx2gene" in e.lower() and "empty" in e.lower() for e in errs)


def test_salmon_index_missing(tmp_path: Path):
    tx = tmp_path / "tx2gene.tsv"
    tx.write_text("ENSMUST1\tENSMUSG1\tGene1\n")

    errs = validate_reference_paths(str(tmp_path / "nope"), str(tx))

    assert any("does not exist" in e for e in errs)


def test_salmon_index_is_a_file(tmp_path: Path):
    wrong = tmp_path / "salmon_index"
    wrong.write_text("not a directory")
    tx = tmp_path / "tx2gene.tsv"
    tx.write_text("ENSMUST1\tENSMUSG1\tGene1\n")

    errs = validate_reference_paths(str(wrong), str(tx))

    assert any("directory" in e.lower() for e in errs)


def test_salmon_index_directory_without_info_json(tmp_path: Path):
    idx = tmp_path / "salmon_index"
    idx.mkdir()
    # no info.json -> not a real index
    tx = tmp_path / "tx2gene.tsv"
    tx.write_text("ENSMUST1\tENSMUSG1\tGene1\n")

    errs = validate_reference_paths(str(idx), str(tx))

    assert any("info.json" in e for e in errs)


def test_tx2gene_missing(tmp_path: Path):
    idx = _make_index(tmp_path)

    errs = validate_reference_paths(
        str(idx), str(tmp_path / "no_such_tx2gene.tsv")
    )

    assert any("does not exist" in e for e in errs)


def test_tx2gene_points_at_a_directory(tmp_path: Path):
    """The exact footgun from the first live pipeline run."""
    idx = _make_index(tmp_path)
    # User accidentally pasted the parent Reference-tab destination
    # directory into tx2gene_tsv instead of the file inside it.
    wrong_tx = tmp_path / "reference_dir"
    wrong_tx.mkdir()

    errs = validate_reference_paths(str(idx), str(wrong_tx))

    assert any("FILE" in e or "file" in e for e in errs)
    # The error should point the user at the Reference tab button.
    assert any("Reference tab" in e or "Use these paths" in e for e in errs)


def test_tx2gene_empty_file(tmp_path: Path):
    idx = _make_index(tmp_path)
    tx = tmp_path / "tx2gene.tsv"
    tx.touch()  # zero-byte file

    errs = validate_reference_paths(str(idx), str(tx))

    assert any("empty" in e.lower() for e in errs)


def test_multiple_errors_returned_together(tmp_path: Path):
    errs = validate_reference_paths(
        str(tmp_path / "missing_idx"),
        str(tmp_path / "missing_tx"),
    )
    assert len(errs) >= 2
