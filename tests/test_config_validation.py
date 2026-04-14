"""Tests for config.validate_reference_paths and validate_fastq_dir.

Covers two footguns reported from the first live run:

1. A Config-tab text_input ended up with a DIRECTORY in
   ``tx2gene_tsv`` instead of the ``tx2gene.tsv`` file inside it,
   which caused salmon quant to fail silently and anota2seq to
   crash with a German R error. Covered by the
   ``validate_reference_paths`` tests.

2. The Samples tab's "Auto-populate fastq paths" button only
   checked ``if not cfg.fastq_dir:`` (emptiness) — a typo in the
   path produced "0 / 18 ready samples" with no explanation.
   Covered by the ``validate_fastq_dir`` tests below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phosphotrap.config import validate_fastq_dir, validate_reference_paths


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


# ----------------------------------------------------------------------
# validate_fastq_dir (HIGH #4)
# ----------------------------------------------------------------------
def _make_fastq_dir(tmp_path: Path, *files: str) -> Path:
    """Create a directory with the named fastq files as empty placeholders."""
    d = tmp_path / "fastqs"
    d.mkdir()
    for name in files:
        (d / name).touch()
    return d


def test_validate_fastq_dir_happy_path(tmp_path: Path):
    d = _make_fastq_dir(
        tmp_path,
        "A006200122_138011_S46_L002_R1_001.fastq.gz",
        "A006200122_138011_S46_L002_R2_001.fastq.gz",
    )
    assert validate_fastq_dir(str(d)) == []


def test_validate_fastq_dir_empty_string_is_unconfigured_not_error(tmp_path: Path):
    """Empty ``fastq_dir`` returns no errors — callers decide what to do.

    On a fresh install the user genuinely hasn't set the path yet;
    treating that as a validation error would light up the Config tab
    with red before they even had a chance to type anything.
    """
    assert validate_fastq_dir("") == []


def test_validate_fastq_dir_missing_path(tmp_path: Path):
    errs = validate_fastq_dir(str(tmp_path / "no_such_dir"))
    assert any("does not exist" in e for e in errs)


def test_validate_fastq_dir_points_at_a_file(tmp_path: Path):
    wrong = tmp_path / "not_a_dir.fastq.gz"
    wrong.touch()
    errs = validate_fastq_dir(str(wrong))
    assert any("file, not a directory" in e.lower() or "not a directory" in e for e in errs)


def test_validate_fastq_dir_empty_directory_is_flagged(tmp_path: Path):
    """The exact HIGH #4 footgun: path is valid and exists, but no reads."""
    d = tmp_path / "empty_fastqs"
    d.mkdir()
    errs = validate_fastq_dir(str(d))
    assert any("no *.fastq.gz" in e or "fastq.gz files" in e for e in errs)


def test_validate_fastq_dir_directory_with_unrelated_files(tmp_path: Path):
    """A directory full of CSVs but no .fastq.gz is still empty for our purposes."""
    d = tmp_path / "wrong_files"
    d.mkdir()
    (d / "samples.csv").touch()
    (d / "notes.txt").touch()
    errs = validate_fastq_dir(str(d))
    assert any("fastq.gz" in e for e in errs)


def test_validate_fastq_dir_expanduser(tmp_path: Path, monkeypatch):
    """``~`` should expand to the home directory before the existence check."""
    # Point HOME at tmp_path and create a fastqs dir there.
    monkeypatch.setenv("HOME", str(tmp_path))
    # pathlib.Path.expanduser reads $HOME on POSIX.
    d = tmp_path / "myreads"
    d.mkdir()
    (d / "sample_R1_001.fastq.gz").touch()
    errs = validate_fastq_dir("~/myreads")
    assert errs == []
