"""Tests for anota2seq / DESeq2 runner cache invalidation.

These tests mock out ``subprocess.run`` entirely so no R stack is
needed. The point is to prove that the skip-if-cached path actually
verifies the input spec — not just "do the output files exist?".
Before the spec-check was added, changing ``cfg.anota_delta_pt``
between two runs would silently return the old result with a
misleading "cache hit" message.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from phosphotrap import anota2seq_runner, deseq2_runner
from phosphotrap.config import AppConfig
from phosphotrap.samples import SampleRecord


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------
def _six_records() -> list[SampleRecord]:
    """3 NCD pairs + 3 HSD1 pairs = 12 SampleRecords.

    That unfolds to 6 matched animal pairs inside
    ``records_for_contrast(alt=HSD1, ref=NCD)``.
    """
    out = []
    for grp in ("NCD", "HSD1"):
        for rep in (1, 2, 3):
            out.append(SampleRecord(
                ccg_id=f"ccg-{grp}-ip-{rep}",
                sample=f"IP_{grp}_{rep}",
                comment="",
                replicate=rep,
                group=grp,
                fraction="IP",
            ))
            out.append(SampleRecord(
                ccg_id=f"ccg-{grp}-in-{rep}",
                sample=f"IN_{grp}_{rep}",
                comment="",
                replicate=rep,
                group=grp,
                fraction="INPUT",
            ))
    return out


def _fake_proc(returncode: int = 0, stdout: str = "ok\n") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


# ======================================================================
# anota2seq cache tests
# ======================================================================
def test_anota2seq_cache_hit_when_spec_matches(tmp_path: Path):
    """Pre-populate scratch with a spec matching the current cfg and
    fake output TSVs. run_anota2seq must short-circuit (no subprocess
    call) and return cache-hit with ok=True.
    """
    records = _six_records()
    cfg = AppConfig(anota_delta_pt=0.1, anota_delta_tp=0.1)
    output_dir = tmp_path
    salmon_root = tmp_path / "salmon"
    tx2gene = tmp_path / "tx2g.tsv"

    scratch = output_dir / "anota2seq" / "HSD1_vs_NCD"
    scratch.mkdir(parents=True)

    # Build the same spec the runner would build for these inputs.
    from phosphotrap.samples import records_for_contrast

    contrast_pairs = records_for_contrast(records, alt_group="HSD1", ref_group="NCD")
    matching_spec = anota2seq_runner._build_spec(
        contrast_pairs, salmon_root, tx2gene, "HSD1", "NCD", cfg,
    )
    (scratch / "spec.json").write_text(
        anota2seq_runner._serialise_spec(matching_spec)
    )

    # Fake all three regModes output TSVs.
    for name in anota2seq_runner._ANOTA2SEQ_OUTPUT_NAMES:
        (scratch / f"{name}.tsv").write_text(
            "gene_id\tapvRvdPAdj\ng1\t0.01\n"
        )
    # Completed-run sentinel — without this the cache-hit path refuses
    # the cache because the runner can't distinguish a clean finish
    # from an interrupted mid-write.
    (scratch / anota2seq_runner._DONE_MARKER).write_text("test\n")

    with patch.object(subprocess, "run") as mock_run:
        res = anota2seq_runner.run_anota2seq(
            records,
            alt_group="HSD1",
            ref_group="NCD",
            salmon_root=salmon_root,
            tx2gene=tx2gene,
            cfg=cfg,
            output_dir=output_dir,
        )

    assert res.ok, res.message
    assert "cache hit" in res.message
    assert "spec verified" in res.message
    # Most important: the R subprocess was NOT called.
    mock_run.assert_not_called()
    # The cached TSV was read into the result.
    assert not res.translation.empty


def test_anota2seq_cache_miss_when_thresholds_change(tmp_path: Path):
    """Pre-populate scratch with a spec for ``anota_delta_pt=0.1``,
    then call the runner with ``anota_delta_pt=0.2``. The spec
    mismatch must invalidate the cache — subprocess.run must be
    called, and the fresh spec must be written over the old one.
    This is the bug the fix addresses.
    """
    records = _six_records()
    output_dir = tmp_path
    salmon_root = tmp_path / "salmon"
    tx2gene = tmp_path / "tx2g.tsv"

    scratch = output_dir / "anota2seq" / "HSD1_vs_NCD"
    scratch.mkdir(parents=True)

    # Old spec: delta_pt = 0.1
    from phosphotrap.samples import records_for_contrast

    contrast_pairs = records_for_contrast(records, alt_group="HSD1", ref_group="NCD")
    old_cfg = AppConfig(anota_delta_pt=0.1)
    old_spec = anota2seq_runner._build_spec(
        contrast_pairs, salmon_root, tx2gene, "HSD1", "NCD", old_cfg,
    )
    old_spec_text = anota2seq_runner._serialise_spec(old_spec)
    (scratch / "spec.json").write_text(old_spec_text)

    for name in anota2seq_runner._ANOTA2SEQ_OUTPUT_NAMES:
        (scratch / f"{name}.tsv").write_text("gene_id\tfoo\ng1\t1\n")

    # New cfg with different delta_pt — the cached spec should NOT match.
    new_cfg = AppConfig(anota_delta_pt=0.2)

    def fake_run(cmd, **kwargs):
        # Pretend R ran and re-wrote the output TSVs (empty but
        # present) so the post-run output validator finds parseable
        # files with the expected gene_id column. Also write the
        # .done sentinel that the runner now requires as proof the
        # R script finished dump_one for all three regmodes.
        for name in anota2seq_runner._ANOTA2SEQ_OUTPUT_NAMES:
            (scratch / f"{name}.tsv").write_text("gene_id\tfoo\n")
        (scratch / anota2seq_runner._DONE_MARKER).write_text("test\n")
        return _fake_proc(returncode=0, stdout="ok\n")

    with patch.object(subprocess, "run", side_effect=fake_run) as mock_run:
        res = anota2seq_runner.run_anota2seq(
            records,
            alt_group="HSD1",
            ref_group="NCD",
            salmon_root=salmon_root,
            tx2gene=tx2gene,
            cfg=new_cfg,
            output_dir=output_dir,
        )

    assert res.ok, res.message
    # Cache must have been invalidated — subprocess was called.
    mock_run.assert_called_once()
    # The new cfg's threshold must have overwritten the old spec on disk.
    written_spec = json.loads((scratch / "spec.json").read_text())
    assert written_spec["selDeltaPT"] == 0.2


def test_anota2seq_force_rerun_bypasses_matching_cache(tmp_path: Path):
    """Even when the cached spec matches the fresh spec, force_rerun
    must cause a re-invocation.
    """
    records = _six_records()
    output_dir = tmp_path
    salmon_root = tmp_path / "salmon"
    tx2gene = tmp_path / "tx2g.tsv"

    scratch = output_dir / "anota2seq" / "HSD1_vs_NCD"
    scratch.mkdir(parents=True)

    cfg = AppConfig(anota_delta_pt=0.1, force_rerun=True)
    from phosphotrap.samples import records_for_contrast

    contrast_pairs = records_for_contrast(records, alt_group="HSD1", ref_group="NCD")
    matching_spec = anota2seq_runner._build_spec(
        contrast_pairs, salmon_root, tx2gene, "HSD1", "NCD", cfg,
    )
    (scratch / "spec.json").write_text(
        anota2seq_runner._serialise_spec(matching_spec)
    )
    for name in anota2seq_runner._ANOTA2SEQ_OUTPUT_NAMES:
        (scratch / f"{name}.tsv").write_text("gene_id\tfoo\ng1\t1\n")
    (scratch / anota2seq_runner._DONE_MARKER).write_text("test\n")

    def fake_run(cmd, **kwargs):
        for name in anota2seq_runner._ANOTA2SEQ_OUTPUT_NAMES:
            (scratch / f"{name}.tsv").write_text("gene_id\tfoo\n")
        (scratch / anota2seq_runner._DONE_MARKER).write_text("test\n")
        return _fake_proc(returncode=0, stdout="ok\n")

    with patch.object(subprocess, "run", side_effect=fake_run) as mock_run:
        res = anota2seq_runner.run_anota2seq(
            records,
            alt_group="HSD1",
            ref_group="NCD",
            salmon_root=salmon_root,
            tx2gene=tx2gene,
            cfg=cfg,
            output_dir=output_dir,
        )

    assert res.ok, res.message
    mock_run.assert_called_once()
    assert "cache hit" not in res.message


# ======================================================================
# DESeq2 cache tests
# ======================================================================
def test_deseq2_cache_hit_when_spec_matches(tmp_path: Path):
    records = _six_records()
    cfg = AppConfig()
    output_dir = tmp_path
    salmon_root = tmp_path / "salmon"
    tx2gene = tmp_path / "tx2g.tsv"

    scratch = output_dir / "deseq2" / "HSD1_vs_NCD"
    scratch.mkdir(parents=True)

    from phosphotrap.samples import records_for_contrast

    contrast_pairs = records_for_contrast(records, alt_group="HSD1", ref_group="NCD")
    matching_spec = deseq2_runner._build_spec(
        contrast_pairs, salmon_root, tx2gene, "HSD1", "NCD", cfg,
    )
    (scratch / "spec.json").write_text(
        deseq2_runner._serialise_spec(matching_spec)
    )
    (scratch / "interaction.tsv").write_text(
        "gene_id\tlog2FoldChange\tpvalue\ng1\t0.5\t0.01\n"
    )
    # Completed-run sentinel (see deseq2_runner._DONE_MARKER).
    (scratch / deseq2_runner._DONE_MARKER).write_text("test\n")

    with patch.object(subprocess, "run") as mock_run:
        res = deseq2_runner.run_deseq2_interaction(
            records,
            alt_group="HSD1",
            ref_group="NCD",
            salmon_root=salmon_root,
            tx2gene=tx2gene,
            cfg=cfg,
            output_dir=output_dir,
        )

    assert res.ok, res.message
    assert "cache hit" in res.message
    mock_run.assert_not_called()
    assert not res.table.empty


def test_deseq2_cache_miss_when_sample_list_changes(tmp_path: Path):
    """Swap the records between two runs. The cached spec lists the
    first set of sample names; the second run's fresh spec doesn't
    match → cache must invalidate.
    """
    records_a = _six_records()

    # Same records but with a different replicate numbering (names differ).
    records_b = []
    for rec in records_a:
        records_b.append(SampleRecord(
            ccg_id=rec.ccg_id,
            sample=rec.sample,
            comment=rec.comment,
            # +10 changes the replicate, which changes .name() which
            # flows into the sample_names field of the spec.
            replicate=rec.replicate + 10,
            group=rec.group,
            fraction=rec.fraction,
        ))

    output_dir = tmp_path
    salmon_root = tmp_path / "salmon"
    tx2gene = tmp_path / "tx2g.tsv"

    scratch = output_dir / "deseq2" / "HSD1_vs_NCD"
    scratch.mkdir(parents=True)

    from phosphotrap.samples import records_for_contrast

    cfg = AppConfig()

    # Write the spec for records_a
    pairs_a = records_for_contrast(records_a, alt_group="HSD1", ref_group="NCD")
    old_spec = deseq2_runner._build_spec(
        pairs_a, salmon_root, tx2gene, "HSD1", "NCD", cfg,
    )
    (scratch / "spec.json").write_text(
        deseq2_runner._serialise_spec(old_spec)
    )
    (scratch / "interaction.tsv").write_text("gene_id\tpvalue\ng1\t0.01\n")

    def fake_run(cmd, **kwargs):
        # Simulate a successful R run that re-writes the output and
        # drops the .done sentinel the runner now requires as proof
        # the R script completed write.table.
        (scratch / "interaction.tsv").write_text(
            "gene_id\tlog2FoldChange\tpvalue\ng2\t0.3\t0.05\n"
        )
        (scratch / deseq2_runner._DONE_MARKER).write_text("test\n")
        return _fake_proc(returncode=0, stdout="ok\n")

    with patch.object(subprocess, "run", side_effect=fake_run) as mock_run:
        res = deseq2_runner.run_deseq2_interaction(
            records_b,       # <-- different records
            alt_group="HSD1",
            ref_group="NCD",
            salmon_root=salmon_root,
            tx2gene=tx2gene,
            cfg=cfg,
            output_dir=output_dir,
        )

    assert res.ok, res.message
    mock_run.assert_called_once()
    # Spec on disk should now reflect records_b's names (replicate + 10)
    written = json.loads((scratch / "spec.json").read_text())
    assert any("11" in name or "12" in name or "13" in name
               for name in written["names"])
