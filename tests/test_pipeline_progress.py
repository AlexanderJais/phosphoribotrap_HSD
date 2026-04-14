"""Smoke tests for the pipeline runner.

Covers:

* progress-callback monotonicity (composed per-sample + per-step)
* empty-record-list safety
* ``run_salmon`` post-run existence checks for ``quant.sf`` AND
  ``quant.genes.sf`` — the latter is the sanity check that catches
  a silent ``-g <bad_path>`` failure where salmon exits rc=0 without
  writing the gene-level aggregation (fixed in the HIGH #3 audit
  follow-up).

We mock the subprocess layer entirely so no fastp / salmon binary is
needed and the tests run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from phosphotrap import pipeline as pipeline_mod
from phosphotrap.pipeline import StepResult, run_pipeline, run_salmon
from phosphotrap.samples import SampleRecord


def _fake_run_fastp(rec, report_dir, threads, force, log_file, progress_cb=None):
    # Tick several inner-progress steps.
    for f in (0.1, 0.3, 0.5, 0.8, 1.0):
        if progress_cb is not None:
            progress_cb(f)
    return StepResult("fastp", rec.name(), True, 0.0, "ok")


def _fake_run_salmon(rec, salmon_index, tx2gene, output_dir, threads, libtype,
                    force, log_file, use_trimmed, report_dir, progress_cb=None):
    for f in (0.2, 0.6, 1.0):
        if progress_cb is not None:
            progress_cb(f)
    return StepResult("salmon", rec.name(), True, 0.0, "ok")


def _records(n: int = 3) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for i in range(n):
        out.append(SampleRecord(
            ccg_id=f"ccg{i}",
            sample=f"IP{i}",
            comment="",
            replicate=i + 1,
            group="NCD",
            fraction="IP",
            fastq_r1=f"/tmp/fake_R1_{i}.fastq.gz",
            fastq_r2=f"/tmp/fake_R2_{i}.fastq.gz",
        ))
    return out


def test_progress_fractions_are_monotonic_and_reach_one(tmp_path: Path):
    recs = _records(3)
    fractions: list[float] = []

    def cb(sample_idx, step_idx, frac, msg):
        fractions.append(frac)

    with patch.object(pipeline_mod, "run_fastp", side_effect=_fake_run_fastp), \
         patch.object(pipeline_mod, "run_salmon", side_effect=_fake_run_salmon):
        run_pipeline(
            recs,
            salmon_index=tmp_path,
            tx2gene=tmp_path / "tx2g.tsv",
            output_dir=tmp_path / "out",
            report_dir=tmp_path / "rep",
            threads=2,
            run_fastp_step=True,
            libtype="A",
            force=False,
            progress_cb=cb,
            dry_run=False,
        )

    assert fractions, "progress callback was never called"
    # Monotonic non-decreasing.
    for a, b in zip(fractions, fractions[1:]):
        assert b + 1e-9 >= a, f"progress went backwards: {a} -> {b}"
    # Hits the top.
    assert fractions[-1] == 1.0
    # At least one mid-sample tick (a value strictly between 0 and 1).
    assert any(0.0 < f < 1.0 for f in fractions)


def _one_record() -> SampleRecord:
    return SampleRecord(
        ccg_id="ccg1",
        sample="IP1",
        comment="",
        replicate=1,
        group="NCD",
        fraction="IP",
        fastq_r1="/tmp/fake_R1.fastq.gz",
        fastq_r2="/tmp/fake_R2.fastq.gz",
    )


def test_run_salmon_flags_missing_quant_genes_sf(tmp_path: Path):
    """HIGH #3 post-fix: quant.genes.sf must exist for ok=True.

    Simulates the silent failure mode where salmon exits rc=0 without
    writing gene-level aggregation — which is what happens when
    ``-g <tx2gene>`` points at a directory / empty file / a file with
    transcript IDs that don't match the salmon index.
    """
    rec = _one_record()
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "rep"

    # Fake _run_tee: write quant.sf but NOT quant.genes.sf, return rc=0.
    def fake_run_tee(cmd, log_file, progress_cb=None, total_expected=1.0):
        quant_dir = output_dir / "salmon" / rec.name()
        quant_dir.mkdir(parents=True, exist_ok=True)
        (quant_dir / "quant.sf").write_text("Name\tLength\tEffectiveLength\tTPM\tNumReads\n")
        # Deliberately do NOT write quant.genes.sf.
        if progress_cb is not None:
            progress_cb(1.0)
        return 0, "fake salmon tail"

    with patch.object(pipeline_mod, "_run_tee", side_effect=fake_run_tee):
        res = run_salmon(
            rec,
            salmon_index=tmp_path / "idx",
            tx2gene=tmp_path / "tx2g.tsv",
            output_dir=output_dir,
            threads=2,
            libtype="A",
            force=False,
            log_file=tmp_path / "log.log",
            use_trimmed=False,
            report_dir=report_dir,
        )

    assert res.ok is False
    assert res.step == "salmon"
    assert "quant.genes.sf" in res.message
    # Error should point the user at the right fix.
    assert "tx2gene" in res.message.lower() or "Reference tab" in res.message


def test_run_salmon_happy_path(tmp_path: Path):
    """Both quant.sf and quant.genes.sf present -> ok=True."""
    rec = _one_record()
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "rep"

    def fake_run_tee(cmd, log_file, progress_cb=None, total_expected=1.0):
        quant_dir = output_dir / "salmon" / rec.name()
        quant_dir.mkdir(parents=True, exist_ok=True)
        (quant_dir / "quant.sf").write_text("Name\tLength\tEffectiveLength\tTPM\tNumReads\n")
        (quant_dir / "quant.genes.sf").write_text("Name\tLength\tEffectiveLength\tTPM\tNumReads\n")
        if progress_cb is not None:
            progress_cb(1.0)
        return 0, "fake salmon tail"

    with patch.object(pipeline_mod, "_run_tee", side_effect=fake_run_tee):
        res = run_salmon(
            rec,
            salmon_index=tmp_path / "idx",
            tx2gene=tmp_path / "tx2g.tsv",
            output_dir=output_dir,
            threads=2,
            libtype="A",
            force=False,
            log_file=tmp_path / "log.log",
            use_trimmed=False,
            report_dir=report_dir,
        )

    assert res.ok is True
    assert res.message == "ok"


def test_run_salmon_cache_hit_requires_both_files(tmp_path: Path):
    """If only quant.sf is cached, we must NOT short-circuit — the
    pre-existing cache check at the top of run_salmon already enforces
    this (``quant_sf.exists() and quant_genes_sf.exists()``), but let's
    pin it down with a test so nobody accidentally relaxes it later."""
    rec = _one_record()
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "rep"
    quant_dir = output_dir / "salmon" / rec.name()
    quant_dir.mkdir(parents=True, exist_ok=True)
    # Only quant.sf pre-exists, no quant.genes.sf.
    (quant_dir / "quant.sf").write_text("stale")

    # _run_tee should be CALLED (not a cache hit) because quant.genes.sf
    # is absent. Our fake below re-writes quant.sf but still skips
    # quant.genes.sf so the post-run check flags it.
    called = {"n": 0}

    def fake_run_tee(cmd, log_file, progress_cb=None, total_expected=1.0):
        called["n"] += 1
        (quant_dir / "quant.sf").write_text("fresh")
        if progress_cb is not None:
            progress_cb(1.0)
        return 0, "fake"

    with patch.object(pipeline_mod, "_run_tee", side_effect=fake_run_tee):
        res = run_salmon(
            rec,
            salmon_index=tmp_path / "idx",
            tx2gene=tmp_path / "tx2g.tsv",
            output_dir=output_dir,
            threads=2,
            libtype="A",
            force=False,
            log_file=tmp_path / "log.log",
            use_trimmed=False,
            report_dir=report_dir,
        )

    assert called["n"] == 1, "cache check must not short-circuit without quant.genes.sf"
    assert res.ok is False
    assert "quant.genes.sf" in res.message


def test_empty_records_final_callback_has_nonnegative_indices(tmp_path: Path):
    """Running with an empty record list must not emit sample_idx == -1."""
    observed: list[tuple[int, int, float, str]] = []

    def cb(sample_idx, step_idx, frac, msg):
        observed.append((sample_idx, step_idx, frac, msg))

    with patch.object(pipeline_mod, "run_fastp", side_effect=_fake_run_fastp), \
         patch.object(pipeline_mod, "run_salmon", side_effect=_fake_run_salmon):
        run_pipeline(
            [],
            salmon_index=tmp_path,
            tx2gene=tmp_path / "tx2g.tsv",
            output_dir=tmp_path / "out",
            report_dir=tmp_path / "rep",
            threads=2,
            run_fastp_step=True,
            libtype="A",
            force=False,
            progress_cb=cb,
            dry_run=False,
        )

    assert observed, "callback was never called"
    for sample_idx, step_idx, frac, _ in observed:
        assert sample_idx >= 0
        assert step_idx >= 0
        assert 0.0 <= frac <= 1.0
    assert observed[-1][2] == 1.0
