"""Smoke test for progress callback monotonicity.

We mock ``run_fastp`` and ``run_salmon`` so the test does not touch live
fastq files or the fastp/salmon binaries. What we care about is that the
composed progress fraction never goes backwards and finishes at 1.0
after ticking mid-sample at least once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import patch

from phosphotrap import pipeline as pipeline_mod
from phosphotrap.pipeline import StepResult, run_pipeline
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
