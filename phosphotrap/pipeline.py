"""fastp + salmon pipeline runner for the phosphoribotrap app.

Design notes:

* Single-path pipeline: no STAR / HISAT2 / featureCounts / MarkDuplicates.
* **No deduplication** anywhere in the pipeline — the 57–69% duplication
  rate is IP enrichment signal, not a PCR artefact. Salmon handles PCR
  duplication probabilistically at the mapping stage.
* Skip-if-cached: both fastp and salmon short-circuit when their outputs
  already exist, gated by the ``force_rerun`` flag.
* Progress callbacks interpolate within a sample so the UI bar advances
  mid-sample rather than jumping at sample boundaries.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .logger import get_logger
from .samples import SampleRecord

StepCallback = Callable[[int, int, float, str], None]
RunCallback = Callable[[int, int, float, str], None]

logger = get_logger()


@dataclass
class StepResult:
    step: str
    sample: str
    ok: bool
    duration_s: float
    message: str


class PipelineError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def check_environment(rscript: str = "Rscript") -> dict[str, dict]:
    """Probe for fastp / salmon / Rscript + critical R packages."""
    report: dict[str, dict] = {}
    for tool in ("fastp", "salmon"):
        path = shutil.which(tool)
        report[tool] = {"ok": path is not None, "path": path or ""}

    path = shutil.which(rscript)
    report["Rscript"] = {"ok": path is not None, "path": path or ""}

    # R packages are only probed if Rscript is on the PATH.
    for pkg in ("anota2seq", "tximport", "DESeq2"):
        if not path:
            report[f"R:{pkg}"] = {"ok": False, "path": ""}
            continue
        try:
            proc = subprocess.run(
                [path, "-e", f'suppressMessages(library({pkg})); cat("ok")'],
                capture_output=True,
                text=True,
                timeout=60,
            )
            ok = proc.returncode == 0 and "ok" in (proc.stdout or "")
            report[f"R:{pkg}"] = {
                "ok": ok,
                "path": "",
                "stderr": (proc.stderr or "").strip()[-500:],
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            report[f"R:{pkg}"] = {"ok": False, "path": "", "stderr": str(exc)}
    return report


# ----------------------------------------------------------------------
# Subprocess plumbing
# ----------------------------------------------------------------------
def _pretty_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run_tee(
    cmd: list[str],
    log_file: Path,
    progress_cb: Optional[Callable[[float], None]] = None,
    total_expected: float = 1.0,
) -> tuple[int, str]:
    """Run a subprocess, tee stdout+stderr to ``log_file``, return (rc, tail).

    ``progress_cb`` receives a float in [0, 1] each time a line is
    emitted. Since we cannot actually parse fastp/salmon progress lines
    reliably across versions, we fake a smooth ramp via line count: the
    callback sees ``min(0.95, n_lines / total_expected)`` on every line,
    then 1.0 once the process exits cleanly.
    """
    logger.info("exec: %s", _pretty_cmd(cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n# {_pretty_cmd(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
        n_lines = 0
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            tail.append(line)
            if len(tail) > 200:
                tail.pop(0)
            n_lines += 1
            if progress_cb is not None:
                progress_cb(min(0.95, n_lines / max(total_expected, 1.0)))
        rc = proc.wait()
        if progress_cb is not None:
            progress_cb(1.0)
    return rc, "".join(tail)


# ----------------------------------------------------------------------
# Step runners
# ----------------------------------------------------------------------
def _fastp_outputs(report_dir: Path, sample: str) -> tuple[Path, Path, Path, Path]:
    trimmed = report_dir / "trimmed"
    trimmed.mkdir(parents=True, exist_ok=True)
    r1_out = trimmed / f"{sample}_R1.trim.fastq.gz"
    r2_out = trimmed / f"{sample}_R2.trim.fastq.gz"
    html_out = report_dir / "fastp" / f"{sample}.html"
    json_out = report_dir / "fastp" / f"{sample}.json"
    html_out.parent.mkdir(parents=True, exist_ok=True)
    return r1_out, r2_out, html_out, json_out


def run_fastp(
    rec: SampleRecord,
    report_dir: Path,
    threads: int,
    force: bool,
    log_file: Path,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> StepResult:
    start = time.time()
    r1_in = Path(rec.fastq_r1)
    r2_in = Path(rec.fastq_r2)
    r1_out, r2_out, html_out, json_out = _fastp_outputs(report_dir, rec.name())

    if (
        not force
        and r1_out.exists()
        and r2_out.exists()
        and html_out.exists()
        and json_out.exists()
    ):
        msg = f"fastp cache hit for {rec.name()}"
        logger.info(msg)
        if progress_cb is not None:
            progress_cb(1.0)
        return StepResult("fastp", rec.name(), True, time.time() - start, msg)

    cmd = [
        "fastp",
        "-i", str(r1_in),
        "-I", str(r2_in),
        "-o", str(r1_out),
        "-O", str(r2_out),
        "--detect_adapter_for_pe",
        "--html", str(html_out),
        "--json", str(json_out),
        "--thread", str(threads),
    ]
    try:
        rc, tail = _run_tee(cmd, log_file, progress_cb, total_expected=200)
    except FileNotFoundError:
        return StepResult(
            "fastp", rec.name(), False, time.time() - start,
            "fastp not found on PATH — install via bioconda or uncheck 'Run fastp'."
        )
    if rc != 0:
        return StepResult(
            "fastp", rec.name(), False, time.time() - start,
            f"fastp failed (rc={rc}): {tail[-500:]}"
        )
    return StepResult("fastp", rec.name(), True, time.time() - start, "ok")


def run_salmon(
    rec: SampleRecord,
    salmon_index: Path,
    tx2gene: Path,
    output_dir: Path,
    threads: int,
    libtype: str,
    force: bool,
    log_file: Path,
    use_trimmed: bool,
    report_dir: Path,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> StepResult:
    start = time.time()
    quant_dir = output_dir / "salmon" / rec.name()
    quant_sf = quant_dir / "quant.sf"
    quant_genes_sf = quant_dir / "quant.genes.sf"

    if not force and quant_sf.exists() and quant_genes_sf.exists():
        msg = f"salmon cache hit for {rec.name()}"
        logger.info(msg)
        if progress_cb is not None:
            progress_cb(1.0)
        return StepResult("salmon", rec.name(), True, time.time() - start, msg)

    if use_trimmed:
        r1 = report_dir / "trimmed" / f"{rec.name()}_R1.trim.fastq.gz"
        r2 = report_dir / "trimmed" / f"{rec.name()}_R2.trim.fastq.gz"
    else:
        r1 = Path(rec.fastq_r1)
        r2 = Path(rec.fastq_r2)

    quant_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "salmon", "quant",
        "-i", str(salmon_index),
        "-l", libtype,
        "-1", str(r1),
        "-2", str(r2),
        "-p", str(threads),
        "--gcBias",
        "--seqBias",
        "-g", str(tx2gene),
        "-o", str(quant_dir),
    ]
    try:
        rc, tail = _run_tee(cmd, log_file, progress_cb, total_expected=400)
    except FileNotFoundError:
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            "salmon not found on PATH — install via bioconda."
        )
    if rc != 0:
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            f"salmon failed (rc={rc}): {tail[-500:]}"
        )
    if not quant_sf.exists():
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            f"salmon produced no quant.sf in {quant_dir}"
        )
    return StepResult("salmon", rec.name(), True, time.time() - start, "ok")


# ----------------------------------------------------------------------
# Top-level pipeline
# ----------------------------------------------------------------------
def run_pipeline(
    records: list[SampleRecord],
    *,
    salmon_index: Path,
    tx2gene: Path,
    output_dir: Path,
    report_dir: Path,
    threads: int,
    run_fastp_step: bool,
    libtype: str,
    force: bool,
    progress_cb: Optional[RunCallback] = None,
    dry_run: bool = False,
) -> list[StepResult]:
    """Run fastp (optional) + salmon across the supplied records.

    The progress callback composes per-sample progress with per-step
    progress: ``(sample_idx - 1 + inner_fraction) / n_samples``. The
    inner fraction itself is ``(step_idx + step_fraction) / total_steps``
    so the bar advances smoothly through each sample.
    """
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        env = check_environment()
        logger.info("dry-run environment check: %s", env)
        if progress_cb is not None:
            progress_cb(0, 0, 1.0, "dry run — environment check complete")
        return []

    n_samples = max(len(records), 1)
    steps_per_sample = 2 if run_fastp_step else 1
    results: list[StepResult] = []

    for s_idx, rec in enumerate(records):
        sample_log = report_dir / f"{rec.name()}.log"
        sample_log.parent.mkdir(parents=True, exist_ok=True)

        def _inner(step_idx: int, step_fraction: float, label: str) -> None:
            inner = (step_idx + step_fraction) / steps_per_sample
            outer = (s_idx + inner) / n_samples
            if progress_cb is not None:
                progress_cb(s_idx, step_idx, outer, f"{rec.name()}: {label}")

        current_step = 0
        if run_fastp_step:
            step_idx = current_step
            _inner(step_idx, 0.0, "fastp starting")
            res = run_fastp(
                rec,
                report_dir,
                threads=threads,
                force=force,
                log_file=sample_log,
                progress_cb=lambda f, si=step_idx: _inner(si, f, "fastp"),
            )
            results.append(res)
            _inner(step_idx, 1.0, f"fastp {'ok' if res.ok else 'FAILED'}")
            if not res.ok:
                logger.error("fastp failed for %s: %s", rec.name(), res.message)
                continue
            current_step += 1

        step_idx = current_step
        _inner(step_idx, 0.0, "salmon starting")
        res = run_salmon(
            rec,
            salmon_index=Path(salmon_index),
            tx2gene=Path(tx2gene),
            output_dir=output_dir,
            threads=threads,
            libtype=libtype,
            force=force,
            log_file=sample_log,
            use_trimmed=run_fastp_step,
            report_dir=report_dir,
            progress_cb=lambda f, si=step_idx: _inner(si, f, "salmon"),
        )
        results.append(res)
        _inner(step_idx, 1.0, f"salmon {'ok' if res.ok else 'FAILED'}")
        if not res.ok:
            logger.error("salmon failed for %s: %s", rec.name(), res.message)

    if progress_cb is not None:
        progress_cb(n_samples - 1, steps_per_sample - 1, 1.0, "pipeline complete")
    return results
