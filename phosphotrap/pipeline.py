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

import json
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Optional

from .logger import get_logger
from .samples import SampleRecord

RunCallback = Callable[[int, int, float, str], None]

logger = get_logger()

# Wall-clock ceiling for individual fastp / salmon invocations. A
# correctly-behaved run on this design is ~5-10 minutes per sample;
# two hours is "something is catastrophically wrong, kill it and tell
# the user" territory. Without this, a hung subprocess (disk I/O
# wedged, deadlocked mmap, malformed input looping forever) would
# hang the Streamlit UI indefinitely with no way back short of
# killing the streamlit process from a terminal.
_SUBPROCESS_TIMEOUT_S = 2 * 60 * 60

# Progress-bar ramp denominators for the line-count fake. These are
# arbitrary numbers tuned so the bar fills smoothly through a typical
# fastp / salmon run on this design — not actual line counts. Salmon
# in particular has no parseable progress output, so any choice here
# is a guess. Centralised so they're not buried as magic numbers in
# call sites.
_FASTP_PROGRESS_LINES = 200
_SALMON_QUANT_PROGRESS_LINES = 400


@dataclass
class StepResult:
    step: str
    sample: str
    ok: bool
    duration_s: float
    message: str
    # Whether the failure is worth retrying without user intervention.
    # Default True — most failures we see (transient I/O, timeouts,
    # odd-state output dirs) can be cleared by rerunning with
    # force_rerun=True. The explicit False signal is reserved for
    # "the user must fix something first": binary not on PATH, a
    # mis-pointed salmon_index directory, a missing tx2gene file.
    # Loaded JSONs from pre-retryable StepResult runs are tolerated
    # via load_pipeline_results' key-filter (default True kicks in
    # for old rows). ``ok=True`` rows ignore this field.
    retryable: bool = True


# Filename used by ``save_pipeline_results`` / ``load_pipeline_results``
# to persist the Pipeline-tab results table across Streamlit sessions.
# The file lives under ``report_dir`` (same place as the pipeline logs)
# so it moves with the run and is naturally scoped to the user's config.
PIPELINE_RESULTS_FILENAME = "pipeline_results.json"


def save_pipeline_results(results: list[StepResult], report_dir: Path) -> Path:
    """Persist pipeline StepResults to a JSON file under ``report_dir``.

    Streamlit session_state is wiped when the server restarts or the
    browser tab is closed, so the Pipeline-tab results table vanishes
    even though the heavy salmon outputs are still on disk. Dumping the
    StepResult list next to the per-sample logs lets the app rehydrate
    the table on the next startup without re-running anything.

    Returns the path it wrote to. Created directories as needed. Raises
    OSError on write failure — callers are expected to log-and-swallow.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    dest = report_dir / PIPELINE_RESULTS_FILENAME
    payload = {
        "version": 1,
        "saved_at": time.time(),
        "results": [asdict(r) for r in results],
    }
    dest.write_text(json.dumps(payload, indent=2))
    return dest


def load_pipeline_results(report_dir: Path) -> list[StepResult]:
    """Rehydrate a previously-saved pipeline results list.

    Returns an empty list if the JSON file doesn't exist, is unreadable,
    or has a schema the current code doesn't understand. Never raises —
    this is called during Streamlit bootstrap and a corrupt cache file
    should not prevent the app from rendering.
    """
    src = Path(report_dir) / PIPELINE_RESULTS_FILENAME
    if not src.exists():
        return []
    try:
        payload = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not load pipeline results cache %s: %s", src, exc)
        return []
    raw = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    valid_fields = {f.name for f in fields(StepResult)}
    out: list[StepResult] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(StepResult(**{k: row[k] for k in valid_fields if k in row}))
        except TypeError:
            # Schema drift from a future version — skip the row rather
            # than erroring out of the whole load.
            continue
    return out


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
# Per-probe timeout for R package loads. A cold R startup usually
# completes well under 5 s; 10 s leaves headroom without freezing the
# UI for minutes if a package is hung.
_R_PROBE_TIMEOUT_S = 10


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
                timeout=_R_PROBE_TIMEOUT_S,
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
    timeout_s: float = _SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Run a subprocess, tee stdout+stderr to ``log_file``, return (rc, tail).

    ``progress_cb`` receives a float in [0, 1] each time a line is
    emitted. Since we cannot actually parse fastp/salmon progress lines
    reliably across versions, we fake a smooth ramp via line count: the
    callback sees ``min(0.95, n_lines / total_expected)`` on every line,
    then 1.0 once the process exits cleanly.

    ``timeout_s`` is a last-resort wall-clock ceiling. If the subprocess
    outlives it, a watchdog thread sends SIGTERM, then SIGKILL 5 seconds
    later if it's still alive. The function returns ``rc`` from the
    killed process (negative on POSIX, indicating the signal) and the
    tail includes a ``WATCHDOG: ...`` marker so the caller's StepResult
    message makes it obvious what happened.

    The watchdog runs on a daemon thread so it never prevents
    interpreter exit. The main thread continues to read stdout until
    EOF (which happens when the process terminates), so streaming
    progress callbacks keep firing right up to the kill point.
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

        # Watchdog: schedule a SIGTERM + SIGKILL if the process exceeds
        # ``timeout_s`` seconds. ``killed_by_watchdog`` is a simple
        # mutable flag the main thread reads after ``wait()`` to
        # distinguish a watchdog kill from a genuine non-zero exit.
        killed_by_watchdog = {"fired": False}

        def _watchdog() -> None:
            # time.sleep is interruptible by process exit in practice
            # because the thread is a daemon — if the subprocess
            # finishes normally, the main thread will set
            # ``proc.returncode`` and we skip the kill. The race is
            # benign: kill on an already-exited process is a no-op.
            time.sleep(timeout_s)
            if proc.poll() is None:
                killed_by_watchdog["fired"] = True
                logger.warning(
                    "watchdog: subprocess exceeded %ss, sending SIGTERM: %s",
                    timeout_s, _pretty_cmd(cmd),
                )
                try:
                    proc.terminate()
                except Exception:  # pragma: no cover - defensive
                    pass
                # Give it 5 seconds to exit cleanly, then SIGKILL.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "watchdog: SIGTERM did not stop subprocess, sending SIGKILL"
                    )
                    try:
                        proc.kill()
                    except Exception:  # pragma: no cover - defensive
                        pass

        watchdog = threading.Thread(target=_watchdog, daemon=True)
        watchdog.start()

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

        if killed_by_watchdog["fired"]:
            marker = (
                f"\nWATCHDOG: subprocess killed after exceeding "
                f"{timeout_s:.0f}s wall-clock timeout.\n"
            )
            lf.write(marker)
            tail.append(marker)

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
        rc, tail = _run_tee(
            cmd, log_file, progress_cb,
            total_expected=_FASTP_PROGRESS_LINES,
        )
    except FileNotFoundError:
        return StepResult(
            "fastp", rec.name(), False, time.time() - start,
            "fastp not found on PATH — install via bioconda or uncheck 'Run fastp'.",
            retryable=False,
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

    # Defense in depth: the Streamlit Pipeline tab already gates on
    # ``validate_reference_paths`` before letting the user click Start,
    # but ``run_salmon`` is also callable from tests and from any
    # future CLI wrapper. Reject obviously-bad ``salmon_index`` and
    # ``tx2gene`` here too so a caller that bypasses the UI gets a
    # clear error instead of a cryptic salmon failure 30 seconds in.
    if not Path(salmon_index).is_dir():
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            f"salmon_index must be a directory containing info.json, "
            f"got: {salmon_index}",
            retryable=False,
        )
    if not (Path(salmon_index) / "info.json").exists():
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            f"salmon_index directory has no info.json — not a built "
            f"index: {salmon_index}",
            retryable=False,
        )
    if not Path(tx2gene).is_file():
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            f"tx2gene must be an existing file (tx2gene.tsv), got: {tx2gene}",
            retryable=False,
        )

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
        rc, tail = _run_tee(
            cmd, log_file, progress_cb,
            total_expected=_SALMON_QUANT_PROGRESS_LINES,
        )
    except FileNotFoundError:
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            "salmon not found on PATH — install via bioconda.",
            retryable=False,
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
    # ``quant.genes.sf`` is written only when salmon can successfully
    # read the ``-g <tx2gene>`` file AND map every transcript ID in
    # the index to a gene. If tx2gene is a directory (the bug that
    # broke the first live run — see d3b838f), an empty file, or a
    # file with completely mismatched transcript IDs, salmon may
    # exit rc=0 and still fail to produce ``quant.genes.sf``.
    # Without this check, the pipeline reports success and the
    # Analysis tab fails two tabs later with "No quant.genes.sf"
    # from ``fpkm._read_quant_genes`` — a confusing error located
    # far from its actual cause.
    if not quant_genes_sf.exists():
        return StepResult(
            "salmon", rec.name(), False, time.time() - start,
            (
                f"salmon wrote quant.sf but NOT quant.genes.sf in "
                f"{quant_dir}. This usually means the tx2gene.tsv "
                f"file passed via -g was missing, a directory, "
                f"empty, or had transcript IDs that don't match the "
                f"salmon index. Check cfg.tx2gene_tsv in the Config "
                f"tab and rebuild via the Reference tab if needed."
            ),
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
    rscript_path: str = "Rscript",
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
        env = check_environment(rscript_path)
        logger.info("dry-run environment check: %s", env)
        if progress_cb is not None:
            progress_cb(0, 0, 1.0, "dry run — environment check complete")
        return []

    n_samples = max(len(records), 1)
    steps_per_sample = 2 if run_fastp_step else 1
    results: list[StepResult] = []

    # Per-sample logs are co-located with the central app log under
    # ``report_dir/logs/per-sample/`` so the Logs tab can list them
    # via a single glob instead of scanning report_dir's root.
    per_sample_log_dir = report_dir / "logs" / "per-sample"
    per_sample_log_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, rec in enumerate(records):
        sample_log = per_sample_log_dir / f"{rec.name()}.log"
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
        # Clamp indices for the empty-records case (len(records) == 0),
        # where max(len, 1) made n_samples == 1 but the loop never ran.
        final_sample = max(len(records) - 1, 0)
        final_step = max(steps_per_sample - 1, 0)
        progress_cb(final_sample, final_step, 1.0, "pipeline complete")
    return results
