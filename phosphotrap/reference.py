"""GENCODE reference downloader + salmon index builder.

The whole point of this module is to make the "where do I get a salmon
index?" workflow a one-button operation in the Streamlit app instead of
a fight with curl / zcat / awk / sed across macOS, Linux, and whatever
shell the user happens to be running.

Everything is pure Python except the final ``salmon index`` shell-out:

* Downloads use ``urllib.request`` with streaming + atomic rename, so a
  half-finished file never poisons a re-run.
* Decoys, gentrome, and tx2gene are all built from the gzipped GENCODE
  files directly via the ``gzip`` module — no ``zcat`` (which on macOS
  is the legacy ``.Z``-only tool, not GNU zcat).
* Salmon is invoked the same way ``pipeline.py`` invokes it: subprocess
  with stdout teed to a log file and a progress callback driven by line
  count.

The orchestrator ``build_reference`` composes a single 0-to-1 progress
fraction across all seven stages so the Streamlit progress bar
advances smoothly through:

    download transcriptome -> download genome -> download GTF ->
    decoys -> gentrome -> salmon index -> tx2gene

(Downloads are split into three stages so the genome's ~700 MB
dominates the bar's movement during the slowest step.)

Caching is per-file: every step skips its work if its output already
exists with non-zero size and ``force=False``. Re-running the build
after a successful run is effectively a no-op.
"""

from __future__ import annotations

import gzip
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .logger import get_logger

logger = get_logger()

# Progress callback signature. Fraction is in [0, 1]; message is a short
# user-facing label. Matches the convention used by pipeline.run_pipeline
# so the same Streamlit progress-bar wiring works for both.
ProgressCB = Callable[[float, str], None]

# Idle timeout for download reads. ``urllib.request.urlopen(req, timeout=N)``
# applies ``N`` to EACH blocking socket operation (connect + each recv),
# not to the total download wall-clock. So a sustained-slow download
# still completes: each chunk just needs to arrive inside this window.
# A completely stalled connection (no bytes at all) raises
# ``socket.timeout`` after this many seconds and ``download_file``
# re-raises it as a user-facing ``RuntimeError`` with a clearer message.
#
# 120 seconds is the sweet spot: long enough that hotel wifi burps
# don't kill a legitimate download, short enough that the user isn't
# staring at a frozen progress bar for half an hour before being told
# the server stalled.
_DOWNLOAD_IDLE_TIMEOUT_S = 120

# Progress-bar ramp denominator for ``salmon index``. Same arbitrary
# line-count fake as the pipeline runner uses for fastp / salmon quant
# — salmon index emits hundreds of progress lines but no parseable
# total. Centralised so it isn't a magic number at the call site.
_SALMON_INDEX_PROGRESS_LINES = 800


# ----------------------------------------------------------------------
# Release metadata
# ----------------------------------------------------------------------
# We hard-code the GENCODE mouse layout because the file naming is
# stable across releases — only the ``REL`` token changes. If GENCODE
# ever renames a file, the user can override the URL via the Streamlit
# field rather than waiting for a code change.

GENCODE_MOUSE_BASE = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_{rel}"
)

# Default release. Bump this when a newer one ships; the UI lets the
# user override it without editing code.
DEFAULT_GENCODE_MOUSE_RELEASE = "M38"


@dataclass(frozen=True)
class GencodeFiles:
    """Resolved download URLs and local filenames for a GENCODE release."""

    release: str
    transcripts_url: str
    genome_url: str
    gtf_url: str
    transcripts_name: str
    genome_name: str
    gtf_name: str

    @classmethod
    def for_mouse(cls, release: str) -> "GencodeFiles":
        rel = release.strip()
        if not rel:
            raise ValueError("release must be non-empty (e.g. 'M38')")
        base = GENCODE_MOUSE_BASE.format(rel=rel)
        t = f"gencode.v{rel}.transcripts.fa.gz"
        g = "GRCm39.primary_assembly.genome.fa.gz"
        a = f"gencode.v{rel}.primary_assembly.annotation.gtf.gz"
        return cls(
            release=rel,
            transcripts_url=f"{base}/{t}",
            genome_url=f"{base}/{g}",
            gtf_url=f"{base}/{a}",
            transcripts_name=t,
            genome_name=g,
            gtf_name=a,
        )


@dataclass
class ReferenceArtifacts:
    """Paths produced by a successful ``build_reference`` run."""

    index_dir: Path
    tx2gene_tsv: Path
    transcripts_fa: Path
    genome_fa: Path
    gtf: Path
    decoys: Path
    gentrome: Path
    n_transcripts: int
    # Number of transcripts in the transcriptome FASTA (the salmon
    # index's universe) and fraction of those that appear in the
    # tx2gene.tsv file. A value < 0.99 means salmon quant will emit
    # per-missing-transcript warnings at runtime and drop those
    # transcripts out of the gene-level ``quant.genes.sf`` output.
    n_fasta_transcripts: int = 0
    tx2gene_coverage: float = 0.0


# ----------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------
def download_file(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    progress_cb: Optional[ProgressCB] = None,
    chunk_size: int = 1024 * 1024,
    label: Optional[str] = None,
) -> Path:
    """Stream ``url`` to ``dest``. Skip if ``dest`` already exists.

    Writes to ``dest.with_suffix(dest.suffix + '.partial')`` first and
    only renames on a clean read, so an interrupted download never
    leaves a corrupt file in place.
    """
    dest = Path(dest)
    label = label or dest.name

    if not force and dest.exists() and dest.stat().st_size > 0:
        if progress_cb is not None:
            progress_cb(1.0, f"{label} (cached)")
        logger.info("download cache hit: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()

    logger.info("downloading %s -> %s", url, dest)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "phosphotrap-reference/1.0"}
        )
        with urllib.request.urlopen(
            req, timeout=_DOWNLOAD_IDLE_TIMEOUT_S
        ) as resp:
            total = int(resp.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            last_emit = 0.0
            with tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None and (
                        time.monotonic() - last_emit > 0.2
                    ):
                        last_emit = time.monotonic()
                        if total > 0:
                            frac = min(0.99, downloaded / total)
                            mb = downloaded / 1_000_000
                            tot_mb = total / 1_000_000
                            progress_cb(
                                frac, f"{label}: {mb:.0f}/{tot_mb:.0f} MB"
                            )
                        else:
                            mb = downloaded / 1_000_000
                            progress_cb(0.5, f"{label}: {mb:.0f} MB")
        tmp.replace(dest)
    except socket.timeout as exc:
        # Idle stall: the server accepted the connection but stopped
        # sending data for longer than ``_DOWNLOAD_IDLE_TIMEOUT_S``.
        # Bubble up with a specific message so the user knows it's
        # a network issue, not a path issue.
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"download stalled for {url}: no bytes received for "
            f"{_DOWNLOAD_IDLE_TIMEOUT_S}s. Check your network "
            f"connection and retry."
        ) from exc
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        # The urllib errors wrap socket.timeout as ``URLError(reason=...)``
        # in some Python versions, so we peek inside and emit the same
        # "stalled" message when that's the case.
        if tmp.exists():
            tmp.unlink()
        reason = getattr(exc, "reason", None)
        if isinstance(reason, socket.timeout):
            raise RuntimeError(
                f"download stalled for {url}: no bytes received for "
                f"{_DOWNLOAD_IDLE_TIMEOUT_S}s. Check your network "
                f"connection and retry."
            ) from exc
        raise RuntimeError(f"download failed for {url}: {exc}") from exc

    if progress_cb is not None:
        progress_cb(1.0, f"{label} (done)")
    return dest


# ----------------------------------------------------------------------
# Decoys + gentrome
# ----------------------------------------------------------------------
def build_decoys(genome_fa_gz: Path, dest: Path) -> int:
    """Write decoy contig names (one per line) from a gzipped genome FASTA.

    Returns the number of decoy contigs written. This is pure Python so
    it doesn't depend on ``zgrep``, which on macOS is GNU-incompatible.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(genome_fa_gz, "rt") as src, dest.open("w") as out:
        for line in src:
            if line.startswith(">"):
                # ">chr1 some description" -> "chr1"
                name = line[1:].split()[0] if len(line) > 1 else ""
                if name:
                    out.write(name + "\n")
                    n += 1
    logger.info("wrote %d decoy contig names to %s", n, dest)
    return n


def count_fasta_transcripts(transcripts_fa_gz: Path) -> int:
    """Count ``>`` records in a gzipped transcriptome FASTA.

    Used by the post-build sanity check to compute tx2gene coverage:
    ``n_tx2gene_rows / n_fasta_transcripts``. A fraction below 0.99
    means salmon quant will warn about missing transcript->gene
    entries at runtime.
    """
    n = 0
    with gzip.open(transcripts_fa_gz, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def tx2gene_transcript_ids(tx2gene_tsv: Path) -> set[str]:
    """Return the set of transcript IDs (column 1) in a tx2gene TSV."""
    out: set[str] = set()
    with Path(tx2gene_tsv).open("r") as fh:
        for line in fh:
            if not line:
                continue
            tid = line.split("\t", 1)[0].strip()
            if tid:
                out.add(tid)
    return out


def fasta_transcript_ids(
    transcripts_fa_gz: Path, *, gencode: bool = True
) -> set[str]:
    """Return the set of transcript IDs in a gzipped transcriptome FASTA.

    When ``gencode`` is True (the default), splits ``|``-delimited
    GENCODE-style headers on the first ``|`` just like
    ``salmon index --gencode`` does internally, so the returned IDs
    match what the salmon index stores.
    """
    out: set[str] = set()
    with gzip.open(transcripts_fa_gz, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].split()[0] if len(line) > 1 else ""
            if gencode and "|" in header:
                header = header.split("|", 1)[0]
            if header:
                out.add(header)
    return out


def build_gentrome(
    transcripts_fa_gz: Path,
    genome_fa_gz: Path,
    dest: Path,
) -> Path:
    """Concatenate transcriptome + genome into a single gzipped file.

    gzip streams concatenate at the byte level, so we don't need to
    decompress and recompress — this is just ``cat`` in Python.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for src in (transcripts_fa_gz, genome_fa_gz):
            with Path(src).open("rb") as fh:
                shutil.copyfileobj(fh, out, length=4 * 1024 * 1024)
    logger.info("wrote gentrome %s (%d bytes)", dest, dest.stat().st_size)
    return dest


# ----------------------------------------------------------------------
# tx2gene
# ----------------------------------------------------------------------
_TID_RE = re.compile(r'transcript_id "([^"]+)"')
_GID_RE = re.compile(r'gene_id "([^"]+)"')
_GNAME_RE = re.compile(r'gene_name "([^"]+)"')


def build_tx2gene(
    gtf_gz: Path,
    dest: Path,
    *,
    progress_cb: Optional[ProgressCB] = None,
) -> int:
    """Build a 3-column ``transcript_id<TAB>gene_id<TAB>gene_name`` TSV.

    Reads the gzipped GTF directly. We only emit one row per
    ``feature == "transcript"`` line — the GTF has many other feature
    types (exon, CDS, UTR, gene) we don't want.

    If a transcript has no ``gene_name`` attribute (rare, mostly happens
    on novel pseudogene predictions) we fall back to ``gene_id`` so
    ``tximport`` still has a 3-column file with no gaps.

    Returns the number of transcripts written.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # GENCODE mouse M38 has ~150k transcript lines. Use that as the
    # denominator for the fake progress ramp; any newer release will
    # still finish at roughly 1.0 since we clamp.
    expected_transcripts = 150_000

    n = 0
    with gzip.open(gtf_gz, "rt") as src, dest.open("w") as out:
        for line in src:
            if not line or line.startswith("#"):
                continue
            # Split on tab so we can quickly check feature type before
            # running the regex on the (long) attribute string.
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "transcript":
                continue
            attrs = cols[8]
            tid_m = _TID_RE.search(attrs)
            gid_m = _GID_RE.search(attrs)
            if not tid_m or not gid_m:
                continue
            tid = tid_m.group(1)
            gid = gid_m.group(1)
            gname_m = _GNAME_RE.search(attrs)
            gname = gname_m.group(1) if gname_m else gid
            out.write(f"{tid}\t{gid}\t{gname}\n")
            n += 1
            if progress_cb is not None and n % 5_000 == 0:
                frac = min(0.99, n / expected_transcripts)
                progress_cb(frac, f"tx2gene: {n} transcripts")

    if progress_cb is not None:
        progress_cb(1.0, f"tx2gene: {n} transcripts")
    logger.info("wrote tx2gene %s with %d rows", dest, n)
    return n


# ----------------------------------------------------------------------
# Salmon index
# ----------------------------------------------------------------------
def _index_already_built(index_dir: Path) -> bool:
    """Heuristic cache check: an index dir with ``info.json`` is done."""
    return (index_dir / "info.json").exists()


def build_salmon_index(
    gentrome: Path,
    decoys: Path,
    index_dir: Path,
    *,
    threads: int = 4,
    kmer: int = 31,
    gencode: bool = True,
    force: bool = False,
    log_file: Optional[Path] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> Path:
    """Run ``salmon index`` and return the resulting directory.

    Skip-if-cached: if ``index_dir/info.json`` exists and ``force`` is
    False, the existing index is returned without re-running salmon.
    """
    index_dir = Path(index_dir)
    if not force and _index_already_built(index_dir):
        if progress_cb is not None:
            progress_cb(1.0, f"salmon index (cached): {index_dir.name}")
        logger.info("salmon index cache hit: %s", index_dir)
        return index_dir

    if shutil.which("salmon") is None:
        raise RuntimeError(
            "salmon not found on PATH. Activate the project conda env "
            "(`conda activate phosphotrap`) or install salmon via bioconda."
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "salmon", "index",
        "-t", str(gentrome),
        "-d", str(decoys),
        "-i", str(index_dir),
        "-k", str(kmer),
        "-p", str(threads),
    ]
    if gencode:
        cmd.append("--gencode")

    log_file = log_file or (index_dir.parent / f"{index_dir.name}.salmon-index.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("running: %s", " ".join(cmd))

    # ``salmon index`` emits hundreds of progress lines; use a fake
    # line-count ramp like pipeline._run_tee does for salmon quant.
    expected_lines = _SALMON_INDEX_PROGRESS_LINES
    n_lines = 0
    last_emit = 0.0
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n# {' '.join(cmd)}\n")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "salmon not found on PATH (FileNotFoundError)"
            ) from exc

        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            lf.write(line)
            tail.append(line)
            if len(tail) > 100:
                tail.pop(0)
            n_lines += 1
            if progress_cb is not None and (time.monotonic() - last_emit > 0.2):
                last_emit = time.monotonic()
                frac = min(0.99, n_lines / expected_lines)
                progress_cb(frac, f"salmon index: {n_lines} lines")
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"salmon index failed (rc={rc}). Tail:\n{''.join(tail)[-2000:]}"
        )
    if not _index_already_built(index_dir):
        raise RuntimeError(
            f"salmon index ran (rc=0) but {index_dir}/info.json is missing."
        )
    if progress_cb is not None:
        progress_cb(1.0, f"salmon index built: {index_dir.name}")
    logger.info("salmon index built: %s", index_dir)
    return index_dir


# ----------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------
# Stage weights for the composite progress bar. Roughly matches wall-
# clock proportions on a typical laptop with a fast connection: the
# downloads and the salmon index dominate; decoys / gentrome / tx2gene
# are bookkeeping. They sum to 1.0.
_STAGE_WEIGHTS = {
    "download_transcripts": 0.04,
    "download_genome": 0.20,
    "download_gtf": 0.04,
    "decoys": 0.01,
    "gentrome": 0.05,
    "salmon_index": 0.60,
    "tx2gene": 0.06,
}
assert abs(sum(_STAGE_WEIGHTS.values()) - 1.0) < 1e-6


def _stage_callback(
    progress_cb: Optional[ProgressCB],
    completed_before: float,
    stage_weight: float,
) -> Optional[ProgressCB]:
    """Wrap a progress_cb so a 0..1 stage maps into a slice of [0, 1]."""
    if progress_cb is None:
        return None

    def inner(frac: float, msg: str) -> None:
        outer = completed_before + max(0.0, min(1.0, frac)) * stage_weight
        progress_cb(min(1.0, outer), msg)

    return inner


def build_reference(
    *,
    release: str = DEFAULT_GENCODE_MOUSE_RELEASE,
    dest_dir: Path,
    threads: int = 4,
    force: bool = False,
    progress_cb: Optional[ProgressCB] = None,
) -> ReferenceArtifacts:
    """Download GENCODE mouse and build a salmon index + tx2gene.

    All artifacts land under ``dest_dir``. Re-running with the same
    ``dest_dir`` is a no-op when every output already exists. Set
    ``force=True`` to rebuild the salmon index from scratch (downloads
    are still cached unless their files are missing).
    """
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = GencodeFiles.for_mouse(release)
    transcripts_fa = dest_dir / files.transcripts_name
    genome_fa = dest_dir / files.genome_name
    gtf = dest_dir / files.gtf_name
    decoys = dest_dir / "decoys.txt"
    gentrome = dest_dir / "gentrome.fa.gz"
    index_dir = dest_dir / f"salmon_index_gencode_{release}"
    tx2gene = dest_dir / "tx2gene.tsv"
    log_file = dest_dir / "build_reference.log"

    completed = 0.0

    def _stage(name: str) -> Optional[ProgressCB]:
        return _stage_callback(progress_cb, completed, _STAGE_WEIGHTS[name])

    # 1. Downloads (3 files, weighted by approximate size).
    download_file(
        files.transcripts_url, transcripts_fa, force=False,
        progress_cb=_stage("download_transcripts"),
        label=f"transcriptome ({files.release})",
    )
    completed += _STAGE_WEIGHTS["download_transcripts"]

    download_file(
        files.genome_url, genome_fa, force=False,
        progress_cb=_stage("download_genome"),
        label="genome (GRCm39)",
    )
    completed += _STAGE_WEIGHTS["download_genome"]

    download_file(
        files.gtf_url, gtf, force=False,
        progress_cb=_stage("download_gtf"),
        label=f"GTF ({files.release})",
    )
    completed += _STAGE_WEIGHTS["download_gtf"]

    # 2. Decoys.
    if force or not decoys.exists() or decoys.stat().st_size == 0:
        n_decoys = build_decoys(genome_fa, decoys)
        if progress_cb is not None:
            progress_cb(
                completed + _STAGE_WEIGHTS["decoys"],
                f"decoys: {n_decoys} contigs",
            )
    completed += _STAGE_WEIGHTS["decoys"]

    # 3. gentrome.
    need_gentrome = (
        force
        or not gentrome.exists()
        or gentrome.stat().st_size
        < (transcripts_fa.stat().st_size + genome_fa.stat().st_size) * 0.95
    )
    if need_gentrome:
        if progress_cb is not None:
            progress_cb(completed, "concatenating transcriptome + genome")
        build_gentrome(transcripts_fa, genome_fa, gentrome)
    completed += _STAGE_WEIGHTS["gentrome"]
    if progress_cb is not None:
        progress_cb(completed, "gentrome ready")

    # 4. Salmon index (the big one).
    build_salmon_index(
        gentrome,
        decoys,
        index_dir,
        threads=threads,
        kmer=31,
        gencode=True,
        force=force,
        log_file=log_file,
        progress_cb=_stage("salmon_index"),
    )
    completed += _STAGE_WEIGHTS["salmon_index"]

    # 5. tx2gene.
    if force or not tx2gene.exists() or tx2gene.stat().st_size == 0:
        n_tx = build_tx2gene(gtf, tx2gene, progress_cb=_stage("tx2gene"))
    else:
        # Count rows for the return value without rebuilding.
        with tx2gene.open("r") as fh:
            n_tx = sum(1 for _ in fh)
        if progress_cb is not None:
            progress_cb(
                completed + _STAGE_WEIGHTS["tx2gene"],
                f"tx2gene cached: {n_tx} transcripts",
            )
    completed += _STAGE_WEIGHTS["tx2gene"]

    # 6. Coverage sanity check. Compare the transcript IDs in the
    #    transcriptome FASTA (the salmon index's universe) against the
    #    transcript IDs in tx2gene.tsv. If coverage is below 99%, salmon
    #    quant will emit per-missing-transcript warnings at runtime and
    #    those transcripts will fall through to "transcript as its own
    #    gene" — they'll still be quantified, but downstream tximport
    #    won't aggregate them into gene-level counts correctly.
    try:
        fasta_ids = fasta_transcript_ids(transcripts_fa, gencode=True)
        tx2g_ids = tx2gene_transcript_ids(tx2gene)
        n_fasta = len(fasta_ids)
        if n_fasta > 0:
            missing = fasta_ids - tx2g_ids
            coverage = 1.0 - (len(missing) / n_fasta)
        else:
            coverage = 0.0
        if coverage < 0.99:
            logger.warning(
                "tx2gene coverage %.2f%% (%d / %d FASTA transcripts "
                "found in tx2gene.tsv). Missing %d — salmon quant "
                "will warn on those at runtime.",
                coverage * 100,
                n_fasta - len(missing),
                n_fasta,
                len(missing),
            )
        else:
            logger.info(
                "tx2gene coverage %.2f%% (%d / %d)",
                coverage * 100,
                n_fasta - len(missing),
                n_fasta,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("coverage check failed: %s", exc)
        n_fasta = 0
        coverage = 0.0

    if progress_cb is not None:
        progress_cb(1.0, "reference build complete")

    return ReferenceArtifacts(
        index_dir=index_dir,
        tx2gene_tsv=tx2gene,
        transcripts_fa=transcripts_fa,
        genome_fa=genome_fa,
        gtf=gtf,
        decoys=decoys,
        gentrome=gentrome,
        n_transcripts=n_tx,
        n_fasta_transcripts=n_fasta,
        tx2gene_coverage=coverage,
    )
