"""anota2seq shell-out runner for per-contrast differential translation.

anota2seq is the primary analysis for this app. Because it's an R
package, we generate a small Rscript, pass it the contrast-specific
sample list through a JSON spec, and parse the three output TSVs from
a scratch directory per contrast.

The three output categories are the anota2seq *regulatory modes*
produced by :func:`anota2seqRegModes` and retrieved via
``anota2seqGetOutput(ads, output = "regModes", analysis = ...)``:

* ``translation``     — IP changes without a matching INPUT change
* ``buffering``       — INPUT changes but IP compensates
* ``mRNA_abundance``  — both IP and INPUT change coherently
  (this is anota2seq's taxonomy for the "both change" case; the
  user-facing prompt called it "mRNA+translation" in loose language)

Graceful degradation: every call path that touches R is wrapped in
``try/except FileNotFoundError`` so the UI stays up when Rscript, r-base,
or the Bioconductor packages aren't installed.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AppConfig, resolve_rscript
from .logger import get_logger
from .samples import Pair, SampleRecord, records_for_contrast

logger = get_logger()


@dataclass
class Anota2seqResult:
    contrast: str
    ok: bool
    message: str
    # Three anota2seq regulatory-mode categories
    translation: pd.DataFrame     # IP changes, INPUT doesn't
    buffering: pd.DataFrame       # INPUT changes, IP compensates
    mrna_abundance: pd.DataFrame  # both change coherently
    scratch_dir: Path


RSCRIPT_TEMPLATE = r"""
suppressMessages({
  library(tximport)
  library(anota2seq)
})

args <- commandArgs(trailingOnly = TRUE)
spec_path <- args[1]
out_dir   <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

spec <- jsonlite::fromJSON(spec_path)

tx2g_raw <- read.table(spec$tx2gene, sep = "\t", header = FALSE,
                       stringsAsFactors = FALSE)
tx2g <- tx2g_raw[, 1:2]
colnames(tx2g) <- c("TXNAME", "GENEID")

# Build a gene_id -> gene_name map from the 3rd column (if present)
# so the output TSVs carry human-readable symbols alongside the
# Ensembl IDs that tximport aggregates on. Falls back gracefully for
# legacy 2-column tx2gene files.
if (ncol(tx2g_raw) >= 3) {
  gene_name_map <- tapply(
    tx2g_raw[, 3], tx2g_raw[, 2],
    function(x) x[!is.na(x) & nzchar(x)][1]
  )
} else {
  gene_name_map <- setNames(character(0), character(0))
}

ip_files    <- spec$ip_files
input_files <- spec$input_files
names(ip_files)    <- spec$ip_names
names(input_files) <- spec$input_names

txi_ip    <- tximport(ip_files,    type = "salmon", tx2gene = tx2g)
txi_input <- tximport(input_files, type = "salmon", tx2gene = tx2g)

dataP <- txi_ip$counts
dataT <- txi_input$counts

# Align gene order between the two matrices.
shared <- intersect(rownames(dataP), rownames(dataT))
dataP <- dataP[shared, , drop = FALSE]
dataT <- dataT[shared, , drop = FALSE]

phenoVec <- spec$phenoVec

# NB: anota2seq's ``normalize`` is a logical (TRUE/FALSE) flag that
# controls *whether* to normalize — the *method* goes in
# ``transformation``. Passing the method name here ("TMM-log") trips
# anota2seqCheckParameter with "normalize parameter must be set to
# TRUE or FALSE" and aborts before any analysis runs. The
# ``transformation`` argument only accepts "rlog" or "TMM-log2"
# (anota2seqCheckParameter hard-codes that whitelist); "TMM-log" is
# *not* accepted and aborts with "transformation parameter must be
# either rlog or TMM-log2".
ads <- anota2seqDataSetFromMatrix(
  dataP     = dataP,
  dataT     = dataT,
  phenoVec  = phenoVec,
  dataType  = "RNAseq",
  normalize = TRUE,
  transformation = "TMM-log2",
  filterZeroGenes = TRUE,
  varCutOff = NULL
)

ads <- anota2seqAnalyze(
  ads,
  analysis = c("translation", "buffering", "translated mRNA", "total mRNA")
)

ads <- anota2seqSelSigGenes(
  ads,
  selDeltaPT          = spec$selDeltaPT,
  selDeltaTP          = spec$selDeltaTP,
  maxPAdj             = spec$maxPAdj,
  minSlopeTranslation = spec$minSlopeTranslation,
  maxSlopeTranslation = spec$maxSlopeTranslation,
  minSlopeBuffering   = spec$minSlopeBuffering,
  maxSlopeBuffering   = spec$maxSlopeBuffering,
  selContrast         = 1
)

# anota2seqRegModes partitions genes into translation / buffering /
# mRNA abundance / background. All three we care about are retrieved
# via output = "regModes" with the matching analysis label.
ads <- anota2seqRegModes(ads)

dump_one <- function(df, name) {
  if (is.null(df)) {
    df <- data.frame()
  } else if (!is.data.frame(df)) {
    df <- as.data.frame(df)
  }
  # tximport aggregates on Ensembl gene_id, so the rownames are
  # ENSMUSG.../ENSG... — add a ``gene_name`` column joined from the
  # tx2gene mapping so downstream TSVs carry MGI / HGNC symbols. Put
  # gene_id + gene_name first for readability.
  gene_ids <- rownames(df)
  if (length(gene_ids) > 0) {
    df$gene_id <- gene_ids
    if (length(gene_name_map) > 0) {
      df$gene_name <- unname(gene_name_map[gene_ids])
      df$gene_name[is.na(df$gene_name)] <- gene_ids[is.na(df$gene_name)]
    } else {
      df$gene_name <- gene_ids
    }
    lead <- c("gene_id", "gene_name")
    df <- df[, c(lead, setdiff(colnames(df), lead)), drop = FALSE]
  } else {
    df$gene_id <- character(0)
    df$gene_name <- character(0)
  }
  path <- file.path(out_dir, paste0(name, ".tsv"))
  write.table(df, file = path, sep = "\t", row.names = FALSE,
              quote = FALSE)
}

translation <- tryCatch(
  anota2seqGetOutput(ads, output = "regModes",
                     analysis = "translation", selContrast = 1),
  error = function(e) NULL
)
buffering <- tryCatch(
  anota2seqGetOutput(ads, output = "regModes",
                     analysis = "buffering", selContrast = 1),
  error = function(e) NULL
)
mrna_abundance <- tryCatch(
  anota2seqGetOutput(ads, output = "regModes",
                     analysis = "mRNA abundance", selContrast = 1),
  error = function(e) NULL
)

dump_one(translation,    "translation")
dump_one(buffering,      "buffering")
dump_one(mrna_abundance, "mrna_abundance")

# Sentinel file used by the Python-side cache-hit check. Only written
# AFTER dump_one has completed for all three regmode tables, so any
# partial / interrupted / R-crashed run leaves the scratch dir without
# a .done marker and the next invocation refuses the cache. The file
# contents are an ISO-ish timestamp for human debugging; the Python
# side only checks existence.
writeLines(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
           file.path(out_dir, ".done"))

cat("ok\n")
"""


def _build_spec(
    contrast_pairs: list[Pair],
    salmon_root: Path,
    tx2gene: Path,
    alt_group: str,
    ref_group: str,
    cfg: AppConfig,
) -> dict:
    """Construct the Rscript input spec as a plain dict.

    Kept separate from the on-disk write so we can compare the
    in-memory fresh spec against whatever is cached in the scratch
    directory *before* deciding whether to skip the R invocation.
    """
    # Order: ref group first, then alt — matches records_for_contrast().
    ips = [p.ip for p in contrast_pairs]
    ins = [p.input for p in contrast_pairs]

    tx2gene_path = Path(tx2gene)
    try:
        tx2gene_mtime = tx2gene_path.stat().st_mtime
    except OSError:
        # Missing file — emit a sentinel that still participates in
        # cache invalidation (a later rebuild will have a real mtime
        # and mismatch this value). The R script will surface the
        # missing-file error itself.
        tx2gene_mtime = -1.0

    return {
        # Spec schema version — bump this whenever the RSCRIPT_TEMPLATE
        # output schema changes so existing caches are force-rerun.
        # v1 = initial schema with gene_id + gene_name regmode TSVs and
        # the .done sentinel marker. Kept in sync with the DESeq2
        # runner's own _schema_version field.
        "_schema_version": 1,
        "tx2gene": str(tx2gene),
        # Content-drift guard: captures in-place reference rebuilds
        # that leave the path unchanged but replace the gene_id ->
        # gene_name mapping (e.g., GENCODE M38 -> M39). Without this,
        # the cache-hit path would silently return stale regmode TSVs
        # where gene_name lookups had been performed against the old
        # tx2gene contents — exactly the "gene_name looks like
        # gene_id" symptom we kept chasing.
        "tx2gene_mtime": tx2gene_mtime,
        "ip_files": [str(Path(salmon_root) / r.name() / "quant.sf") for r in ips],
        "ip_names": [r.name() for r in ips],
        "input_files": [str(Path(salmon_root) / r.name() / "quant.sf") for r in ins],
        "input_names": [r.name() for r in ins],
        # anota2seq phenoVec is applied to dataP (IP) columns
        "phenoVec": [r.group for r in ips],
        "selDeltaPT": cfg.anota_delta_pt,
        "selDeltaTP": cfg.anota_delta_tp,
        "maxPAdj": cfg.anota_max_padj,
        "minSlopeTranslation": cfg.anota_min_slope_trans,
        "maxSlopeTranslation": cfg.anota_max_slope_trans,
        "minSlopeBuffering": cfg.anota_min_slope_buff,
        "maxSlopeBuffering": cfg.anota_max_slope_buff,
    }


def _serialise_spec(spec: dict) -> str:
    """Canonical JSON serialisation used for cache comparison."""
    return json.dumps(spec, indent=2, sort_keys=True)


_ANOTA2SEQ_OUTPUT_NAMES = ("translation", "buffering", "mrna_abundance")
_SPEC_FILENAME = "spec.json"
# Written by the R script AFTER all three regmode TSVs have been
# dumped; used on the Python side as the ground-truth "this run
# completed cleanly" signal. A cache hit requires both the spec
# match AND this marker — file existence alone isn't enough because
# an R crash mid-write can leave truncated TSVs on disk.
_DONE_MARKER = ".done"


def _validate_and_read_outputs(scratch: Path) -> dict[str, pd.DataFrame]:
    """Strict: every expected TSV must exist, parse, and carry ``gene_id``.

    Returns a ``{name: DataFrame}`` dict on success. Raises
    :class:`RuntimeError` on the first failure so callers can treat a
    corrupt / partial output dir as a hard cache miss (on the cache-
    hit path) or a hard run failure (on the post-subprocess path)
    instead of silently returning empty DataFrames.

    A regmode table with zero significant genes is a legitimate
    anota2seq result — the R ``dump_one`` helper writes a header-only
    TSV in that case (``gene_id``, ``gene_name``, then the regular
    columns). ``pd.read_csv`` on a header-only file returns an empty
    DataFrame with the correct ``columns``, which passes the
    ``"gene_id" in df.columns`` check below. The failure signals are:
    missing file, ``pd.errors.EmptyDataError`` (truly empty file — R
    crashed before writing the header), parser error on malformed
    content, or missing ``gene_id`` header (schema drift).
    """
    out: dict[str, pd.DataFrame] = {}
    for name in _ANOTA2SEQ_OUTPUT_NAMES:
        p = scratch / f"{name}.tsv"
        if not p.exists():
            raise RuntimeError(f"expected output file missing: {p}")
        try:
            df = pd.read_csv(p, sep="\t")
        except pd.errors.EmptyDataError as exc:
            raise RuntimeError(f"output file has no header: {p}") from exc
        except Exception as exc:
            raise RuntimeError(f"failed to parse {p}: {exc}") from exc
        if "gene_id" not in df.columns:
            raise RuntimeError(
                f"{p} missing expected gene_id column "
                f"(got {list(df.columns)})"
            )
        out[name] = df
    return out


def run_anota2seq(
    records: list[SampleRecord],
    *,
    alt_group: str,
    ref_group: str,
    salmon_root: Path,
    tx2gene: Path,
    cfg: AppConfig,
    output_dir: Path,
) -> Anota2seqResult:
    """Run anota2seq for a single contrast, returning a typed result."""
    contrast = f"{alt_group}_vs_{ref_group}"
    scratch = Path(output_dir) / "anota2seq" / contrast
    scratch.mkdir(parents=True, exist_ok=True)

    contrast_pairs = records_for_contrast(
        records, alt_group=alt_group, ref_group=ref_group
    )
    if len(contrast_pairs) != 6:
        return Anota2seqResult(
            contrast=contrast,
            ok=False,
            message=(
                f"Expected 6 matched pairs (3 ref + 3 alt) for {contrast}, "
                f"got {len(contrast_pairs)}. Check sample sheet pairing."
            ),
            translation=pd.DataFrame(),
            buffering=pd.DataFrame(),
            mrna_abundance=pd.DataFrame(),
            scratch_dir=scratch,
        )

    # Build the fresh spec in memory so we can compare it against the
    # on-disk cached spec BEFORE deciding to skip. The old behaviour
    # was "skip if output TSVs exist" — which silently returned stale
    # results when the user changed thresholds on the Config tab.
    fresh_spec = _build_spec(
        contrast_pairs, Path(salmon_root), Path(tx2gene),
        alt_group, ref_group, cfg,
    )
    fresh_spec_text = _serialise_spec(fresh_spec)
    cached_spec_path = scratch / _SPEC_FILENAME

    # Skip-if-cached only when: user hasn't forced a rerun, the cached
    # spec file exists AND matches the fresh spec byte-for-byte (after
    # canonical sorted-key serialisation), the .done sentinel marker
    # is present (proves the R script finished dump_one for all three
    # regmodes, not just crashed mid-write), AND _validate_and_read_outputs
    # can actually parse every expected TSV back into a DataFrame with
    # a gene_id column. Any failure in that chain invalidates the cache
    # and falls through to rerun.
    cached_spec_text: str | None = None
    if cached_spec_path.exists():
        try:
            cached_spec_text = cached_spec_path.read_text()
        except OSError as exc:
            # File exists but can't be read — permission issue,
            # truncated JSON from an interrupted write, stale NFS
            # handle, etc. Surface it explicitly so the subsequent
            # "cache miss, rerunning" log line isn't confused with a
            # normal "no prior run" cache miss.
            logger.warning(
                "anota2seq cached spec %s exists but is unreadable "
                "(%s); treating as cache miss and rerunning",
                cached_spec_path, exc,
            )

    done_marker = scratch / _DONE_MARKER
    cache_hit = (
        not cfg.force_rerun
        and cached_spec_text == fresh_spec_text
        and done_marker.exists()
    )
    if cache_hit:
        try:
            cached = _validate_and_read_outputs(scratch)
        except RuntimeError as exc:
            # Cache dir has the spec + done marker but the TSVs are
            # corrupt / truncated / schema-drifted. Log loudly and
            # fall through to the rerun path rather than silently
            # returning empty DataFrames as success.
            logger.warning(
                "anota2seq cache hit for %s rejected, rerunning: %s",
                contrast, exc,
            )
        else:
            spec_hash = hashlib.sha256(fresh_spec_text.encode()).hexdigest()[:12]
            logger.info(
                "anota2seq cache hit for %s (spec sha256:%s)", contrast, spec_hash
            )
            return Anota2seqResult(
                contrast=contrast,
                ok=True,
                message=(
                    f"anota2seq cache hit for {contrast} "
                    f"(spec verified, sha256:{spec_hash})"
                ),
                translation=cached["translation"],
                buffering=cached["buffering"],
                mrna_abundance=cached["mrna_abundance"],
                scratch_dir=scratch,
            )

    # Cache miss — write the fresh spec and run R. We also stomp over
    # any stale output TSVs AND the .done marker so a subsequent
    # failure can't accidentally leave a mixed cache behind that the
    # next invocation would accept.
    cached_spec_path.write_text(fresh_spec_text)
    for name in _ANOTA2SEQ_OUTPUT_NAMES:
        stale = scratch / f"{name}.tsv"
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
    if done_marker.exists():
        try:
            done_marker.unlink()
        except OSError:
            pass

    r_script = scratch / "run_anota2seq.R"
    r_script.write_text(RSCRIPT_TEMPLATE)

    rscript_bin = resolve_rscript(cfg)
    cmd = [rscript_bin, str(r_script), str(cached_spec_path), str(scratch)]
    logger.info("exec: %s", " ".join(shlex.quote(c) for c in cmd))
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=60 * 60,
        )
    except FileNotFoundError:
        return Anota2seqResult(
            contrast=contrast,
            ok=False,
            message=(
                f"{rscript_bin} not found. Install r-base + bioconductor-anota2seq "
                "via the conda bootstrap, then set the Rscript path in Config."
            ),
            translation=pd.DataFrame(),
            buffering=pd.DataFrame(),
            mrna_abundance=pd.DataFrame(),
            scratch_dir=scratch,
        )

    log = scratch / "anota2seq.log"
    log.write_text((proc.stdout or "") + "\n----STDERR----\n" + (proc.stderr or ""))

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        return Anota2seqResult(
            contrast=contrast,
            ok=False,
            message=f"anota2seq Rscript failed (rc={proc.returncode}): {tail}",
            translation=pd.DataFrame(),
            buffering=pd.DataFrame(),
            mrna_abundance=pd.DataFrame(),
            scratch_dir=scratch,
        )

    # Returncode 0 is necessary but not sufficient: the R script could
    # have exited cleanly without reaching the dump_one calls (e.g.,
    # anota2seqSelSigGenes raised and the enclosing tryCatch absorbed
    # the error), leaving the scratch dir without a .done marker and/
    # or without valid output TSVs. Validate both before declaring
    # success — the previous behaviour silently returned empty
    # DataFrames as ok=True, which the UI rendered as "analysis ran,
    # no significant genes" instead of the real "analysis failed".
    if not done_marker.exists():
        return Anota2seqResult(
            contrast=contrast,
            ok=False,
            message=(
                f"anota2seq Rscript finished with rc=0 but did not "
                f"write the .done marker — check "
                f"{scratch / 'anota2seq.log'} for silent errors in "
                f"anota2seqAnalyze / anota2seqSelSigGenes / "
                f"anota2seqRegModes."
            ),
            translation=pd.DataFrame(),
            buffering=pd.DataFrame(),
            mrna_abundance=pd.DataFrame(),
            scratch_dir=scratch,
        )
    try:
        cached = _validate_and_read_outputs(scratch)
    except RuntimeError as exc:
        return Anota2seqResult(
            contrast=contrast,
            ok=False,
            message=(
                f"anota2seq Rscript finished with rc=0 but output "
                f"validation failed: {exc}"
            ),
            translation=pd.DataFrame(),
            buffering=pd.DataFrame(),
            mrna_abundance=pd.DataFrame(),
            scratch_dir=scratch,
        )

    dur = time.time() - start
    return Anota2seqResult(
        contrast=contrast,
        ok=True,
        message=f"anota2seq complete in {dur:.1f}s",
        translation=cached["translation"],
        buffering=cached["buffering"],
        mrna_abundance=cached["mrna_abundance"],
        scratch_dir=scratch,
    )
