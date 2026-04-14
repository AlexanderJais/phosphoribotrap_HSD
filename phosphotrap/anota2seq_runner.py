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

tx2g <- read.table(spec$tx2gene, sep = "\t", header = FALSE,
                   stringsAsFactors = FALSE)
tx2g <- tx2g[, 1:2]
colnames(tx2g) <- c("TXNAME", "GENEID")

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

ads <- anota2seqDataSetFromMatrix(
  dataP     = dataP,
  dataT     = dataT,
  phenoVec  = phenoVec,
  dataType  = "RNAseq",
  normalize = "TMM-log",
  transformation = "TMM-log",
  filterZeroGenes = TRUE,
  varCutOff = NULL
)

ads <- anota2seqAnalyze(
  ads,
  useRVM   = TRUE,
  analysis = c("translation", "buffering", "translated mRNA", "total mRNA")
)

ads <- anota2seqSelSigGenes(
  ads,
  selDeltaPT          = spec$selDeltaPT,
  selDeltaTP          = spec$selDeltaTP,
  maxPAdj             = spec$maxPAdj,
  minSlopeTranslation = spec$minSlopeTranslation,
  maxSlopeTranslation = spec$maxSlopeTranslation,
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
  df$gene_id <- rownames(df)
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

    return {
        "tx2gene": str(tx2gene),
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
    }


def _serialise_spec(spec: dict) -> str:
    """Canonical JSON serialisation used for cache comparison."""
    return json.dumps(spec, indent=2, sort_keys=True)


_ANOTA2SEQ_OUTPUT_NAMES = ("translation", "buffering", "mrna_abundance")
_SPEC_FILENAME = "spec.json"


def _read_cached_outputs(scratch: Path) -> dict[str, pd.DataFrame]:
    """Read the three TSV outputs from a scratch dir, if present."""
    out: dict[str, pd.DataFrame] = {}
    for name in _ANOTA2SEQ_OUTPUT_NAMES:
        p = scratch / f"{name}.tsv"
        if not p.exists() or p.stat().st_size == 0:
            out[name] = pd.DataFrame()
            continue
        try:
            out[name] = pd.read_csv(p, sep="\t")
        except Exception:
            out[name] = pd.DataFrame()
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

    # Skip-if-cached only when: user hasn't forced a rerun, all three
    # output TSVs exist, the cached spec file exists, AND the cached
    # spec matches the fresh spec byte-for-byte after canonical
    # sorted-key serialisation. Any mismatch (threshold change, sample
    # order change, salmon root change) invalidates the cache.
    cached_spec_text: str | None = None
    if cached_spec_path.exists():
        try:
            cached_spec_text = cached_spec_path.read_text()
        except OSError as exc:
            logger.warning("could not read cached spec %s: %s", cached_spec_path, exc)

    cache_hit = (
        not cfg.force_rerun
        and cached_spec_text == fresh_spec_text
        and all((scratch / f"{name}.tsv").exists() for name in _ANOTA2SEQ_OUTPUT_NAMES)
    )
    if cache_hit:
        cached = _read_cached_outputs(scratch)
        logger.info("anota2seq cache hit for %s (spec verified)", contrast)
        return Anota2seqResult(
            contrast=contrast,
            ok=True,
            message=f"anota2seq cache hit for {contrast} (spec verified)",
            translation=cached["translation"],
            buffering=cached["buffering"],
            mrna_abundance=cached["mrna_abundance"],
            scratch_dir=scratch,
        )

    # Cache miss — write the fresh spec and run R. We also stomp over
    # any stale output TSVs so a subsequent failure can't accidentally
    # leave a mixed cache behind.
    cached_spec_path.write_text(fresh_spec_text)
    for name in _ANOTA2SEQ_OUTPUT_NAMES:
        stale = scratch / f"{name}.tsv"
        if stale.exists():
            try:
                stale.unlink()
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

    cached = _read_cached_outputs(scratch)
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
