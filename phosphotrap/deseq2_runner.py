"""DESeq2 interaction model cross-check.

This is an *optional* secondary analysis — not a replacement for
anota2seq. It fits ``~ group + fraction + group:fraction`` where
``fraction`` is IP vs INPUT and reports the ``group<alt>.fractionIP``
interaction term as a reviewer-friendly cross-validation. Gated by the
same graceful-degradation pattern as anota2seq.
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
from .samples import SampleRecord, records_for_contrast

logger = get_logger()


@dataclass
class DESeq2Result:
    contrast: str
    ok: bool
    message: str
    table: pd.DataFrame
    scratch_dir: Path


RSCRIPT_TEMPLATE = r"""
suppressMessages({
  library(tximport)
  library(DESeq2)
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
# so the interaction.tsv carries human-readable MGI / HGNC symbols
# alongside the Ensembl gene_ids that tximport aggregates on. This
# matches the behaviour of the anota2seq runner (see
# phosphotrap/anota2seq_runner.py dump_one) so every TSV the pipeline
# writes to disk is readable without a second lookup. Falls back
# gracefully to a gene_id-as-gene_name copy for legacy 2-column
# tx2gene files — the user gets a hint in the Analysis tab if that
# degradation kicks in.
if (ncol(tx2g_raw) >= 3) {
  gene_name_map <- tapply(
    tx2g_raw[, 3], tx2g_raw[, 2],
    function(x) x[!is.na(x) & nzchar(x)][1]
  )
} else {
  gene_name_map <- setNames(character(0), character(0))
}

files <- spec$files
names(files) <- spec$names
txi <- tximport(files, type = "salmon", tx2gene = tx2g)

coldata <- data.frame(
  sample   = spec$names,
  group    = factor(spec$group,    levels = c(spec$ref_group, spec$alt_group)),
  fraction = factor(spec$fraction, levels = c("INPUT", "IP")),
  stringsAsFactors = FALSE
)
rownames(coldata) <- coldata$sample

dds <- DESeqDataSetFromTximport(txi, colData = coldata,
                                design = ~ group + fraction + group:fraction)
dds <- DESeq(dds)

interaction_name <- paste0("group", spec$alt_group, ".fractionIP")
res_names <- resultsNames(dds)
if (!(interaction_name %in% res_names)) {
  cat("ERROR: expected interaction term", interaction_name,
      "not found. Available:", paste(res_names, collapse = ", "), "\n")
  quit(status = 2)
}

res <- results(dds, name = interaction_name)
out <- as.data.frame(res)
gene_ids <- rownames(out)
out$gene_id <- gene_ids
# Insert gene_name right after gene_id so the TSV is readable at a
# glance. Unmapped gene_ids fall back to the gene_id itself so a
# missing symbol never becomes a silent empty cell.
if (length(gene_name_map) > 0) {
  out$gene_name <- unname(gene_name_map[gene_ids])
  out$gene_name[is.na(out$gene_name)] <- gene_ids[is.na(out$gene_name)]
} else {
  out$gene_name <- gene_ids
}
lead <- c("gene_id", "gene_name")
out <- out[, c(lead, setdiff(colnames(out), lead)), drop = FALSE]
write.table(out, file = file.path(out_dir, "interaction.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

cat("ok\n")
"""


_DESEQ2_OUTPUT_NAME = "interaction.tsv"
_SPEC_FILENAME = "spec.json"


def _build_spec(
    contrast_pairs,
    salmon_root: Path,
    tx2gene: Path,
    alt_group: str,
    ref_group: str,
) -> dict:
    # Flatten 6 pairs into 12 libraries in deterministic order.
    subset: list[SampleRecord] = []
    for p in contrast_pairs:
        subset.append(p.ip)
        subset.append(p.input)

    return {
        # Spec schema version — bump this whenever the on-disk output
        # schema changes so existing caches are force-rerun. v2 adds
        # the ``gene_name`` column to interaction.tsv (the R script
        # inserts it as the second column, right after gene_id). A
        # stale v1 cache has no ``gene_name`` column and would
        # otherwise be returned as a cache hit.
        "_schema_version": 2,
        "tx2gene": str(tx2gene),
        "ref_group": ref_group,
        "alt_group": alt_group,
        "files": [str(Path(salmon_root) / r.name() / "quant.sf") for r in subset],
        "names": [r.name() for r in subset],
        "group": [r.group for r in subset],
        "fraction": [r.fraction for r in subset],
    }


def _serialise_spec(spec: dict) -> str:
    return json.dumps(spec, indent=2, sort_keys=True)


def run_deseq2_interaction(
    records: list[SampleRecord],
    *,
    alt_group: str,
    ref_group: str,
    salmon_root: Path,
    tx2gene: Path,
    cfg: AppConfig,
    output_dir: Path,
) -> DESeq2Result:
    contrast = f"{alt_group}_vs_{ref_group}"
    scratch = Path(output_dir) / "deseq2" / contrast
    scratch.mkdir(parents=True, exist_ok=True)

    contrast_pairs = records_for_contrast(
        records, alt_group=alt_group, ref_group=ref_group
    )
    if len(contrast_pairs) != 6:
        return DESeq2Result(
            contrast=contrast,
            ok=False,
            message=f"Expected 6 matched pairs for {contrast}, got {len(contrast_pairs)}",
            table=pd.DataFrame(),
            scratch_dir=scratch,
        )

    # Build the fresh spec in memory so we can verify the cache
    # against it before skipping. The previous skip-if-cached check
    # was "output file exists" — which silently returned stale
    # results after the user changed samples or the salmon root.
    fresh_spec = _build_spec(
        contrast_pairs, Path(salmon_root), Path(tx2gene), alt_group, ref_group,
    )
    fresh_spec_text = _serialise_spec(fresh_spec)
    cached_spec_path = scratch / _SPEC_FILENAME
    cached_table = scratch / _DESEQ2_OUTPUT_NAME

    cached_spec_text: str | None = None
    if cached_spec_path.exists():
        try:
            cached_spec_text = cached_spec_path.read_text()
        except OSError as exc:
            logger.warning("could not read cached spec %s: %s", cached_spec_path, exc)

    cache_hit = (
        not cfg.force_rerun
        and cached_spec_text == fresh_spec_text
        and cached_table.exists()
        and cached_table.stat().st_size > 0
    )
    if cache_hit:
        try:
            table = pd.read_csv(cached_table, sep="\t")
            spec_hash = hashlib.sha256(fresh_spec_text.encode()).hexdigest()[:12]
            logger.info(
                "DESeq2 cache hit for %s (spec sha256:%s)", contrast, spec_hash
            )
            return DESeq2Result(
                contrast=contrast,
                ok=True,
                message=(
                    f"DESeq2 cache hit for {contrast} "
                    f"(spec verified, sha256:{spec_hash})"
                ),
                table=table,
                scratch_dir=scratch,
            )
        except Exception as exc:
            logger.warning("DESeq2 cache read failed for %s: %s", contrast, exc)

    # Cache miss — write the fresh spec, clear any stale output, run R.
    cached_spec_path.write_text(fresh_spec_text)
    if cached_table.exists():
        try:
            cached_table.unlink()
        except OSError:
            pass

    r_script = scratch / "run_deseq2.R"
    r_script.write_text(RSCRIPT_TEMPLATE)

    rscript_bin = resolve_rscript(cfg)
    cmd = [rscript_bin, str(r_script), str(cached_spec_path), str(scratch)]
    logger.info("exec: %s", " ".join(shlex.quote(c) for c in cmd))
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, shell=False, timeout=60 * 60,
        )
    except FileNotFoundError:
        return DESeq2Result(
            contrast=contrast,
            ok=False,
            message=(
                f"{rscript_bin} not found. Install r-base + bioconductor-deseq2 "
                "via the conda bootstrap, then set the Rscript path in Config."
            ),
            table=pd.DataFrame(),
            scratch_dir=scratch,
        )

    log = scratch / "deseq2.log"
    log.write_text((proc.stdout or "") + "\n----STDERR----\n" + (proc.stderr or ""))

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        return DESeq2Result(
            contrast=contrast,
            ok=False,
            message=f"DESeq2 Rscript failed (rc={proc.returncode}): {tail}",
            table=pd.DataFrame(),
            scratch_dir=scratch,
        )

    table = pd.DataFrame()
    if cached_table.exists() and cached_table.stat().st_size > 0:
        try:
            table = pd.read_csv(cached_table, sep="\t")
        except Exception:
            table = pd.DataFrame()

    return DESeq2Result(
        contrast=contrast,
        ok=True,
        message=f"DESeq2 interaction cross-check complete in {time.time() - start:.1f}s",
        table=table,
        scratch_dir=scratch,
    )
