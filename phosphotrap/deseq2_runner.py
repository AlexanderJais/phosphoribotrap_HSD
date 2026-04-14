"""DESeq2 interaction model cross-check.

This is an *optional* secondary analysis — not a replacement for
anota2seq. It fits ``~ group + fraction + group:fraction`` where
``fraction`` is IP vs INPUT and reports the ``group<alt>.fractionIP``
interaction term as a reviewer-friendly cross-validation. Gated by the
same graceful-degradation pattern as anota2seq.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AppConfig
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

tx2g <- read.table(spec$tx2gene, sep = "\t", header = FALSE,
                   stringsAsFactors = FALSE)
tx2g <- tx2g[, 1:2]
colnames(tx2g) <- c("TXNAME", "GENEID")

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
out$gene_id <- rownames(out)
out <- out[, c("gene_id", setdiff(colnames(out), "gene_id"))]
write.table(out, file = file.path(out_dir, "interaction.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

cat("ok\n")
"""


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

    # Flatten 6 pairs into 12 libraries in deterministic order.
    subset: list[SampleRecord] = []
    for p in contrast_pairs:
        subset.append(p.ip)
        subset.append(p.input)

    spec = {
        "tx2gene": str(tx2gene),
        "ref_group": ref_group,
        "alt_group": alt_group,
        "files": [str(Path(salmon_root) / r.name() / "quant.sf") for r in subset],
        "names": [r.name() for r in subset],
        "group": [r.group for r in subset],
        "fraction": [r.fraction for r in subset],
    }
    spec_path = scratch / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    r_script = scratch / "run_deseq2.R"
    r_script.write_text(RSCRIPT_TEMPLATE)

    rscript_bin = cfg.rscript_path or "Rscript"
    cmd = [rscript_bin, str(r_script), str(spec_path), str(scratch)]
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

    tbl_path = scratch / "interaction.tsv"
    table = pd.DataFrame()
    if tbl_path.exists() and tbl_path.stat().st_size > 0:
        try:
            table = pd.read_csv(tbl_path, sep="\t")
        except Exception:
            table = pd.DataFrame()

    return DESeq2Result(
        contrast=contrast,
        ok=True,
        message=f"DESeq2 interaction cross-check complete in {time.time() - start:.1f}s",
        table=table,
        scratch_dir=scratch,
    )
