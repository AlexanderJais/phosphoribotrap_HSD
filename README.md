# Phosphoribotrap RNA-seq — 3-group Streamlit app

Streamlit-driven fastp + salmon + anota2seq pipeline for a phosphorylated-
ribosome immunoprecipitation RNA-seq experiment.

**This is not Ribo-seq.** The reads are full-length mRNA from ribosome-
associated transcripts, not 28–30 nt ribosome-protected footprints, so
footprint-oriented tools (RiboWaltz, RiboFlow, ORFquant, RiboToolkit,
riboSeqR) do not apply and are not part of this pipeline.

## Experimental design

- **18 paired-end libraries** sequenced on a single NovaSeq lane (run
  `A006200122`, lane `L002`, 101 bp reads).
- **3 groups × 3 biological replicates**. Every animal contributes one
  IP pulldown and one hypothalamic INPUT control:
  - **NCD** (normal chow diet, reference) — replicates 1, 3, 4
  - **HSD1** (high-sugar diet, condition 1) — replicates 5, 6, 8
  - **HSD3** (high-sugar diet, condition 3) — replicates 9, 10, 11
- Replicate numbers are **non-contiguous** — IDs 2 and 7 were dropped
  from the cohort. Do not assume a 1…N index.
- IP ↔ INPUT libraries are paired by `(group, replicate)` — same animal.
- **Chronic stimulus** — effect sizes are expected to be mild and
  consistent, so the analysis is tuned for sensitivity over stringency
  (see anota2seq thresholds below).

Default sample sheet (baked into the app, editable in the Samples tab):

| CCG ID | Sample   | Group | Replicate | Fraction |
|--------|----------|-------|-----------|----------|
| 138011 | IP1      | NCD   | 1         | IP       |
| 138013 | IP3      | NCD   | 3         | IP       |
| 138015 | IP4      | NCD   | 4         | IP       |
| 138017 | IP5      | HSD1  | 5         | IP       |
| 138019 | IP6      | HSD1  | 6         | IP       |
| 138021 | IP8      | HSD1  | 8         | IP       |
| 138023 | IP9      | HSD3  | 9         | IP       |
| 138025 | IP10     | HSD3  | 10        | IP       |
| 138027 | IP11     | HSD3  | 11        | IP       |
| 138029 | INPUT1   | NCD   | 1         | INPUT    |
| 138031 | INPUT3   | NCD   | 3         | INPUT    |
| 138033 | INPUT4   | NCD   | 4         | INPUT    |
| 138035 | INPUT5   | HSD1  | 5         | INPUT    |
| 138037 | INPUT6   | HSD1  | 6         | INPUT    |
| 138039 | INPUT8   | HSD1  | 8         | INPUT    |
| 138041 | INPUT9   | HSD3  | 9         | INPUT    |
| 138043 | INPUT10  | HSD3  | 10        | INPUT    |
| 138045 | INPUT11  | HSD3  | 11        | INPUT    |

Fastq naming convention: `A006200122_<CCG>_S<n>_L002_R{1,2}_001.fastq.gz`.
The Samples tab auto-discovers files with that regex.

## QC "failures" that are actually signal — do not remediate

MultiQC was already run; do not re-run FastQC. Interpret the failures
correctly before thinking about fixes:

- **57–69% sequence duplication across all 36 files.** IPs run 63.1–
  66.7%, INPUTs 57.5–65.7% — the 4–6% gap **is the pulldown working**.
  **Do not deduplicate before quantification.** Salmon handles PCR
  duplicates probabilistically during mapping; stripping duplicates
  would destroy the enrichment signal that the whole experiment depends
  on.
- **Overrepresented sequences (R1 fail, R2 warn)** — same cause.
- **Per-base sequence content warnings** — end-repair / sonication
  bias typical of IP protocols.
- **Per-sequence GC content warnings in 27/36 files** — consistent
  with GC-biased phospho-target regions.

**Actual sequencing artefact, flag only:**

- **Per-tile quality: all 18 R1 files fail, all 18 R2 files pass.**
  This is a lane-level flowcell/reagent issue on R1 cycles. Per-base
  quality scores still pass across all 36 files, so impact is mild.
  Salmon's default mapping validation handles it. We do not try to
  "fix" this.

## Pipeline

Single path — no STAR / HISAT2 / featureCounts / MarkDuplicates option:

1. **fastp** — optional via a checkbox. Defaults, paired-end adapter
   detection on.
2. **salmon quant** with `--gcBias --seqBias -g <tx2gene.tsv>` so
   `quant.genes.sf` is produced. `-l A` (auto-detect library type).
   `--validateMappings` is **not** passed — it's default-on in salmon
   ≥1.5 and deprecated.

Both steps are skip-if-cached. Re-running the pipeline short-circuits
samples whose outputs already exist; toggle "Force rerun" in the Config
tab to override.

If you want BAMs for IGV, run that path separately — this app is
deliberately scoped to quantification.

## Analysis — contrast-driven

Every analysis layer runs **per contrast**. Default contrasts:
`HSD1_vs_NCD` and `HSD3_vs_NCD`. `HSD3_vs_HSD1` is available as an
optional extra. The reference group is user-configurable (default NCD).

### 1. anota2seq (primary, first-class)

[anota2seq](https://bioconductor.org/packages/release/bioc/html/anota2seq.html)
is the Bioconductor tool built specifically for paired "polysome-
associated RNA vs. total RNA" designs. The app shells out to `Rscript`
with a generated R script that imports salmon's `quant.sf` via
`tximport`, builds contrast-specific 6-column `dataP` (IP) and `dataT`
(INPUT) matrices, and runs:

```
anota2seqAnalyze(..., useRVM = TRUE)
anota2seqSelSigGenes(
  ...,
  selDeltaPT          = 0.1,   # was 0.2 — chronic-stimulus relaxed
  selDeltaTP          = 0.1,   # was 0.2 — chronic-stimulus relaxed
  maxPAdj             = 0.1,   # was 0.05
  minSlopeTranslation = 0,     # was 0.5
  maxSlopeTranslation = 2,     # was 1.5
  selContrast         = 1
)
```

It reports three output categories per contrast:

- **translation** — IP changes without an INPUT change
- **buffered** — INPUT changes but IP compensates
- **mRNA+translation** — both change

All thresholds are adjustable in the Config tab; the "chronic-stimulus
preset" toggle resets them to the loose defaults above.

**Graceful degradation.** If `Rscript`, `r-base`,
`bioconductor-anota2seq`, or `bioconductor-tximport` aren't installed,
the Analysis tab shows an install hint instead of crashing. The
Python-side cross-check below still works without any R stack.

### 2. Python sign-consistency + between-group Mann-Whitney (secondary)

A no-R-dependency cross-check computed by `phosphotrap.fpkm`:

- **FPKM** per gene from `NumReads` and `EffectiveLength` using the
  Mortazavi formula
  `FPKM = NumReads × 1e9 / (EffectiveLength × total_NumReads)`.
- **Per-pair log2 ratio** `log2(FPKM_IP / max(FPKM_Input, min_fpkm))`
  with a small FPKM floor (default 0.1, user-adjustable) to avoid
  division-by-zero.
- **Per-group summaries** — geometric mean, arithmetic mean, and median
  ratios across the 3 replicates within each group.
- **Sign-consistency** within a group: 3/3 is the ceiling for a 3-per-
  group design. Binomial p-values are reported for completeness but
  are dominated by the discrete sample size.
- **Between-group Mann-Whitney** (the actual secondary statistic for a
  3-vs-3 design): for each gene, Mann-Whitney U on the 3 alt-group log2
  ratios vs the 3 ref-group log2 ratios. BH FDR is applied.
- **Ranked list export** — two-column `gene_id ⇥ delta_log2` TSV per
  contrast, ready for external `fgsea` / preranked GSEA tools.

### Why both

Cross-reference the two outputs: "genes flagged by anota2seq **and** by
the between-group Mann-Whitney" is the high-confidence set for a mild-
stimulus 3-vs-3 experiment. The small-n caveat is unavoidable.

### 3. DESeq2 interaction cross-check (optional, reviewer-friendly)

A button in the Analysis tab shells out to Rscript and fits
`~ group + fraction + group:fraction` via DESeq2, reporting the
`group<alt>.fractionIP` interaction term as an independent cross-
validation. This is explicitly **not** the primary analysis — it's for
"does this also show up in a model a reviewer will recognise?". Same
graceful-degradation pattern as anota2seq.

## Small-n design note

3 vs 3 is small. We know. The chronic-stimulus thresholds are
deliberately loose because we expect mild, consistent effects and we're
optimising for sensitivity at the cost of specificity. The between-
group Mann-Whitney p-values can only take a few discrete values on a
3-vs-3 comparison, so rank-order them alongside the anota2seq calls
rather than treating them as independent tests. Independent validation
(qPCR, biological replication, orthogonal assay) is the path to
publication confidence — no amount of statistical gymnastics will
substitute for that at n = 3 per group.

## Repository layout

```
app.py                      # Streamlit entry point (5 tabs)
phosphotrap/
  __init__.py
  logger.py                 # rotating-file logger, idempotent for Streamlit reruns
  config.py                 # AppConfig dataclass + JSON persistence
  samples.py                # default 18-row sheet, fastq discovery, pair resolver
  pipeline.py               # fastp + salmon runners, progress callbacks
  fpkm.py                   # FPKM, sign-consistency, Mann-Whitney
  anota2seq_runner.py       # primary analysis — Rscript shell-out
  deseq2_runner.py          # optional DESeq2 interaction cross-check
tests/                      # pytest smoke tests
requirements.txt            # pip side (Python only)
environment.yml             # conda bootstrap (includes bioconda tools)
```

## Streamlit app — the five tabs

1. **Config** — paths, runtime options, reference group, contrasts, the
   chronic-stimulus preset toggle, anota2seq thresholds, auto-save on
   pipeline start, "Check environment" button. Shows an unsaved-
   changes indicator vs. the JSON on disk.
2. **Samples** — the 18-row default sheet, editable via `st.data_editor`
   with `num_rows="dynamic"`. Auto-populate fastq paths from a scanned
   directory. Upload custom TSV/CSV. Summary shows groups detected and
   matched IP/INPUT pairs per group. NaN-safe.
3. **Pipeline** — multiselect of ready samples, start button, live
   progress bar that advances mid-sample (not just at sample
   boundaries), per-sample `<report_dir>/<sample>.log` files, dry-run
   checkbox.
4. **Analysis** — contrast selector, "Load salmon quant outputs",
   FPKM preview + TSV download, "Compute IP/Input ratios" (per-group
   sign-consistency + between-group Mann-Whitney for the selected
   contrast), "Run anota2seq" and "DESeq2 interaction cross-check"
   buttons. Histogram, top-30 bar chart, volcano plot, and separate
   anota2seq translation/buffered/mRNA+translation tables. Downloads
   everywhere.
5. **Logs** — live tail of `logs/phosphotrap.log` with substring
   filter. Stitches rolled-over backups so a long run straddling a
   rotation boundary doesn't drop recent lines. The filter-clear
   button stages a reset and reruns the script *before* the widget is
   instantiated (writing to `st.session_state[key]` after a widget
   renders is a no-op).

## Install

### Conda (recommended — includes the Bioconda tools)

```
conda env create -f environment.yml
conda activate phosphotrap
streamlit run app.py
```

`environment.yml` installs `fastp`, `salmon`, `r-base`,
`bioconductor-anota2seq`, `bioconductor-tximport`, and
`bioconductor-deseq2` alongside the Python stack.

### Pip (Python-only; external tools must already be on `PATH`)

```
pip install -r requirements.txt
streamlit run app.py
```

You'll need `fastp`, `salmon`, and `Rscript` + the Bioconductor
packages available some other way — the Analysis tab's anota2seq /
DESeq2 buttons will fall back to a clear install hint if they're
missing, but the primary analysis won't run.

## Tests

```
pytest tests/
```

The smoke tests cover:

- Default sample sheet → 18 records → 3 groups → 3 IP/INPUT pairs per
  group with correct labels and replicate numbers.
- `records_for_contrast("HSD1", "NCD")` returns exactly the 6 matched
  animal pairs in canonical `ref → alt, sorted by replicate` order.
- NaN-safe `to_records`.
- 2-column and 3-column `tx2gene` loader.
- FPKM against a hand-computed expected value.
- Per-group 3-of-3 sign consistency on a synthetic enriched matrix.
- Between-group Mann-Whitney on a synthetic matrix where one group is
  shifted up.
- `AppConfig.load` with bad types / malformed `contrasts` list falling
  through to defaults.
- Progress-callback fractions monotonically non-decreasing, reaching
  1.0, and ticking mid-sample (subprocess layer mocked; no live fastq
  or R required).

## Things this pipeline deliberately does NOT do

- **No** STAR / HISAT2 / alignment path. If you need BAMs, do that
  separately.
- **No** MarkDuplicates / dedup step — the duplication is IP signal,
  not PCR noise. Removing it would destroy the mild-effect statistics
  this experiment was designed to detect.
- **No** `--validateMappings` flag on salmon — it's already default-on
  in salmon ≥ 1.5 and passing it is deprecated.
- **No** footprint-oriented tools (RiboWaltz, ORFquant, RiboToolkit,
  riboSeqR, RiboFlow). Those are for 28–30 nt ribosome-protected
  fragments; our reads are full-length IP mRNA.
- **No** hard-coded R paths — the `Rscript` binary is taken from the
  Config tab so you can point at a conda env or a system install.
- **No** collapsing the 3 groups into 9 undifferentiated pairs. The
  whole point of the experiment is HSD1 vs NCD and HSD3 vs NCD, and
  the analysis layer is built contrast-first.
