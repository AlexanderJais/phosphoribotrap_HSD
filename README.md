# Phosphoribotrap RNA-seq — 3-group Streamlit app

Streamlit-driven fastp + salmon + anota2seq pipeline for a phosphorylated-
ribosome immunoprecipitation RNA-seq experiment.

**This is not Ribo-seq.** The reads are full-length mRNA from ribosome-
associated transcripts, not 28–30 nt ribosome-protected footprints, so
footprint-oriented tools (RiboWaltz, RiboFlow, ORFquant, RiboToolkit,
riboSeqR) do not apply and are not part of this pipeline.

## Quickstart — first run, in order

If you've never run this app before, do these steps in this order. No
shell commands beyond `conda` and `streamlit run`. The whole reference
build, sample sheet, pipeline, and analysis live inside the app.

1. **Install the conda env** (one time).

   ```
   conda env create -f environment.yml
   conda activate phosphotrap
   ```

   The env is named **`phosphotrap`**, not `phosphoribotrap`. See the
   Install section below for the `mamba` fast-path if conda's solver
   hangs.

2. **Start the app.**

   ```
   streamlit run app.py
   ```

   Six tabs across the top: **Config**, **Reference**, **Samples**,
   **Pipeline**, **Analysis**, **Logs**. Work them left to right.

3. **Reference tab — build the salmon index.** Leave the defaults
   (release `M38`, destination `~/phosphotrap_refs/gencode_mouse_M38/`,
   threads = your core count). Click **Build reference**. One progress
   bar walks through download → decoys → gentrome → `salmon index` →
   `tx2gene.tsv`. Takes ~30–60 minutes on a laptop with reasonable
   wifi; ~15 GB on disk when done. Re-running over an existing
   destination is a no-op.

   When the build finishes, click **Use these paths in Config** — that
   auto-writes `salmon_index` and `tx2gene_tsv` and saves the config.
   You never need to touch those text fields by hand.

4. **Config tab — set your fastq directory.** Type the absolute path
   to the folder containing the 36 `*.fastq.gz` files in the
   **Fastq directory** field, then click **Save config**. Everything
   else on this tab has sane defaults; ignore them on the first run.

5. **Samples tab — auto-populate fastq paths.** Click **Auto-populate
   fastq paths**. The 18-row default sample sheet matches your CCG
   IDs and lane convention; the auto-populator fills in R1/R2 paths
   from the directory you set in step 4. The summary at the bottom
   should say "18 / 18 samples have existing R1 + R2 fastq files" and
   "3 matched IP/INPUT pairs" per group. If it doesn't, your fastq
   filenames don't match the expected
   `A006200122_<CCG>_S<n>_L002_R{1,2}_001.fastq.gz` pattern — fix the
   filenames or edit the sample sheet directly in the data editor.

6. **Pipeline tab — run fastp + salmon.** Leave all 18 samples
   selected. Click **Start pipeline**. Each sample takes ~5–10 min on
   a laptop. Skip-if-cached is on by default, so a crash mid-run
   resumes from the last completed sample on the next click.

7. **Analysis tab — anota2seq + Mann-Whitney.** Pick a contrast
   (default `HSD1_vs_NCD`). Click **Load salmon quant outputs**,
   then **Run anota2seq**, then **Compute IP/Input ratios**. Tables,
   volcano plot, and TSV downloads appear inline. Repeat for the
   `HSD3_vs_NCD` contrast.

If anything goes wrong, the **Logs** tab tails `phosphotrap.log` with
a substring filter, and the Pipeline tab writes per-sample
fastp+salmon stdout to `<report_dir>/logs/per-sample/<sample>.log`.

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

It partitions genes into the three anota2seq regulatory modes per
contrast (retrieved via ``anota2seqGetOutput(ads, output="regModes",
analysis=...)``):

- **translation** — IP changes without an INPUT change
- **buffering** — INPUT changes but IP compensates
- **mRNA abundance** — both change coherently (the "both" category
  the spec called "mRNA+translation")

All thresholds are adjustable in the Config tab; the **Apply chronic-
stimulus preset** button resets them to the loose defaults above.

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
app.py                      # Streamlit entry point (6 tabs)
phosphotrap/
  __init__.py
  logger.py                 # rotating-file logger, idempotent for Streamlit reruns
  config.py                 # AppConfig dataclass + JSON persistence
  samples.py                # default 18-row sheet, fastq discovery, pair resolver
  pipeline.py               # fastp + salmon runners, progress callbacks
  reference.py              # GENCODE downloader + salmon index + tx2gene builder
  fpkm.py                   # FPKM, sign-consistency, Mann-Whitney
  anota2seq_runner.py       # primary analysis — Rscript shell-out
  deseq2_runner.py          # optional DESeq2 interaction cross-check
tests/                      # pytest smoke tests
requirements.txt            # pip side (Python only)
environment.yml             # conda bootstrap (includes bioconda tools)
```

## Streamlit app — the six tabs

1. **Config** — paths, runtime options, reference group, contrasts, the
   chronic-stimulus preset toggle, anota2seq thresholds, auto-save on
   pipeline start, "Check environment" button. Shows an unsaved-
   changes indicator vs. the JSON on disk.
2. **Reference** — one-button GENCODE mouse downloader + decoy-aware
   `salmon index` builder + `tx2gene.tsv` builder. Skip-if-cached at
   every stage, single composite progress bar, and a "Use these paths
   in Config" button that auto-populates `salmon_index` /
   `tx2gene_tsv` and saves the config. Implemented in pure Python
   (`phosphotrap/reference.py`) on `urllib` + `gzip` + one
   `salmon index` shell-out — no `curl` / `zcat` / `awk` dependencies.
   See the "Reference: building the salmon index + tx2gene" section
   below for the manual command-line equivalent.
3. **Samples** — the 18-row default sheet, editable via `st.data_editor`
   with `num_rows="dynamic"`. Auto-populate fastq paths from a scanned
   directory. Upload custom TSV/CSV. Summary shows groups detected and
   matched IP/INPUT pairs per group. NaN-safe.
4. **Pipeline** — multiselect of ready samples, start button, live
   progress bar that advances mid-sample (not just at sample
   boundaries), per-sample `<report_dir>/<sample>.log` files, dry-run
   checkbox.
5. **Analysis** — contrast selector, "Load salmon quant outputs",
   FPKM preview + TSV download, "Compute IP/Input ratios" (per-group
   sign-consistency + between-group Mann-Whitney for the selected
   contrast), "Run anota2seq" and "DESeq2 interaction cross-check"
   buttons. Histogram, top-30 bar chart, volcano plot, and separate
   anota2seq translation/buffering/mRNA-abundance tables. Downloads
   everywhere.
6. **Logs** — live tail of `logs/phosphotrap.log` with substring
   filter. Stitches rolled-over backups so a long run straddling a
   rotation boundary doesn't drop recent lines. The filter-clear
   button stages a reset and reruns the script *before* the widget is
   instantiated (writing to `st.session_state[key]` after a widget
   renders is a no-op).

## Install

For a guided first run, see [Quickstart](#quickstart--first-run-in-order)
above — this section just covers the env setup.

### Conda (recommended — includes the Bioconda tools)

```
conda env create -f environment.yml
conda activate phosphotrap
streamlit run app.py
```

> **Heads up — env name vs repo name.** The conda environment is called
> **`phosphotrap`** (no "ribo"), defined at `environment.yml:1`. The
> repository is `phosphoribotrap_HSD`. `conda activate phosphoribotrap`
> will fail with `EnvironmentNameNotFound`; always use
> `conda activate phosphotrap`.

`environment.yml` installs `fastp`, `salmon`, `r-base`,
`bioconductor-anota2seq`, `bioconductor-tximport`, and
`bioconductor-deseq2` alongside the Python stack.

If conda's solver hangs, install [`mamba`](https://mamba.readthedocs.io/)
(`conda install -n base -c conda-forge mamba`) and run
`mamba env create -f environment.yml` instead — same result, much
faster.

### Pip (Python-only; external tools must already be on `PATH`)

```
pip install -r requirements.txt
streamlit run app.py
```

You'll need `fastp`, `salmon`, and `Rscript` + the Bioconductor
packages available some other way — the Analysis tab's anota2seq /
DESeq2 buttons will fall back to a clear install hint if they're
missing, but the primary analysis won't run.

## Reference: building the salmon index + tx2gene (mouse / GRCm39, GENCODE M38)

The pipeline doesn't ship a salmon index — `salmon_index` and
`tx2gene_tsv` in `phosphotrap/config.py` are paths you provide. Build
them once, store them somewhere stable (e.g. `~/phosphotrap_refs/gencode_mouse_M38/`),
and point every mouse project at the same directory. The index is
~15 GB and reusable; you only build it once.

### Recommended: the **Reference** tab in the Streamlit app

Open the app (`streamlit run app.py`) and switch to the **Reference**
tab — second from the left. It hides the entire download / decoys /
gentrome / `salmon index` / `tx2gene.tsv` workflow behind a single
button:

1. Set the GENCODE release (default `M38`) and a destination directory
   (default `~/phosphotrap_refs/gencode_mouse_M38/`). Expand the
   "Preview download URLs" panel to sanity-check the release name
   before kicking off ~1 GB of downloads.
2. Click **Build reference**. A single progress bar tracks all five
   stages: download transcriptome, download genome, download GTF,
   build decoys + gentrome, run `salmon index`, build `tx2gene.tsv`.
   Every step is skip-if-cached, so re-running over an existing
   destination directory is a no-op.
3. When the build finishes, click **Use these paths in Config** to
   auto-populate `salmon_index` and `tx2gene_tsv` and save the config
   to disk in one go. Switch to the Pipeline tab and you're ready.

The whole thing is implemented in pure Python (`phosphotrap/reference.py`)
on top of `urllib`, `gzip`, and a single `salmon index` shell-out — no
`curl`, `wget`, `zcat`, `awk`, or `sed` involved, so macOS / Linux /
Windows-via-WSL all behave the same.

### Manual command-line fallback (if you can't run the app)

If you'd rather build the reference from a terminal — for example on a
headless cluster node, or because you want to script it into a larger
pipeline — the rest of this section walks through the same steps by
hand. The Reference tab does exactly what's below; the only reason to
prefer the manual route is automation.

These reads are mouse, so the reference is **GRCm39** (the current
mouse assembly). Use **GENCODE** so transcript IDs match between the
FASTA and the GTF; the `--gencode` flag at index time strips the
version suffixes (`ENSMUST00000193812.2` → `ENSMUST00000193812`) so
`tximport` lines up downstream. This guide pins to **GENCODE release
M38** (published 2025-09-02), the current release at
<https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/>.
If a newer release exists when you read this, just bump the `REL`
variable below — every other command stays the same.

### What you need from the M38 directory

The release directory has 30+ files. You only need three:

| File                                                | Size  | Purpose                                |
|-----------------------------------------------------|-------|----------------------------------------|
| `gencode.vM38.transcripts.fa.gz`                    |  72 M | transcriptome — what salmon quantifies |
| `GRCm39.primary_assembly.genome.fa.gz`              | 739 M | genome — used as decoys only           |
| `gencode.vM38.primary_assembly.annotation.gtf.gz`   |  36 M | GTF — used to build `tx2gene.tsv`      |

Do **not** grab `pc_transcripts.fa.gz` (protein-coding only — drops
every lncRNA / pseudogene), `chr_patch_hapl_scaff.*` (inflates the
index with patch / haplotype scaffolds for no benefit), or
`*.basic.annotation.*` (subset of transcripts, will mismatch the full
transcriptome FASTA).

### Step-by-step

Pick a working directory with **at least 30 GB free** for the
intermediate files. On a small-SSD MacBook, build on an external drive.

```bash
# zsh on macOS doesn't treat `#` as a comment in interactive mode by
# default — enable it so this whole block pastes verbatim.
setopt interactive_comments 2>/dev/null || true

# 0. activate the project env (the env is "phosphotrap", NOT
#    "phosphoribotrap" — see the Install section above).
conda activate phosphotrap

# 1. download the three files from GENCODE mouse M38.
#    macOS ships curl, not wget. -L follows redirects, -O saves under
#    the remote filename, -C - resumes a partial download if the
#    connection drops.
REL=M38
BASE=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_${REL}

curl -L -C - -O ${BASE}/gencode.v${REL}.transcripts.fa.gz
curl -L -C - -O ${BASE}/GRCm39.primary_assembly.genome.fa.gz
curl -L -C - -O ${BASE}/gencode.v${REL}.primary_assembly.annotation.gtf.gz

# Optional: verify integrity against the release's MD5SUMS.
curl -L -O ${BASE}/MD5SUMS
md5sum -c MD5SUMS 2>/dev/null \
    | grep -E "(transcripts\.fa|primary_assembly\.genome\.fa|primary_assembly\.annotation\.gtf)"
# (macOS doesn't ship `md5sum`; use `md5 -r <file>` and eyeball-compare
#  to the matching line in MD5SUMS, or `brew install md5sha1sum`.)

# 2. build the decoy list + concatenated reference for selective
#    alignment. The decoys file is the list of chromosome names from
#    the genome FASTA; gentrome.fa.gz is the transcriptome with the
#    genome appended.
zgrep "^>" GRCm39.primary_assembly.genome.fa.gz \
    | cut -d ' ' -f1 | sed 's/>//' > decoys.txt
cat gencode.v${REL}.transcripts.fa.gz GRCm39.primary_assembly.genome.fa.gz \
    > gentrome.fa.gz

# Sanity check before the long step:
ls -lh decoys.txt gentrome.fa.gz   # expect ~60 lines, ~810 MB
head -3 decoys.txt                 # expect chr1 / chr2 / chr3
wc -l decoys.txt                   # expect ~61 lines

# 3. build the salmon index. ~20-40 minutes, ~15 GB RAM. Close Chrome.
#    -k 31 is the standard k-mer size for ≥75 bp reads (these are 101 bp).
#    --gencode strips the .N transcript-version suffix.
#    -p 8 uses 8 threads; bump to match your core count.
salmon index \
    -t gentrome.fa.gz \
    -d decoys.txt \
    -i salmon_index_gencode_M${REL} \
    -k 31 \
    --gencode \
    -p 8

# 4. build the matching tx2gene.tsv from the SAME GTF you used in
#    step 1. Three columns: transcript_id, gene_id, gene_name.
#    tximport in phosphotrap/anota2seq_runner.py expects exactly this.
zcat gencode.v${REL}.primary_assembly.annotation.gtf.gz \
  | awk '$3=="transcript"' \
  | sed -E 's/.*transcript_id "([^"]+)".*gene_id "([^"]+)".*gene_name "([^"]+)".*/\1\t\2\t\3/' \
  > tx2gene.tsv

# Sanity check:
wc -l tx2gene.tsv          # expect ~150,000 lines for M38
head -3 tx2gene.tsv        # three tab-separated columns
```

> **macOS note.** `zcat` on macOS is GNU-incompatible and only reads
> `.Z` files. If `zcat ... .gtf.gz` errors, use `gunzip -c` instead:
> `gunzip -c gencode.vM38.primary_assembly.annotation.gtf.gz | awk ...`.

### Pointing the app at the result

In the Streamlit **Config** tab, set:

- **Salmon index** → the absolute path of the
  `salmon_index_gencode_M38/` **directory** (the one containing
  `info.json`, `pos.bin`, `seq.bin`, …), **not** a file inside it.
  Salmon's `-i` argument is a directory and so is the Config field.
- **tx2gene TSV** → the absolute path of `tx2gene.tsv`.

After the build, you can delete the intermediates if disk is tight:
`gentrome.fa.gz`, `decoys.txt`, the genome FASTA, and the
transcriptome FASTA are all unused once the index is built. Keep the
GTF if you might want to rebuild `tx2gene.tsv` with a different
column layout later. The index directory itself is the only thing
the pipeline actually reads from.

### Things that silently break this

- **Mismatched releases.** Transcriptome FASTA, genome FASTA, and GTF
  must all come from the same GENCODE release. Mixing M37 transcripts
  with an M38 GTF will silently drop genes during `tximport`.
- **GENCODE vs Ensembl.** Both publish GRCm39 and are equivalent, but
  GENCODE keeps `.N` version suffixes on transcript IDs while Ensembl
  doesn't. The `--gencode` flag handles this **only** for GENCODE
  inputs. If you switch to Ensembl, drop `--gencode` and rebuild
  `tx2gene.tsv` from the Ensembl GTF.
- **`pc_transcripts.fa.gz` instead of `transcripts.fa.gz`.** The
  former is protein-coding only — using it makes salmon silently miss
  every lncRNA / pseudogene in the data. Always grab the plain
  `gencode.vM38.transcripts.fa.gz`.
- **`primary_assembly` vs `chr_patch_hapl_scaff`.** Use
  `primary_assembly` for both the genome and the GTF. The patch /
  haplotype scaffolds inflate the index for no benefit on a standard
  RNA-seq run. (For mouse M38, `GRCm39.genome.fa.gz` and
  `GRCm39.primary_assembly.genome.fa.gz` happen to be byte-identical
  at 739 MB, but stick to the `primary_assembly` name for consistency
  with the GTF.)
- **`*.basic.annotation.*` GTFs.** GENCODE's "basic" subset drops
  alternative transcripts. If you build `tx2gene.tsv` from the basic
  GTF but quantify against the full transcriptome FASTA, every
  alternative transcript silently fails the tximport join.
- **Pointing the Streamlit field at a file.** Salmon's `-i` argument
  is a directory. The Config field expects the same.

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
