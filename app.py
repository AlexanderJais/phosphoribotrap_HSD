"""Streamlit UI for the phosphoribotrap 3-group RNA-seq pipeline.

Run with::

    streamlit run app.py

See README.md for the design rationale — in particular why the QC
"failures" reported by MultiQC are mostly expected IP enrichment signal,
why we do not deduplicate, and why anota2seq is the primary analysis
rather than footprint-oriented tools.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from phosphotrap.anota2seq_runner import run_anota2seq
from phosphotrap.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    contrasts_for_reference,
    reconcile_contrasts,
    resolve_rscript,
)
from phosphotrap.deseq2_runner import run_deseq2_interaction
from phosphotrap.fpkm import (
    between_group_contrast,
    load_salmon_matrix,
    pair_ratios,
)
from phosphotrap.logger import (
    attach_file_handler,
    get_logger,
    list_per_sample_logs,
    read_log_file,
    tail_log,
)
from phosphotrap.pipeline import check_environment, run_pipeline
from phosphotrap.reference import (
    DEFAULT_GENCODE_MOUSE_RELEASE,
    GencodeFiles,
    build_reference,
)
from phosphotrap.samples import (
    GROUPS,
    default_sample_df,
    pairs_by_group,
    populate_fastq_paths,
    ready_records,
    records_to_df,
    summary,
    to_records,
)

# Chronic-stimulus threshold preset — relaxed defaults for the app's
# "apply preset" button (3-vs-3, mild effect sizes).
CHRONIC_PRESET = {
    "anota_delta_pt": 0.1,
    "anota_delta_tp": 0.1,
    "anota_max_padj": 0.1,
    "anota_min_slope_trans": 0.0,
    "anota_max_slope_trans": 2.0,
}

logger = get_logger()

st.set_page_config(
    page_title="Phosphoribotrap RNA-seq",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Session-state bootstrap
# ----------------------------------------------------------------------
if "disk_config" not in st.session_state:
    st.session_state.disk_config = AppConfig.load(DEFAULT_CONFIG_PATH)
if "cfg" not in st.session_state:
    st.session_state.cfg = AppConfig.load(DEFAULT_CONFIG_PATH)
if "sample_df" not in st.session_state:
    st.session_state.sample_df = default_sample_df()
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = []
if "log_filter_staged_clear" not in st.session_state:
    st.session_state.log_filter_staged_clear = False
if "analysis" not in st.session_state:
    st.session_state.analysis = {}  # keyed by contrast
if "_preset_just_applied" not in st.session_state:
    st.session_state._preset_just_applied = False


cfg: AppConfig = st.session_state.cfg
disk_cfg: AppConfig = st.session_state.disk_config

# Attach the file handler to the configured report dir. Idempotent, so
# Streamlit reruns don't stack handlers; if the user changes report_dir
# and saves, the next rerun rotates the handler onto the new location.
# attach_file_handler itself is defensive — if the requested directory
# can't be created (PermissionError, empty string, read-only FS) it
# falls back to DEFAULT_LOG_DIR and logs a warning rather than
# crashing the app. ``cfg.effective_report_dir()`` additionally
# coerces blank / ``"."`` user input back to the dataclass default
# so a cleared text input doesn't silently create ``logs/`` in cwd.
attach_file_handler(cfg.effective_report_dir() / "logs")

# Sample records are derived from st.session_state.sample_df and used
# by every tab. Compute once per rerun rather than 5+ times scattered
# through the tab bodies — same input, same output, no point.
sample_records = to_records(st.session_state.sample_df)
ready_samples = ready_records(sample_records)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
st.title("Phosphoribotrap RNA-seq — 3-group (NCD / HSD1 / HSD3)")

tabs = st.tabs(
    ["Config", "Reference", "Samples", "Pipeline", "Analysis", "Logs"]
)

# ======================================================================
# CONFIG TAB
# ======================================================================
with tabs[0]:
    st.header("Configuration")

    # The unsaved-changes indicator is rendered at the *end* of this
    # tab, after every widget has written its live value back into
    # cfg. Rendering it at the top would show state from the previous
    # rerun — always one step behind whatever the user sees on screen.

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Paths")
        cfg.fastq_dir    = st.text_input("Fastq directory", cfg.fastq_dir)
        cfg.salmon_index = st.text_input("Salmon index",     cfg.salmon_index)
        cfg.tx2gene_tsv  = st.text_input("tx2gene TSV (2- or 3-col)", cfg.tx2gene_tsv)
        cfg.output_dir   = st.text_input("Output directory",  cfg.output_dir)
        cfg.report_dir   = st.text_input("Report/log directory", cfg.report_dir)
        cfg.rscript_path = st.text_input("Rscript path",       cfg.rscript_path)
    with col_b:
        st.subheader("Runtime")
        cfg.threads      = int(st.number_input("Threads", min_value=1, max_value=128, value=int(cfg.threads)))
        cfg.run_fastp    = st.checkbox("Run fastp trimming",    value=cfg.run_fastp)
        cfg.force_rerun  = st.checkbox("Force rerun (ignore cache)", value=cfg.force_rerun)
        cfg.salmon_libtype = st.selectbox(
            "Salmon libType",
            options=["A", "IU", "ISR", "ISF", "MU", "OU"],
            index=["A", "IU", "ISR", "ISF", "MU", "OU"].index(cfg.salmon_libtype)
            if cfg.salmon_libtype in ["A", "IU", "ISR", "ISF", "MU", "OU"] else 0,
        )

    st.divider()
    st.subheader("Design")

    # Reference group drives the contrasts list. Changing it invalidates
    # any selections held in the multiselect's session state, so we
    # reconcile before rendering — otherwise Streamlit will either crash
    # (older versions) or silently drop the stale values (newer
    # versions). Either way the user would see the wrong thing.
    d1, d2 = st.columns(2)
    with d1:
        cfg.reference_group = st.selectbox(
            "Reference group",
            options=list(GROUPS),
            index=list(GROUPS).index(cfg.reference_group)
            if cfg.reference_group in GROUPS else 0,
            help="Contrast strings on downstream tabs are derived from this.",
        )
        available = contrasts_for_reference(cfg.reference_group, GROUPS)

        _contrasts_key = "contrasts_multiselect"
        if _contrasts_key in st.session_state:
            # Filter any stale session-state values against the current
            # options. If nothing survives, drop the key entirely so
            # ``default=`` below re-initialises the widget.
            reconciled = reconcile_contrasts(
                st.session_state[_contrasts_key], available
            )
            if reconciled:
                st.session_state[_contrasts_key] = reconciled
            else:
                del st.session_state[_contrasts_key]

        cfg.contrasts = st.multiselect(
            "Contrasts",
            options=available,
            default=reconcile_contrasts(cfg.contrasts, available) or available[:2],
            key=_contrasts_key,
        )
    with d2:
        st.caption(
            "Chronic-stimulus preset: loose thresholds (deltaPT/TP=0.1, "
            "maxPAdj=0.1, slopeTrans 0–2) for mild effect sizes and n=3."
        )
        if st.button("Apply chronic-stimulus preset"):
            # Write directly to widget session state so the next render
            # reflects the preset values regardless of prior user edits.
            for key, value in CHRONIC_PRESET.items():
                st.session_state[f"widget_{key}"] = value
            # Flag a one-shot confirmation banner for the next render.
            st.session_state._preset_just_applied = True
            st.rerun()
        if st.session_state._preset_just_applied:
            st.success("Chronic-stimulus preset applied.")
            st.session_state._preset_just_applied = False

    # ------------------------------------------------------------------
    # anota2seq thresholds. Each number_input has an explicit widget key
    # so the "Apply preset" button above can rewrite them via
    # st.session_state *before* the widgets render on the next run.
    # After the widgets render we copy their live values back into cfg
    # so diff() and save() see the same thing the user sees.
    # ------------------------------------------------------------------
    for _k, _default in (
        ("anota_delta_pt", cfg.anota_delta_pt),
        ("anota_delta_tp", cfg.anota_delta_tp),
        ("anota_max_padj", cfg.anota_max_padj),
        ("anota_min_slope_trans", cfg.anota_min_slope_trans),
        ("anota_max_slope_trans", cfg.anota_max_slope_trans),
        ("min_fpkm", cfg.min_fpkm),
    ):
        st.session_state.setdefault(f"widget_{_k}", float(_default))

    st.subheader("anota2seq thresholds")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.number_input("selDeltaPT", key="widget_anota_delta_pt", step=0.05, format="%.2f")
        st.number_input("selDeltaTP", key="widget_anota_delta_tp", step=0.05, format="%.2f")
    with t2:
        st.number_input("maxPAdj", key="widget_anota_max_padj", step=0.01, format="%.2f")
        st.number_input("FPKM floor (ratio denom)", key="widget_min_fpkm", step=0.05, format="%.2f")
    with t3:
        st.number_input("minSlopeTranslation", key="widget_anota_min_slope_trans", step=0.1, format="%.2f")
        st.number_input("maxSlopeTranslation", key="widget_anota_max_slope_trans", step=0.1, format="%.2f")

    # Copy widget state back into cfg so save/diff see the live values.
    cfg.anota_delta_pt        = float(st.session_state["widget_anota_delta_pt"])
    cfg.anota_delta_tp        = float(st.session_state["widget_anota_delta_tp"])
    cfg.anota_max_padj        = float(st.session_state["widget_anota_max_padj"])
    cfg.anota_min_slope_trans = float(st.session_state["widget_anota_min_slope_trans"])
    cfg.anota_max_slope_trans = float(st.session_state["widget_anota_max_slope_trans"])
    cfg.min_fpkm              = float(st.session_state["widget_min_fpkm"])

    st.divider()

    # Diff is computed here, at the end of the tab, so it reflects the
    # live widget values the user has just edited — not the state from
    # the previous rerun.
    unsaved = cfg.diff(disk_cfg)
    if unsaved:
        st.warning(
            f"Unsaved changes vs. {DEFAULT_CONFIG_PATH}: "
            + ", ".join(sorted(unsaved.keys()))
        )
    else:
        st.success(f"Config in sync with {DEFAULT_CONFIG_PATH}")

    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        if st.button("Save config", type="primary"):
            try:
                path = cfg.save(DEFAULT_CONFIG_PATH)
                st.session_state.disk_config = AppConfig.load(DEFAULT_CONFIG_PATH)
                st.success(f"Saved {path}")
                logger.info("config saved to %s", path)
            except OSError as exc:
                # The save button used to surface a raw traceback in
                # the Streamlit error panel on a read-only config
                # path. Catch it and show a friendly message instead.
                logger.exception("config save failed")
                st.error(
                    f"Could not save config to {DEFAULT_CONFIG_PATH}: {exc}"
                )
    with bcol2:
        if st.button("Check environment"):
            st.session_state.last_env_check = check_environment(
                resolve_rscript(cfg)
            )

    # Render the last environment-check result outside the button
    # block so it persists across reruns. The previous version only
    # rendered inside the ``if st.button:`` guard, so any other
    # interaction (e.g., typing in a text field) cleared the display.
    last_env = st.session_state.get("last_env_check")
    if last_env:
        rows = [{"tool": k, **v} for k, v in last_env.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ======================================================================
# REFERENCE TAB
# ======================================================================
with tabs[1]:
    st.header("Reference — GENCODE mouse + salmon index")
    st.caption(
        "One-button download + index build. Hides the curl / zcat / awk "
        "song-and-dance behind a progress bar. Re-running with the same "
        "destination is a no-op once everything is cached."
    )

    st.markdown(
        "**What this builds:** GRCm39 transcriptome + genome from a "
        "GENCODE mouse release, a decoy-aware `salmon index`, and a "
        "matching `tx2gene.tsv` — the two paths the Config tab needs "
        "to run the pipeline."
    )

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        ref_release = st.text_input(
            "GENCODE mouse release",
            value=st.session_state.get(
                "ref_release", DEFAULT_GENCODE_MOUSE_RELEASE
            ),
            key="ref_release",
            help=(
                "e.g. M38 (current as of 2025-09-02). Pick the latest "
                "from https://ftp.ebi.ac.uk/pub/databases/gencode/"
                "Gencode_mouse/ — only the release token changes, the "
                "filenames are stable."
            ),
        )
        default_dest = str(
            Path.home() / "phosphotrap_refs" / f"gencode_mouse_{ref_release}"
        )
        ref_dest = st.text_input(
            "Destination directory",
            value=st.session_state.get("ref_dest", default_dest),
            key="ref_dest",
            help=(
                "Where the downloads, gentrome, salmon index, and "
                "tx2gene.tsv go. ~15 GB total. Reusable across every "
                "mouse RNA-seq project — point it somewhere stable."
            ),
        )
    with rcol2:
        ref_threads = int(
            st.number_input(
                "Threads (for salmon index)",
                min_value=1,
                max_value=128,
                value=int(st.session_state.get("ref_threads", cfg.threads)),
                key="ref_threads",
            )
        )
        ref_force = st.checkbox(
            "Force rebuild (ignore cached salmon index)",
            value=False,
            key="ref_force",
            help=(
                "Downloads are still skipped if the .fa.gz / .gtf.gz "
                "files are already on disk. Only the index rebuild is "
                "forced."
            ),
        )

    # Preview the URLs the build will hit so the user can sanity-check
    # the release name before kicking off a 1 GB download.
    try:
        _preview = GencodeFiles.for_mouse(ref_release)
        with st.expander("Preview download URLs", expanded=False):
            st.code(
                "\n".join(
                    [
                        _preview.transcripts_url,
                        _preview.genome_url,
                        _preview.gtf_url,
                    ]
                ),
                language="text",
            )
    except ValueError as exc:
        st.error(str(exc))

    # Progress bar lives in an st.empty() so it disappears between runs
    # — same pattern as the Pipeline tab, for the same reason (a static
    # 1.0-filled bar after a completed run looks like work in progress).
    ref_progress_container = st.empty()
    ref_status_container = st.empty()

    if "reference_artifacts" not in st.session_state:
        st.session_state.reference_artifacts = None

    if st.button("Build reference (download + index + tx2gene)", type="primary"):
        progress_bar = ref_progress_container.progress(0.0, text="starting…")

        def _ref_cb(frac: float, msg: str) -> None:
            frac = max(0.0, min(1.0, float(frac)))
            progress_bar.progress(frac, text=msg)

        with st.spinner("Building reference — this takes a while…"):
            try:
                artifacts = build_reference(
                    release=ref_release,
                    dest_dir=Path(ref_dest),
                    threads=ref_threads,
                    force=ref_force,
                    progress_cb=_ref_cb,
                )
                st.session_state.reference_artifacts = artifacts
                ref_status_container.success(
                    f"Built salmon index at {artifacts.index_dir} "
                    f"and tx2gene.tsv with {artifacts.n_transcripts} transcripts."
                )
                logger.info(
                    "reference build complete: index=%s tx2gene=%s n=%d",
                    artifacts.index_dir,
                    artifacts.tx2gene_tsv,
                    artifacts.n_transcripts,
                )
            except Exception as exc:
                logger.exception("reference build failed")
                ref_status_container.error(f"Reference build failed: {exc}")
                st.session_state.reference_artifacts = None

    artifacts = st.session_state.get("reference_artifacts")
    if artifacts is not None:
        st.divider()
        st.subheader("Build artifacts")
        st.json(
            {
                "index_dir": str(artifacts.index_dir),
                "tx2gene_tsv": str(artifacts.tx2gene_tsv),
                "transcripts_fa": str(artifacts.transcripts_fa),
                "genome_fa": str(artifacts.genome_fa),
                "gtf": str(artifacts.gtf),
                "n_transcripts": artifacts.n_transcripts,
            }
        )

        acol1, acol2 = st.columns(2)
        with acol1:
            if st.button("Use these paths in Config", type="primary"):
                cfg.salmon_index = str(artifacts.index_dir)
                cfg.tx2gene_tsv = str(artifacts.tx2gene_tsv)
                try:
                    cfg.save(DEFAULT_CONFIG_PATH)
                    st.session_state.disk_config = AppConfig.load(
                        DEFAULT_CONFIG_PATH
                    )
                    st.success(
                        f"Config updated and saved: salmon_index → "
                        f"{artifacts.index_dir.name}/, tx2gene_tsv → "
                        f"{artifacts.tx2gene_tsv.name}"
                    )
                except OSError as exc:
                    st.error(f"Config save failed: {exc}")
        with acol2:
            st.caption(
                "After clicking, switch to the Config tab to verify, then "
                "head to Pipeline. The intermediate FASTAs and gentrome.fa.gz "
                "can be deleted if disk is tight — only the index directory "
                "and tx2gene.tsv are needed for downstream runs."
            )

# ======================================================================
# SAMPLES TAB
# ======================================================================
with tabs[2]:
    st.header("Sample sheet (3 groups × 3 replicates × IP + INPUT)")
    st.caption(
        "Replicate IDs 2 and 7 were dropped from the cohort — numbering is "
        "intentionally non-contiguous. IP ↔ INPUT are paired by (group, replicate)."
    )

    up = st.file_uploader("Upload custom sample sheet (TSV/CSV)", type=["tsv", "csv"])
    if up is not None:
        try:
            sep = "\t" if up.name.lower().endswith(".tsv") else ","
            st.session_state.sample_df = pd.read_csv(up, sep=sep)
            st.success(f"Loaded {len(st.session_state.sample_df)} rows from {up.name}")
        except Exception as exc:
            st.error(f"Failed to parse sample sheet: {exc}")

    edited = st.data_editor(
        st.session_state.sample_df,
        num_rows="dynamic",
        use_container_width=True,
        key="sample_editor",
    )
    st.session_state.sample_df = edited

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset to default 18-row sheet"):
            st.session_state.sample_df = default_sample_df()
            st.rerun()
    with c2:
        if st.button("Auto-populate fastq paths", type="primary"):
            if not cfg.fastq_dir:
                st.error("Set a fastq directory in Config first.")
            else:
                # This button mutates sample_df and reruns; the cached
                # sample_records/ready_samples above will be
                # recomputed from the fresh df on the next rerun.
                populate_fastq_paths(sample_records, Path(cfg.fastq_dir))
                st.session_state.sample_df = records_to_df(sample_records)
                st.rerun()
    with c3:
        st.download_button(
            "Download sheet (TSV)",
            data=st.session_state.sample_df.to_csv(sep="\t", index=False),
            file_name="phosphotrap_samples.tsv",
            mime="text/tab-separated-values",
        )

    st.subheader("Summary")
    st.json(summary(sample_records))
    st.write(
        f"**{len(ready_samples)} / {len(sample_records)} samples have existing "
        "R1 + R2 fastq files.**"
    )
    by_group = pairs_by_group(ready_samples)
    for g in GROUPS:
        n = len(by_group.get(g, []))
        st.write(f"- {g}: {n} matched IP/INPUT pair(s)")

# ======================================================================
# PIPELINE TAB
# ======================================================================
with tabs[3]:
    st.header("Pipeline — fastp (optional) + salmon")
    st.caption(
        "Skip-if-cached by default; use the force_rerun checkbox in Config to override. "
        "No dedup step — IP enrichment drives the duplication rate, do not strip it."
    )

    options = [r.name() for r in ready_samples]
    selected_names = st.multiselect(
        "Samples to run",
        options=options,
        default=options,
        help="Only samples whose R1+R2 fastq paths already exist are listed.",
    )
    selected = [r for r in ready_samples if r.name() in selected_names]

    dry_run = st.checkbox("Dry run (environment check only)")

    # Progress bar lives inside an ``st.empty()`` placeholder so it
    # only renders while a pipeline is actually running. Previously we
    # stored ``progress_value`` in session state and re-rendered a
    # static 1.0-filled bar after a completed run, which looked like
    # a pipeline was still running when the user was just switching
    # tabs. st.empty() is recreated on every rerun — if nothing writes
    # into it during this render, it simply draws nothing.
    progress_container = st.empty()
    st.caption(
        "Progress is a smoothed line-count ramp from the subprocess output, "
        "not a wall-clock ETA — fastp and salmon don't emit parseable progress."
    )

    if st.button("Start pipeline", type="primary", disabled=not selected and not dry_run):
        # Auto-save current config first.
        try:
            cfg.save(DEFAULT_CONFIG_PATH)
            st.session_state.disk_config = AppConfig.load(DEFAULT_CONFIG_PATH)
        except OSError as exc:
            logger.warning("auto-save before pipeline start failed: %s", exc)

        progress_bar = progress_container.progress(0.0, text="starting…")

        def cb(sample_idx: int, step_idx: int, frac: float, msg: str) -> None:
            frac = max(0.0, min(1.0, float(frac)))
            progress_bar.progress(frac, text=msg)

        with st.spinner("Running pipeline..."):
            try:
                results = run_pipeline(
                    selected,
                    salmon_index=Path(cfg.salmon_index),
                    tx2gene=Path(cfg.tx2gene_tsv),
                    output_dir=cfg.effective_output_dir(),
                    report_dir=cfg.effective_report_dir(),
                    threads=cfg.threads,
                    run_fastp_step=cfg.run_fastp,
                    libtype=cfg.salmon_libtype,
                    force=cfg.force_rerun,
                    progress_cb=cb,
                    dry_run=dry_run,
                    rscript_path=resolve_rscript(cfg),
                )
            except Exception as exc:
                logger.exception("pipeline crashed")
                st.error(f"Pipeline crashed: {exc}")
                results = []
        st.session_state.pipeline_results = results
        st.success(f"Pipeline complete: {len(results)} step(s)")

    if st.session_state.pipeline_results:
        rows = [asdict(r) for r in st.session_state.pipeline_results]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ======================================================================
# ANALYSIS TAB
# ======================================================================
with tabs[4]:
    st.header("Analysis — per contrast")

    # Contrasts are derived from the configured reference group; fall
    # back to the first two valid options for that reference if the
    # user hasn't selected anything yet.
    derived = contrasts_for_reference(cfg.reference_group, GROUPS)
    available_contrasts = [c for c in cfg.contrasts if c in derived] or derived[:2]
    contrast = st.selectbox(
        "Contrast",
        options=available_contrasts,
        help="anota2seq, sign-consistency, and Mann-Whitney are all run per contrast.",
    )
    alt_group, ref_group = contrast.split("_vs_")

    if st.button("Load salmon quant outputs"):
        try:
            result = load_salmon_matrix(
                ready_samples, cfg.effective_output_dir() / "salmon"
            )
            st.session_state.salmon_matrices = {
                "counts": result.counts,
                "eff": result.eff_length,
                "fpkm": result.fpkm,
                "records": result.loaded,
                "missing": result.missing,
            }
            msg = (
                f"Loaded {result.fpkm.shape[0]} genes × "
                f"{result.fpkm.shape[1]} samples"
            )
            if result.missing:
                st.warning(
                    msg
                    + f" — {len(result.missing)} sample(s) had no salmon output: "
                    + ", ".join(r.name() for r in result.missing)
                )
            else:
                st.success(msg)
        except Exception as exc:
            logger.exception("load_salmon_matrix crashed")
            st.error(f"Failed to load salmon output: {exc}")

    matrices = st.session_state.get("salmon_matrices")
    if matrices and not matrices["fpkm"].empty:
        st.subheader("FPKM preview (top 20 rows)")
        st.dataframe(matrices["fpkm"].head(20), use_container_width=True)
        st.download_button(
            "Download FPKM TSV",
            data=matrices["fpkm"].to_csv(sep="\t"),
            file_name="phosphotrap_fpkm.tsv",
            mime="text/tab-separated-values",
        )

    st.divider()
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        run_ratio = st.button("Compute IP/Input ratios + Mann-Whitney")
    with bcol2:
        run_anota = st.button("Run anota2seq")
    with bcol3:
        run_deseq = st.button("DESeq2 interaction cross-check")

    if run_ratio:
        if not matrices:
            st.error("Load salmon output first.")
        else:
            ratios = pair_ratios(
                matrices["fpkm"], matrices["records"], min_fpkm=cfg.min_fpkm
            )
            cr = between_group_contrast(ratios, alt_group=alt_group, ref_group=ref_group)
            st.session_state.analysis.setdefault(contrast, {})
            st.session_state.analysis[contrast]["ratios"] = ratios
            st.session_state.analysis[contrast]["contrast_result"] = cr
            st.success(f"Ratios + Mann-Whitney computed for {contrast}")

    if run_anota:
        with st.spinner("Shelling out to anota2seq..."):
            res = run_anota2seq(
                ready_samples,
                alt_group=alt_group,
                ref_group=ref_group,
                salmon_root=cfg.effective_output_dir() / "salmon",
                tx2gene=Path(cfg.tx2gene_tsv),
                cfg=cfg,
                output_dir=cfg.effective_output_dir(),
            )
        st.session_state.analysis.setdefault(contrast, {})
        st.session_state.analysis[contrast]["anota2seq"] = res
        if res.ok:
            st.success(res.message)
        else:
            st.error(res.message)
            st.info(
                "Graceful degradation: install r-base + bioconductor-anota2seq + "
                "bioconductor-tximport via `conda env create -f environment.yml`, "
                "then set the Rscript path in Config."
            )

    if run_deseq:
        with st.spinner("Shelling out to DESeq2..."):
            res = run_deseq2_interaction(
                ready_samples,
                alt_group=alt_group,
                ref_group=ref_group,
                salmon_root=cfg.effective_output_dir() / "salmon",
                tx2gene=Path(cfg.tx2gene_tsv),
                cfg=cfg,
                output_dir=cfg.effective_output_dir(),
            )
        st.session_state.analysis.setdefault(contrast, {})
        st.session_state.analysis[contrast]["deseq2"] = res
        if res.ok:
            st.success(res.message)
        else:
            st.error(res.message)

    # ------------------------------------------------------------------
    # Results panels
    # ------------------------------------------------------------------
    panel = st.session_state.analysis.get(contrast, {})

    cr = panel.get("contrast_result")
    if cr is not None and not cr.table.empty:
        st.subheader(f"Between-group Mann-Whitney — {contrast}")
        st.dataframe(cr.table.head(200), use_container_width=True)
        st.download_button(
            f"Download full table ({contrast})",
            data=cr.table.to_csv(sep="\t"),
            file_name=f"mannwhitney_{contrast}.tsv",
            mime="text/tab-separated-values",
        )
        st.download_button(
            f"Download preranked (.rnk) ({contrast})",
            data=cr.ranked.to_csv(sep="\t", index=False, header=False),
            file_name=f"preranked_{contrast}.rnk",
            mime="text/tab-separated-values",
        )
        try:
            t = cr.table.copy()
            t["-log10_p"] = -np.log10(t["mannwhitney_p"])
            fig = px.scatter(
                t.reset_index(),
                x="delta_log2",
                y="-log10_p",
                hover_data=["gene_id"],
                title=f"Volcano — {contrast}",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            logger.warning("volcano failed: %s", exc)

    ratios = panel.get("ratios")
    if ratios is not None and ratios.ratios.shape[1]:
        st.subheader("Per-group log2(IP/Input) summary")
        for g, df in ratios.per_group.items():
            with st.expander(f"{g} — {df.shape[0]} genes", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
        # Histogram per group
        hist_df = pd.concat(
            [df[["log2_geomean_ratio"]].assign(group=g) for g, df in ratios.per_group.items()]
        )
        fig = px.histogram(
            hist_df,
            x="log2_geomean_ratio",
            color="group",
            nbins=80,
            barmode="overlay",
            title="log2(geomean IP/Input) per group",
        )
        st.plotly_chart(fig, use_container_width=True)

        if cr is not None and not cr.table.empty:
            top30 = cr.table.head(30).reset_index()
            fig2 = px.bar(
                top30,
                x="delta_log2",
                y="gene_id",
                orientation="h",
                title=(
                    f"Top 30 by Mann-Whitney significance — log2 FC shown on x-axis — "
                    f"{contrast}"
                ),
            )
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)

    anota = panel.get("anota2seq")
    if anota is not None and anota.ok:
        st.subheader(f"anota2seq regulatory modes — {contrast}")
        st.write(
            f"translation: {len(anota.translation)} · "
            f"buffering: {len(anota.buffering)} · "
            f"mRNA abundance (both change): {len(anota.mrna_abundance)}"
        )
        with st.expander("translation (IP changes, INPUT does not)", expanded=False):
            st.dataframe(anota.translation, use_container_width=True)
        with st.expander("buffering (INPUT changes, IP compensates)", expanded=False):
            st.dataframe(anota.buffering, use_container_width=True)
        with st.expander("mRNA abundance (both change coherently)", expanded=False):
            st.dataframe(anota.mrna_abundance, use_container_width=True)

    deseq = panel.get("deseq2")
    if deseq is not None and deseq.ok:
        st.subheader(f"DESeq2 interaction cross-check — {contrast}")
        st.dataframe(deseq.table.head(200), use_container_width=True)
        st.download_button(
            f"Download DESeq2 interaction ({contrast})",
            data=deseq.table.to_csv(sep="\t", index=False),
            file_name=f"deseq2_interaction_{contrast}.tsv",
            mime="text/tab-separated-values",
        )

# ======================================================================
# LOGS TAB
# ======================================================================
with tabs[5]:
    st.header("Logs")

    # Staged-clear pattern: if the user hit "Clear filter" on the previous
    # rerun, blank the widget's state BEFORE it renders this run. Setting
    # st.session_state["log_filter"] = "" after the widget is instantiated
    # is a no-op — Streamlit already locked its value.
    if st.session_state.log_filter_staged_clear:
        st.session_state["log_filter"] = ""
        st.session_state.log_filter_staged_clear = False

    fcol, bcol, rcol = st.columns([3, 1, 1])
    with fcol:
        filter_str = st.text_input("Filter (substring, case-insensitive)", key="log_filter")
    with bcol:
        if st.button("Clear filter"):
            st.session_state.log_filter_staged_clear = True
            st.rerun()
    with rcol:
        if st.button("Refresh"):
            st.rerun()

    st.caption(
        "Filter is applied *before* tailing, so with a narrow filter you "
        "may see old matching lines at the top rather than only recent log "
        "activity. Showing up to 800 matching lines across rolled-over backups."
    )

    log_dir = cfg.effective_report_dir() / "logs"
    tail = tail_log(log_dir, max_lines=800, filter_substr=filter_str)
    st.code(tail or "(log empty)", language="text")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "Download current view",
            data=tail,
            file_name="phosphotrap_view.log",
            mime="text/plain",
            help="Exports exactly what's shown above (filtered and tailed).",
        )
    with dcol2:
        # Fetch unfiltered, uncapped content for the full-log button so
        # the label is honest — the previous version silently downloaded
        # whatever the filter was showing.
        full_log = tail_log(log_dir, max_lines=10**9, filter_substr="")
        st.download_button(
            "Download full log",
            data=full_log,
            file_name="phosphotrap.log",
            mime="text/plain",
            help="Exports the entire rolling log, stitched across backups.",
        )

    st.divider()
    st.subheader("Per-sample logs")
    # The pipeline runner writes each sample's fastp+salmon stdout
    # to ``report_dir/logs/per-sample/<name>.log`` so we can surface
    # them here alongside the central app log. Before the recent
    # move, per-sample logs lived directly in report_dir and weren't
    # accessible from the UI.
    per_sample_logs = list_per_sample_logs(cfg.effective_report_dir())
    if not per_sample_logs:
        st.caption(
            "No per-sample logs yet. They appear here after the Pipeline "
            "tab runs fastp or salmon for at least one sample."
        )
    else:
        labels = [p.stem for p in per_sample_logs]
        by_label = dict(zip(labels, per_sample_logs))
        picked = st.selectbox(
            "Sample",
            options=labels,
            key="per_sample_log_picker",
            help="Last 2 MB of the selected sample's fastp+salmon stdout.",
        )
        content = read_log_file(by_label[picked])
        st.code(content or "(sample log empty)", language="text")
        st.download_button(
            f"Download {picked}.log",
            data=content,
            file_name=f"{picked}.log",
            mime="text/plain",
            key=f"download_sample_log_{picked}",
        )
