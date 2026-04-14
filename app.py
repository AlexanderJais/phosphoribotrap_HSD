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
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from phosphotrap.anota2seq_runner import run_anota2seq
from phosphotrap.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    contrasts_for_reference,
)
from phosphotrap.deseq2_runner import run_deseq2_interaction
from phosphotrap.fpkm import (
    between_group_contrast,
    load_salmon_matrix,
    pair_ratios,
)
from phosphotrap.logger import attach_file_handler, get_logger, tail_log
from phosphotrap.pipeline import check_environment, run_pipeline
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
if "progress_value" not in st.session_state:
    st.session_state.progress_value = 0.0
if "progress_msg" not in st.session_state:
    st.session_state.progress_msg = ""
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = []
if "log_filter_staged_clear" not in st.session_state:
    st.session_state.log_filter_staged_clear = False
if "analysis" not in st.session_state:
    st.session_state.analysis = {}  # keyed by contrast


cfg: AppConfig = st.session_state.cfg
disk_cfg: AppConfig = st.session_state.disk_config

# Attach the file handler to the configured report dir. Idempotent, so
# Streamlit reruns don't stack handlers; if the user changes report_dir
# and saves, the next rerun rotates the handler onto the new location.
attach_file_handler(Path(cfg.report_dir) / "logs")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
st.title("Phosphoribotrap RNA-seq — 3-group (NCD / HSD1 / HSD3)")

tabs = st.tabs(["Config", "Samples", "Pipeline", "Analysis", "Logs"])

# ======================================================================
# CONFIG TAB
# ======================================================================
with tabs[0]:
    st.header("Configuration")

    unsaved = cfg.diff(disk_cfg)
    if unsaved:
        st.warning(
            f"Unsaved changes vs. {DEFAULT_CONFIG_PATH}: "
            + ", ".join(sorted(unsaved.keys()))
        )
    else:
        st.success(f"Config in sync with {DEFAULT_CONFIG_PATH}")

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

    # Reference group drives the contrasts list. Changing it resets the
    # available contrast options; downstream tabs read ``cfg.contrasts``
    # for whatever the user currently has selected.
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
        cfg.contrasts = st.multiselect(
            "Contrasts",
            options=available,
            default=[c for c in cfg.contrasts if c in available] or available[:2],
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
            st.rerun()

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
    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        if st.button("Save config", type="primary"):
            path = cfg.save(DEFAULT_CONFIG_PATH)
            st.session_state.disk_config = AppConfig.load(DEFAULT_CONFIG_PATH)
            st.success(f"Saved {path}")
            logger.info("config saved to %s", path)
    with bcol2:
        if st.button("Check environment"):
            env = check_environment(cfg.rscript_path or "Rscript")
            rows = [{"tool": k, **v} for k, v in env.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ======================================================================
# SAMPLES TAB
# ======================================================================
with tabs[1]:
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
                recs = to_records(st.session_state.sample_df)
                populate_fastq_paths(recs, Path(cfg.fastq_dir))
                st.session_state.sample_df = records_to_df(recs)
                st.rerun()
    with c3:
        st.download_button(
            "Download sheet (TSV)",
            data=st.session_state.sample_df.to_csv(sep="\t", index=False),
            file_name="phosphotrap_samples.tsv",
            mime="text/tab-separated-values",
        )

    st.subheader("Summary")
    records = to_records(st.session_state.sample_df)
    ready = ready_records(records)
    st.json(summary(records))
    st.write(f"**{len(ready)} / {len(records)} samples have existing R1 + R2 fastq files.**")
    by_group = pairs_by_group(ready)
    for g in GROUPS:
        n = len(by_group.get(g, []))
        st.write(f"- {g}: {n} matched IP/INPUT pair(s)")

# ======================================================================
# PIPELINE TAB
# ======================================================================
with tabs[2]:
    st.header("Pipeline — fastp (optional) + salmon")
    st.caption(
        "Skip-if-cached by default; use the force_rerun checkbox in Config to override. "
        "No dedup step — IP enrichment drives the duplication rate, do not strip it."
    )

    recs = to_records(st.session_state.sample_df)
    ready = ready_records(recs)
    options = [r.name() for r in ready]
    selected_names = st.multiselect(
        "Samples to run",
        options=options,
        default=options,
        help="Only samples whose R1+R2 fastq paths already exist are listed.",
    )
    selected = [r for r in ready if r.name() in selected_names]

    dry_run = st.checkbox("Dry run (environment check only)")

    progress_bar = st.progress(st.session_state.progress_value, text=st.session_state.progress_msg)

    if st.button("Start pipeline", type="primary", disabled=not selected and not dry_run):
        # Auto-save current config first.
        cfg.save(DEFAULT_CONFIG_PATH)
        st.session_state.disk_config = AppConfig.load(DEFAULT_CONFIG_PATH)

        def cb(sample_idx: int, step_idx: int, frac: float, msg: str) -> None:
            frac = max(0.0, min(1.0, float(frac)))
            st.session_state.progress_value = frac
            st.session_state.progress_msg = msg
            progress_bar.progress(frac, text=msg)

        with st.spinner("Running pipeline..."):
            try:
                results = run_pipeline(
                    selected,
                    salmon_index=Path(cfg.salmon_index),
                    tx2gene=Path(cfg.tx2gene_tsv),
                    output_dir=Path(cfg.output_dir),
                    report_dir=Path(cfg.report_dir),
                    threads=cfg.threads,
                    run_fastp_step=cfg.run_fastp,
                    libtype=cfg.salmon_libtype,
                    force=cfg.force_rerun,
                    progress_cb=cb,
                    dry_run=dry_run,
                    rscript_path=cfg.rscript_path or "Rscript",
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
with tabs[3]:
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
    alt_group, ref_group = contrast.split("_vs_") if contrast else (None, None)

    if st.button("Load salmon quant outputs"):
        recs = to_records(st.session_state.sample_df)
        ready = ready_records(recs)
        try:
            counts, eff, fpkm = load_salmon_matrix(
                ready, Path(cfg.output_dir) / "salmon"
            )
            st.session_state.setdefault("salmon_matrices", {})
            st.session_state.salmon_matrices = {
                "counts": counts, "eff": eff, "fpkm": fpkm, "records": ready,
            }
            st.success(f"Loaded {fpkm.shape[0]} genes × {fpkm.shape[1]} samples")
        except Exception as exc:
            st.error(f"Failed to load salmon output: {exc}")

    matrices = st.session_state.get("salmon_matrices")
    if matrices:
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
        recs = to_records(st.session_state.sample_df)
        ready = ready_records(recs)
        with st.spinner("Shelling out to anota2seq..."):
            res = run_anota2seq(
                ready,
                alt_group=alt_group,
                ref_group=ref_group,
                salmon_root=Path(cfg.output_dir) / "salmon",
                tx2gene=Path(cfg.tx2gene_tsv),
                cfg=cfg,
                output_dir=Path(cfg.output_dir),
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
        recs = to_records(st.session_state.sample_df)
        ready = ready_records(recs)
        with st.spinner("Shelling out to DESeq2..."):
            res = run_deseq2_interaction(
                ready,
                alt_group=alt_group,
                ref_group=ref_group,
                salmon_root=Path(cfg.output_dir) / "salmon",
                tx2gene=Path(cfg.tx2gene_tsv),
                cfg=cfg,
                output_dir=Path(cfg.output_dir),
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
            t["-log10_p"] = -np.log10(t["mannwhitney_p"].replace(0, np.nan))
            fig = px.scatter(
                t.reset_index().rename(columns={"index": "gene_id"}),
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
            top30 = cr.table.head(30).reset_index().rename(columns={"index": "gene_id"})
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
with tabs[4]:
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

    tail = tail_log(Path(cfg.report_dir) / "logs", max_lines=800, filter_substr=filter_str)
    st.code(tail or "(log empty)", language="text")
    st.download_button(
        "Download full log",
        data=tail,
        file_name="phosphotrap.log",
        mime="text/plain",
    )
