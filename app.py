"""Streamlit UI for the phosphoribotrap 3-group RNA-seq pipeline.

Run with::

    streamlit run app.py

See README.md for the design rationale — in particular why the QC
"failures" reported by MultiQC are mostly expected IP enrichment signal,
why we do not deduplicate, and why anota2seq is the primary analysis
rather than footprint-oriented tools.
"""

from __future__ import annotations

import shutil
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
    validate_fastq_dir,
    validate_reference_paths,
)
from phosphotrap.deseq2_runner import run_deseq2_interaction
from phosphotrap.figures import (
    DEFAULT_FONT_SIZE as FIG_DEFAULT_FONT_SIZE,
    GALANIN_GENES,
    cross_contrast_scatter,
    expression_heatmap,
    figure_export_bytes,
    load_gene_symbol_map,
    parse_highlight_text,
    per_gene_strip,
    regmode_classification,
    resolve_symbols,
    volcano_plot,
)
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
from phosphotrap.pipeline import (
    check_environment,
    load_pipeline_results,
    run_pipeline,
    save_pipeline_results,
)
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
    # Rehydrate the last saved pipeline run from disk so users who
    # restart streamlit (or reopen the browser tab) don't lose the
    # Pipeline-tab results table. The heavy outputs (fastp / salmon)
    # already survive in the output dir via skip-if-cached, but the
    # in-memory StepResult list — what the table renders — was wiped
    # on every session reset. load_pipeline_results is defensive and
    # returns [] on any read / schema failure.
    try:
        st.session_state.pipeline_results = load_pipeline_results(
            st.session_state.cfg.effective_report_dir()
        )
    except Exception:  # pragma: no cover - defensive
        st.session_state.pipeline_results = []
if "log_filter_staged_clear" not in st.session_state:
    st.session_state.log_filter_staged_clear = False
if "analysis" not in st.session_state:
    st.session_state.analysis = {}  # keyed by contrast
if "_preset_just_applied" not in st.session_state:
    st.session_state._preset_just_applied = False


cfg: AppConfig = st.session_state.cfg
disk_cfg: AppConfig = st.session_state.disk_config


@st.cache_data(show_spinner=False)
def _cached_symbol_map(tx2gene_path: str, mtime: float) -> dict[str, str]:
    """Cached wrapper around ``phosphotrap.figures.load_gene_symbol_map``.

    The raw helper reparses the full tx2gene TSV on every call — for
    a GENCODE mouse M38 build that's ~278 k rows, ~14 MB, and
    ~100-200 ms per call. Without caching, the Figures-tab body
    (which runs on every Streamlit rerun regardless of the active
    tab) would re-parse it on every keystroke in any tab. That's
    enough latency to show up as visible UI lag.

    The ``mtime`` argument is the cache-invalidation key — Streamlit
    re-runs the wrapped function only when the argument tuple
    changes. Passing the file's ``mtime`` alongside its path means
    the cache refreshes automatically when the Reference tab
    rebuilds the reference and writes a new tx2gene.tsv.
    """
    return load_gene_symbol_map(Path(tx2gene_path))


def _figure_download_row(fig, basename: str) -> None:
    """Render a three-button row: HTML / SVG / PNG download for a
    plotly figure.

    HTML export is the always-works fallback (pure Python, no
    external binary). SVG and PNG both go through kaleido — which
    isn't a required dependency, so the buttons silently degrade to
    a caption hint if ``fig.to_image(format=...)`` raises. Callers
    pass a unique ``basename`` (e.g. ``"volcano_HSD1_vs_NCD"``) that
    seeds both the file names AND the Streamlit widget keys so
    multiple download rows on the same rerun don't collide.
    """
    col_html, col_svg, col_png = st.columns(3)
    with col_html:
        st.download_button(
            "Download HTML",
            data=fig.to_html(full_html=True, include_plotlyjs="cdn"),
            file_name=f"{basename}.html",
            mime="text/html",
            key=f"fig_dl_html_{basename}",
        )
    for col, fmt, mime in (
        (col_svg, "svg", "image/svg+xml"),
        (col_png, "png", "image/png"),
    ):
        with col:
            # figure_export_bytes centralises the ``fig.to_image``
            # try/except so the decision logic (success vs. caption
            # fallback) can be unit-tested without a Streamlit
            # runtime. See phosphotrap.figures.figure_export_bytes
            # and tests/test_figures.py for the coverage.
            img_bytes, err = figure_export_bytes(fig, fmt)
            if err is not None:
                st.caption(
                    f"Install `kaleido` for {fmt.upper()} export "
                    f"(`pip install kaleido` or `mamba install -c "
                    f"conda-forge kaleido`). Error: {err}"
                )
                continue
            st.download_button(
                f"Download {fmt.upper()}",
                data=img_bytes,
                file_name=f"{basename}.{fmt}",
                mime=mime,
                key=f"fig_dl_{fmt}_{basename}",
            )

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
    ["Config", "Reference", "Samples", "Pipeline", "Analysis", "Figures", "Logs"]
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

    # Path text_inputs use explicit keys so other tabs (most importantly
    # the Reference tab's "Use these paths in Config" button) can update
    # them via ``st.session_state[key] = ...`` and have the change show
    # up on the next render. Without explicit keys, Streamlit generates
    # its own key from the label + default value, then ignores the
    # ``value=`` argument on subsequent renders — which meant a
    # programmatic update to ``cfg.salmon_index`` from another tab
    # would silently get overwritten by the stale widget value on the
    # next Config-tab render. That's how v5029e4d put a directory path
    # into ``cfg.tx2gene_tsv`` and broke salmon quant + anota2seq.
    _PATH_KEYS = {
        "widget_fastq_dir": "fastq_dir",
        "widget_salmon_index": "salmon_index",
        "widget_tx2gene_tsv": "tx2gene_tsv",
        "widget_output_dir": "output_dir",
        "widget_report_dir": "report_dir",
        "widget_rscript_path": "rscript_path",
    }
    for _wkey, _attr in _PATH_KEYS.items():
        st.session_state.setdefault(_wkey, getattr(cfg, _attr))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Paths")
        st.text_input(
            "Fastq directory",
            key="widget_fastq_dir",
            help="Directory containing the raw *_R1_001.fastq.gz / *_R2_001.fastq.gz files.",
        )
        st.text_input(
            "Salmon index (directory)",
            key="widget_salmon_index",
            help=(
                "Directory containing info.json, pos.bin, seq.bin, … — "
                "the output of `salmon index`. NOT a single file."
            ),
        )
        st.text_input(
            "tx2gene TSV (file, 2- or 3-col)",
            key="widget_tx2gene_tsv",
            help=(
                "Path to the tx2gene.tsv FILE (transcript_id ⇥ gene_id "
                "[⇥ gene_name]). NOT the directory containing it."
            ),
        )
        st.text_input(
            "Output directory",
            key="widget_output_dir",
            help="Where salmon quant outputs land (one subdir per sample).",
        )
        st.text_input(
            "Report/log directory",
            key="widget_report_dir",
            help="Where fastp reports, rolling logs, and per-sample logs land.",
        )
        st.text_input(
            "Rscript binary (file)",
            key="widget_rscript_path",
            help=(
                "Absolute path to the Rscript executable, or just "
                "'Rscript' if it's on PATH."
            ),
        )
    with col_b:
        st.subheader("Runtime")
        # Runtime widgets use the same explicit-key + session-state
        # pattern as the path fields above (fixed in d3b838f) and the
        # anota2seq threshold widgets below. Previously they used the
        # ``cfg.X = st.widget(..., value=cfg.X)`` antipattern, which
        # means a programmatic update from another tab — e.g. a future
        # "retry failed samples" button flipping ``cfg.force_rerun`` —
        # would silently get overwritten by the stale widget value on
        # the next Config-tab render. See the v5029e4d / d3b838f
        # post-mortem on the path fields.
        _LIBTYPE_OPTIONS = ["A", "IU", "ISR", "ISF", "MU", "OU"]
        st.session_state.setdefault("widget_threads", int(cfg.threads))
        st.session_state.setdefault("widget_run_fastp", bool(cfg.run_fastp))
        st.session_state.setdefault("widget_force_rerun", bool(cfg.force_rerun))
        st.session_state.setdefault(
            "widget_salmon_libtype",
            cfg.salmon_libtype if cfg.salmon_libtype in _LIBTYPE_OPTIONS else "A",
        )

        st.number_input(
            "Threads",
            min_value=1,
            max_value=128,
            step=1,
            key="widget_threads",
        )
        st.checkbox("Run fastp trimming", key="widget_run_fastp")
        st.checkbox("Force rerun (ignore cache)", key="widget_force_rerun")
        st.selectbox(
            "Salmon libType",
            options=_LIBTYPE_OPTIONS,
            key="widget_salmon_libtype",
        )

    st.divider()
    st.subheader("Design")

    # Reference group drives the contrasts list. Changing it invalidates
    # any selections held in the multiselect's session state, so we
    # reconcile before rendering — otherwise Streamlit will either crash
    # (older versions) or silently drop the stale values (newer
    # versions). Either way the user would see the wrong thing.
    #
    # Reference-group widget uses the same explicit-key + session-state
    # pattern as the path fields and Runtime widgets. Previously it used
    # ``cfg.reference_group = st.selectbox(..., index=...)`` which (a)
    # is the same antipattern that caused the tx2gene-as-directory bug,
    # and (b) means programmatic updates from other tabs would get
    # silently reverted on the next Config-tab render. See the
    # d3b838f post-mortem.
    st.session_state.setdefault(
        "widget_reference_group",
        cfg.reference_group if cfg.reference_group in GROUPS else GROUPS[0],
    )
    d1, d2 = st.columns(2)
    with d1:
        st.selectbox(
            "Reference group",
            options=list(GROUPS),
            key="widget_reference_group",
            help="Contrast strings on downstream tabs are derived from this.",
        )
        # ``available`` is derived from the FRESH widget value via
        # session state, not from cfg — cfg gets the copy-back at the
        # end of the tab, so during this block it's still showing the
        # value from the previous rerun.
        available = contrasts_for_reference(
            st.session_state["widget_reference_group"], GROUPS
        )

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
    for _wkey, _attr in _PATH_KEYS.items():
        setattr(cfg, _attr, st.session_state[_wkey])
    cfg.threads               = int(st.session_state["widget_threads"])
    cfg.run_fastp             = bool(st.session_state["widget_run_fastp"])
    cfg.force_rerun           = bool(st.session_state["widget_force_rerun"])
    cfg.salmon_libtype        = str(st.session_state["widget_salmon_libtype"])
    cfg.reference_group       = str(st.session_state["widget_reference_group"])
    cfg.anota_delta_pt        = float(st.session_state["widget_anota_delta_pt"])
    cfg.anota_delta_tp        = float(st.session_state["widget_anota_delta_tp"])
    cfg.anota_max_padj        = float(st.session_state["widget_anota_max_padj"])
    cfg.anota_min_slope_trans = float(st.session_state["widget_anota_min_slope_trans"])
    cfg.anota_max_slope_trans = float(st.session_state["widget_anota_max_slope_trans"])
    cfg.min_fpkm              = float(st.session_state["widget_min_fpkm"])

    # Inline validation of the paths that downstream steps actually
    # touch. Catches:
    #   - the "tx2gene_tsv points at a directory" footgun before the
    #     pipeline button is ever clicked (d3b838f, MEDIUM #5)
    #   - fastq_dir that doesn't exist or has no *.fastq.gz files
    #     (HIGH #4 — previously only an emptiness check)
    # Surfaces both through a single error panel so the user sees
    # every issue at once, not one-at-a-time across reruns. Uses the
    # ``validate_reference_paths`` and ``validate_fastq_dir`` helpers
    # so there's a single source of truth shared with the Pipeline,
    # Analysis, and Samples tabs.
    _path_errs: list[str] = []
    _path_errs.extend(
        validate_reference_paths(cfg.salmon_index, cfg.tx2gene_tsv)
    )
    _path_errs.extend(validate_fastq_dir(cfg.fastq_dir))
    if _path_errs:
        st.error(
            "Path validation:\n\n- " + "\n- ".join(_path_errs)
        )

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
        "GENCODE mouse release, a decoy-aware `salmon index` "
        "(a **directory**), and a matching `tx2gene.tsv` (a **file**) — "
        "the two paths the Config tab needs to run the pipeline."
    )

    # Reference-tab widgets use the same explicit-key + setdefault
    # pattern as the Config-tab widgets (d3b838f, 3912527, 7803e36).
    # Passing BOTH ``value=`` and ``key=`` to a Streamlit widget is an
    # anti-pattern that emits ``StreamlitAPIException`` on recent
    # versions and silently breaks programmatic updates on older ones.
    #
    # The default destination is computed ONCE, at first render, from
    # the initial release. If the user later bumps the release from
    # M38 to M39, ``ref_dest`` stays pinned to their existing choice
    # rather than silently flipping under them — safer and more
    # predictable than the recompute-on-every-render behaviour the
    # original code had. The help text nudges the user to bump the
    # destination themselves when they change release.
    st.session_state.setdefault(
        "ref_release", DEFAULT_GENCODE_MOUSE_RELEASE
    )
    st.session_state.setdefault(
        "ref_dest",
        str(
            Path.home()
            / "phosphotrap_refs"
            / f"gencode_mouse_{st.session_state['ref_release']}"
        ),
    )
    st.session_state.setdefault("ref_threads", int(cfg.threads))
    st.session_state.setdefault("ref_force", False)

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.text_input(
            "GENCODE mouse release",
            key="ref_release",
            help=(
                "e.g. M38 (current as of 2025-09-02). Pick the latest "
                "from https://ftp.ebi.ac.uk/pub/databases/gencode/"
                "Gencode_mouse/ — only the release token changes, the "
                "filenames are stable. NOTE: changing this does NOT "
                "auto-update the Destination directory below — bump "
                "it yourself if you want a separate folder per release."
            ),
        )
        st.text_input(
            "Destination directory",
            key="ref_dest",
            help=(
                "Where the downloads, gentrome, salmon index, and "
                "tx2gene.tsv go. ~15 GB total. Reusable across every "
                "mouse RNA-seq project — point it somewhere stable."
            ),
        )
    with rcol2:
        st.number_input(
            "Threads (for salmon index)",
            min_value=1,
            max_value=128,
            step=1,
            key="ref_threads",
        )
        st.checkbox(
            "Force rebuild (ignore cached salmon index)",
            key="ref_force",
            help=(
                "Downloads are still skipped if the .fa.gz / .gtf.gz "
                "files are already on disk. Only the index rebuild is "
                "forced."
            ),
        )

    # Snapshot the current widget values into locals for the rest of
    # the tab. These are live reads from session state, not the
    # widget return values — equivalent but cleaner than relying on
    # the return value of the render call.
    ref_release = st.session_state["ref_release"]
    ref_dest = st.session_state["ref_dest"]
    ref_threads = int(st.session_state["ref_threads"])
    ref_force = bool(st.session_state["ref_force"])

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

    # Pre-flight disk space check. The full build lands ~2 GB of
    # downloads + ~15 GB of index + intermediates. Warn below 25 GB
    # free on the destination's filesystem so the user finds out
    # BEFORE spending an hour downloading into a doomed run.
    try:
        _probe_dir = Path(ref_dest).expanduser()
        while not _probe_dir.exists() and _probe_dir != _probe_dir.parent:
            _probe_dir = _probe_dir.parent
        _free_gb = shutil.disk_usage(_probe_dir).free / 1e9
        if _free_gb < 25:
            st.warning(
                f"Only {_free_gb:.1f} GB free on the destination "
                f"filesystem. A full GENCODE mouse build needs ~20 GB "
                f"of transient space and ~15 GB persistent. Free up "
                f"space or pick a destination on a larger drive."
            )
        else:
            st.caption(f"Disk space OK: {_free_gb:.0f} GB free at {_probe_dir}.")
    except Exception as _exc:  # pragma: no cover - defensive
        st.caption(f"(could not probe disk space: {_exc})")

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
                "n_fasta_transcripts": artifacts.n_fasta_transcripts,
                "tx2gene_coverage": round(artifacts.tx2gene_coverage, 4),
            }
        )

        # Surface a coverage warning inline if the tx2gene doesn't
        # cover ~all of the transcripts in the salmon index's FASTA.
        # 99% is the threshold because GENCODE occasionally ships a
        # handful of transcripts in the FASTA that aren't in the
        # primary-assembly GTF (PAR duplicates, readthrough fusion
        # entries). A few dozen missing is normal; thousands isn't.
        if (
            artifacts.n_fasta_transcripts > 0
            and artifacts.tx2gene_coverage < 0.99
        ):
            _missing = (
                artifacts.n_fasta_transcripts - artifacts.n_transcripts
            )
            st.warning(
                f"tx2gene coverage is "
                f"{artifacts.tx2gene_coverage * 100:.2f}% "
                f"({artifacts.n_transcripts} / "
                f"{artifacts.n_fasta_transcripts}). "
                f"{_missing} transcript(s) in the transcriptome FASTA "
                f"are missing from tx2gene.tsv — salmon quant will "
                f"emit per-missing-transcript warnings for these at "
                f"runtime and they'll fall through as 'transcript as "
                f"its own gene'. For GENCODE mouse this is usually "
                f"<0.1% and harmless, but if the number is large you "
                f"may have a FASTA/GTF release mismatch."
            )
        elif artifacts.n_fasta_transcripts > 0:
            st.caption(
                f"tx2gene coverage "
                f"{artifacts.tx2gene_coverage * 100:.2f}% — OK."
            )

        acol1, acol2 = st.columns(2)
        with acol1:
            if st.button("Use these paths in Config", type="primary"):
                # Write to BOTH the live cfg and the Config tab's
                # text_input session-state keys. Mutating cfg alone is
                # not enough: the Config tab's text_inputs own their
                # session state, and on the next render would read
                # their cached (stale) value and overwrite cfg with
                # it. This is exactly how v5029e4d ended up running
                # salmon quant with `-g <directory>` instead of
                # `-g <dir>/tx2gene.tsv`.
                cfg.salmon_index = str(artifacts.index_dir)
                cfg.tx2gene_tsv = str(artifacts.tx2gene_tsv)
                st.session_state["widget_salmon_index"] = str(
                    artifacts.index_dir
                )
                st.session_state["widget_tx2gene_tsv"] = str(
                    artifacts.tx2gene_tsv
                )
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
            # HIGH #4: "Set a fastq directory in Config first" only
            # fired on an empty string. A typo like
            # /Users/me/Desktop/Raw_Data (missing trailing segment)
            # would silently return "0 / 18 ready samples" with no
            # explanation. validate_fastq_dir now enforces existence,
            # directory-ness, and "at least one *.fastq.gz file inside"
            # — so the user sees a precise error panel instead of
            # guessing why the summary is empty.
            if not cfg.fastq_dir:
                st.error("Set a fastq directory in Config first.")
            else:
                _fq_errs = validate_fastq_dir(cfg.fastq_dir)
                if _fq_errs:
                    st.error(
                        "Cannot auto-populate fastq paths:\n\n- "
                        + "\n- ".join(_fq_errs)
                    )
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

    # Path validation and disk-space preflight. Both run on every
    # render so the user sees the problem immediately, not only after
    # clicking Start pipeline.
    _pipe_errs = validate_reference_paths(cfg.salmon_index, cfg.tx2gene_tsv)
    if _pipe_errs and not dry_run:
        st.error(
            "Cannot start pipeline — reference paths are invalid:\n\n- "
            + "\n- ".join(_pipe_errs)
            + "\n\nBuild or rebuild the index from the **Reference** tab "
            "and click **Use these paths in Config**."
        )

    # Disk-space preflight against the output filesystem. The trimmed
    # fastqs + salmon output for an 18-sample NovaSeq run like this
    # one typically land in the 40-80 GB range, so warn below 50 GB.
    try:
        _out_probe = cfg.effective_output_dir()
        while not _out_probe.exists() and _out_probe != _out_probe.parent:
            _out_probe = _out_probe.parent
        _pipe_free_gb = shutil.disk_usage(_out_probe).free / 1e9
        if _pipe_free_gb < 50 and not dry_run:
            st.warning(
                f"Only {_pipe_free_gb:.1f} GB free on the output "
                f"filesystem at {_out_probe}. Each of the 18 samples "
                f"writes ~2 GB of trimmed fastqs + ~1 GB of salmon "
                f"output, so budget ~50 GB for a complete run. "
                f"Running out of space mid-pipeline corrupts the "
                f"last sample and silently leaves earlier ones "
                f"cached — free space before starting."
            )
    except Exception as _exc:  # pragma: no cover - defensive
        logger.warning("disk-space probe failed: %s", _exc)

    _start_disabled = (not selected and not dry_run) or bool(
        _pipe_errs and not dry_run
    )
    if st.button(
        "Start pipeline", type="primary", disabled=_start_disabled
    ):
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
        # Persist the StepResult list so it survives a streamlit restart
        # or a closed browser tab. save_pipeline_results never raises —
        # on OSError we log-and-swallow so the pipeline run itself is
        # still reported as successful to the user.
        try:
            if results:
                save_pipeline_results(results, cfg.effective_report_dir())
        except OSError as exc:
            logger.warning("could not persist pipeline results: %s", exc)
        st.success(f"Pipeline complete: {len(results)} step(s)")

    # Manual reload: useful when the user has run the pipeline in a
    # previous session and wants the table back without re-running.
    # load_pipeline_results returns [] if nothing has been saved, so
    # the button is always safe to click.
    _reload_col, _clear_col = st.columns([1, 3])
    with _reload_col:
        if st.button("Reload saved results"):
            loaded = load_pipeline_results(cfg.effective_report_dir())
            st.session_state.pipeline_results = loaded
            if loaded:
                st.success(
                    f"Reloaded {len(loaded)} cached pipeline step(s) from "
                    f"{cfg.effective_report_dir() / 'pipeline_results.json'}"
                )
            else:
                st.info(
                    "No saved pipeline results found under "
                    f"{cfg.effective_report_dir()}. Run the pipeline at "
                    "least once to create the cache."
                )

    if st.session_state.pipeline_results:
        rows = [asdict(r) for r in st.session_state.pipeline_results]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Pipeline results are persisted to "
            f"`{cfg.effective_report_dir() / 'pipeline_results.json'}` "
            "and rehydrated on the next streamlit startup. The actual "
            "fastp/salmon outputs are cached under the output directory "
            "and re-used automatically on the next run thanks to "
            "skip-if-cached — so continuing into the Analysis tab does "
            "not recompute the fastq files."
        )

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

    # anota2seq and DESeq2 both read ``tx2gene_tsv`` via tximport, so
    # the same footgun that broke salmon quant (directory path instead
    # of file path) breaks them too. Validate here so the user sees a
    # clear error instead of an R traceback in German.
    _analysis_errs = validate_reference_paths(
        cfg.salmon_index, cfg.tx2gene_tsv
    )
    _r_disabled = bool(_analysis_errs)
    if _analysis_errs:
        st.error(
            "anota2seq and DESeq2 buttons disabled — reference paths "
            "are invalid:\n\n- "
            + "\n- ".join(_analysis_errs)
            + "\n\nFix them on the Config tab (or rebuild via the "
            "Reference tab and click **Use these paths in Config**)."
        )

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        run_ratio = st.button("Compute IP/Input ratios + Mann-Whitney")
    with bcol2:
        run_anota = st.button("Run anota2seq", disabled=_r_disabled)
    with bcol3:
        run_deseq = st.button(
            "DESeq2 interaction cross-check", disabled=_r_disabled
        )

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
# FIGURES TAB
# ======================================================================
with tabs[5]:
    st.header("Figures — Nature-grade galanin signaling panels")
    st.caption(
        "Publication-ready figures built from the current Analysis-tab "
        "state. Galanin core is highlighted by default; add your own "
        "genes to the text area below. Every panel renders as an "
        "interactive plotly chart with SVG / PNG / HTML download buttons."
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    # Explicit widget keys so other tabs could programmatically seed
    # values in future (same pattern as the Config tab — see d3b838f
    # and the REQUIRED_KEYS list in tests/test_app_widget_keys.py
    # once this tab's keys are added there too in chunk 8).
    st.session_state.setdefault("widget_fig_custom_highlights", "")
    st.session_state.setdefault(
        "widget_fig_font_size", FIG_DEFAULT_FONT_SIZE
    )
    st.session_state.setdefault("widget_fig_alpha", 0.1)
    st.session_state.setdefault("widget_fig_heatmap_norm", "zscore")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Galanin core** (always highlighted):")
        st.code(", ".join(GALANIN_GENES), language="text")
        st.text_area(
            "Additional highlight genes "
            "(comma, whitespace, or newline separated)",
            key="widget_fig_custom_highlights",
            height=80,
            help=(
                "Any gene symbol in your tx2gene.tsv. Case-insensitive. "
                "Try e.g. 'Bdnf, Npy, Pomc' to overlay a neuropeptide "
                "control set on the galanin panels."
            ),
        )
    with cc2:
        st.slider(
            "Font size (pt)",
            min_value=8, max_value=22, step=1,
            key="widget_fig_font_size",
            help=(
                "14 is the screen default. Drop to 8 for final-print "
                "panels; bump to 18–22 for slides and posters."
            ),
        )
        st.slider(
            "Significance threshold (padj)",
            min_value=0.01, max_value=0.50, step=0.01,
            key="widget_fig_alpha",
            help=(
                "Horizontal dashed line on the volcano. Genes with "
                "padj ≤ this threshold are drawn in the 'significant' "
                "colour. Chronic-stimulus preset uses 0.1."
            ),
        )
        st.selectbox(
            "Heatmap normalization",
            options=["zscore", "log2", "raw"],
            key="widget_fig_heatmap_norm",
            help=(
                "zscore (row-wise, diverging RdBu) for 'which samples "
                "deviate from the row mean'; log2(FPKM+1) for expression "
                "levels; raw FPKM for direct inspection."
            ),
        )

    font_size = int(st.session_state["widget_fig_font_size"])
    alpha = float(st.session_state["widget_fig_alpha"])
    heatmap_norm = str(st.session_state["widget_fig_heatmap_norm"])
    custom_text = st.session_state["widget_fig_custom_highlights"]

    # ------------------------------------------------------------------
    # Gene resolution via tx2gene
    # ------------------------------------------------------------------
    # The volcano / strip / heatmap / regmode table all need a
    # symbol -> gene_id map to translate user-facing labels into the
    # Ensembl IDs the salmon matrices and contrast tables are indexed
    # by. The tx2gene TSV is the single source of truth — we reuse
    # the same file the Pipeline and Analysis tabs point at.
    _fig_path_errs = validate_reference_paths(
        cfg.salmon_index, cfg.tx2gene_tsv
    )
    if _fig_path_errs:
        st.error(
            "Figures need a valid tx2gene TSV to resolve gene symbols.\n\n- "
            + "\n- ".join(_fig_path_errs)
            + "\n\nFix them on the Config tab (or rebuild via the "
            "Reference tab and click **Use these paths in Config**)."
        )
    else:
        try:
            # Cached by (path, mtime) so the 278 k-row tx2gene isn't
            # re-parsed on every rerun. See ``_cached_symbol_map`` for
            # the rationale.
            _mtime = Path(cfg.tx2gene_tsv).stat().st_mtime
            _symbol_map = _cached_symbol_map(cfg.tx2gene_tsv, _mtime)
        except FileNotFoundError as exc:
            _symbol_map = {}
            st.error(f"Could not read tx2gene: {exc}")

        # LOW #7: when the symbol map is empty (2-column tx2gene — no
        # gene_name column), a previous version emitted TWO warnings
        # stacked: one for the 2-column format and a second one from
        # resolve_symbols(GALANIN_GENES, {}) saying "Galanin genes not
        # found in tx2gene: Gal, Galp, Galr1, Galr2, Galr3". The second
        # warning was technically correct but redundant and alarming.
        # Now: when the symbol map is empty, emit ONE targeted warning
        # and short-circuit the resolution to empty dicts — the per-
        # panel empty-set guards below handle the rendering without
        # raising.
        if not _symbol_map:
            st.warning(
                "tx2gene TSV has no gene-symbol column (2-column "
                "format). Rebuild from the Reference tab to get a "
                "3-column tx2gene with gene names — otherwise the "
                "highlight fields can't map symbols to gene IDs."
            )
            galanin_resolved: dict[str, str] = {}
            galanin_missing: list[str] = []
            custom_resolved: dict[str, str] = {}
            custom_missing = []
        else:
            galanin_resolved, galanin_missing = resolve_symbols(
                GALANIN_GENES, _symbol_map
            )
            custom_symbols = parse_highlight_text(custom_text)
            custom_resolved, custom_missing = resolve_symbols(
                custom_symbols, _symbol_map
            )

            # Surface resolution status so the user sees typos
            # immediately. Only rendered in the symbol-map-present
            # path — when the map is empty we've already warned above
            # and showing "0 / 5 resolved" would just be noise.
            _resolved_cols = st.columns(2)
            with _resolved_cols[0]:
                if galanin_resolved:
                    st.caption(
                        f"✅ Galanin resolved: "
                        f"{len(galanin_resolved)} / {len(GALANIN_GENES)}"
                    )
                if galanin_missing:
                    st.warning(
                        "Galanin genes not found in tx2gene: "
                        + ", ".join(galanin_missing)
                    )
            with _resolved_cols[1]:
                if custom_resolved:
                    st.caption(
                        f"✅ Custom resolved: {len(custom_resolved)}"
                    )
                if custom_missing:
                    st.warning(
                        "Custom genes not found in tx2gene: "
                        + ", ".join(custom_missing)
                    )

        # Merged gene set for the per-gene panels (B and C).
        all_highlighted = {**galanin_resolved, **custom_resolved}
        primary_ids = set(galanin_resolved.keys())

        # --------------------------------------------------------------
        # Data-readiness check — need at least the salmon matrices and
        # one contrast's Mann-Whitney result. Panels degrade gracefully
        # when upstream bits are missing.
        # --------------------------------------------------------------
        _matrices = st.session_state.get("salmon_matrices")
        _analysis = st.session_state.get("analysis", {})

        if not _matrices:
            st.info(
                "Load salmon quant outputs from the **Analysis** tab "
                "first, then click **Compute IP/Input ratios + "
                "Mann-Whitney** for each contrast you want in the "
                "figures. Run **anota2seq** too for the regulatory-"
                "mode classification table."
            )
        else:
            # Sort contrasts alphabetically so panel A (volcanoes) and
            # panel E (cross-contrast scatter) have a deterministic
            # order regardless of which contrast the user loaded first
            # on the Analysis tab. LOW #6: the previous behaviour was
            # dict-insertion order, which depended on click sequence —
            # fine in practice but implicit. ``sorted()`` on the
            # contrast name strings gives HSD1_vs_NCD before
            # HSD3_vs_HSD1 before HSD3_vs_NCD (lexicographic), which
            # matches the natural reading order for this design.
            contrasts_with_results = sorted(
                c for c, panel in _analysis.items()
                if panel.get("contrast_result") is not None
            )

            st.divider()

            # ----------------------------------------------------------
            # Panel A — Volcano plots (one per configured contrast)
            # ----------------------------------------------------------
            st.subheader("A — Volcano plots")
            st.caption(
                "x: delta log₂(IP/Input) (alt − ref). "
                "y: −log₁₀(Mann-Whitney p). "
                "Crimson = galanin core, blue = custom highlights. "
                "Dashed horizontal line marks the padj threshold."
            )
            if not contrasts_with_results:
                st.info(
                    "No contrast Mann-Whitney results yet. Run "
                    "**Compute IP/Input ratios + Mann-Whitney** on the "
                    "Analysis tab for at least one contrast."
                )
            else:
                for _contrast_name in contrasts_with_results:
                    _cr = _analysis[_contrast_name]["contrast_result"]
                    _fig = volcano_plot(
                        _cr.table,
                        title=f"Volcano — {_contrast_name}",
                        alpha=alpha,
                        highlight_primary=galanin_resolved,
                        highlight_secondary=custom_resolved,
                        font_size=font_size,
                    )
                    st.plotly_chart(_fig, use_container_width=True)
                    _figure_download_row(
                        _fig, f"volcano_{_contrast_name}"
                    )

            st.divider()

            # ----------------------------------------------------------
            # Panel B — Per-gene log2(IP/Input) strip plot
            # ----------------------------------------------------------
            st.subheader("B — Per-gene log₂(IP/Input)")
            st.caption(
                "Every animal as an individual dot, grouped by diet. "
                "Short black bars mark group means. This is the panel "
                "reviewers will ask for — it shows the effect is "
                "present at the animal level, not just in aggregate."
            )
            _any_ratios = None
            for _panel in _analysis.values():
                if _panel.get("ratios") is not None:
                    _any_ratios = _panel["ratios"]
                    break
            if _any_ratios is None:
                st.info(
                    "Needs ratios from **Compute IP/Input ratios + "
                    "Mann-Whitney** on the Analysis tab."
                )
            elif not all_highlighted:
                st.info(
                    "No resolved highlight genes. Check the galanin "
                    "resolution status above — if galanin genes aren't "
                    "in your tx2gene, rebuild from the Reference tab."
                )
            else:
                _fig = per_gene_strip(
                    _any_ratios.ratios,
                    _any_ratios.pair_labels,
                    title="Galanin signaling — log₂(IP/Input)",
                    gene_labels=all_highlighted,
                    primary_ids=primary_ids,
                    font_size=font_size,
                    group_order=list(GROUPS),
                )
                st.plotly_chart(_fig, use_container_width=True)
                _figure_download_row(_fig, "per_gene_strip")

            st.divider()

            # ----------------------------------------------------------
            # Panel C — Expression heatmap
            # ----------------------------------------------------------
            st.subheader("C — Expression heatmap")
            st.caption(
                "Rows: genes. Columns: the 18 samples grouped by "
                "(group × fraction). Default z-score across rows — "
                "switch to log₂(FPKM+1) or raw FPKM in the controls "
                "above if you want expression levels instead of "
                "deviation patterns."
            )
            if _matrices["fpkm"].empty:
                st.info("salmon FPKM matrix is empty.")
            elif not all_highlighted:
                st.info("No resolved highlight genes.")
            else:
                _fig = expression_heatmap(
                    _matrices["fpkm"],
                    _matrices["records"],
                    title="Galanin signaling expression",
                    gene_labels=all_highlighted,
                    font_size=font_size,
                    normalize=heatmap_norm,
                    group_order=list(GROUPS),
                )
                st.plotly_chart(_fig, use_container_width=True)
                _figure_download_row(_fig, "heatmap")

            st.divider()

            # ----------------------------------------------------------
            # Panel D — anota2seq regulatory mode table
            # ----------------------------------------------------------
            st.subheader("D — anota2seq regulatory mode")
            st.caption(
                "Per gene × per contrast. Translation hits float to "
                "the top — those are the genes whose ribosome "
                "association genuinely changed without a matching "
                "total-mRNA change. Needs a successful anota2seq run "
                "on the Analysis tab."
            )
            _anota_results = {
                c: panel["anota2seq"]
                for c, panel in _analysis.items()
                if panel.get("anota2seq") is not None
                and getattr(panel["anota2seq"], "ok", False)
            }
            if not _anota_results:
                st.info(
                    "Run **anota2seq** on the Analysis tab for at "
                    "least one contrast to populate this table."
                )
            elif not all_highlighted:
                st.info("No resolved highlight genes.")
            else:
                _regmode_df = regmode_classification(
                    _anota_results, all_highlighted
                )
                st.dataframe(
                    _regmode_df,
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button(
                    "Download regmode table (TSV)",
                    data=_regmode_df.to_csv(sep="\t", index=False),
                    file_name="galanin_regmode.tsv",
                    mime="text/tab-separated-values",
                    key="fig_regmode_tsv_dl",
                )

            st.divider()

            # ----------------------------------------------------------
            # Panel E — Cross-contrast consistency scatter
            # ----------------------------------------------------------
            st.subheader("E — Cross-contrast consistency")
            st.caption(
                "log₂ FC from the first two loaded contrasts (sorted "
                "alphabetically) on x and y. Points near the diagonal "
                "are consistent across both HSD durations — the most "
                "persuasive reproducibility evidence for a chronic-"
                "stimulus n=3-per-group design."
            )
            if len(contrasts_with_results) < 2:
                st.info(
                    "Needs Mann-Whitney results for at least two "
                    "contrasts (e.g. HSD1_vs_NCD AND HSD3_vs_NCD)."
                )
            else:
                # contrasts_with_results is already sorted
                # alphabetically above (LOW #6), so panel E deterministically
                # picks the same pair every rerun. For the default
                # 3-group design this is HSD1_vs_NCD vs HSD3_vs_HSD1 if
                # both are loaded — or HSD1_vs_NCD vs HSD3_vs_NCD if
                # only the two primary contrasts are loaded, which is
                # the typical case.
                _name_a, _name_b = contrasts_with_results[:2]
                st.caption(f"Showing: **{_name_a}** (x) vs **{_name_b}** (y).")
                _fig = cross_contrast_scatter(
                    _analysis[_name_a]["contrast_result"].table,
                    _analysis[_name_b]["contrast_result"].table,
                    label_a=_name_a,
                    label_b=_name_b,
                    title=f"{_name_a}  vs  {_name_b}",
                    highlight_primary=galanin_resolved,
                    highlight_secondary=custom_resolved,
                    font_size=font_size,
                )
                st.plotly_chart(_fig, use_container_width=True)
                _figure_download_row(_fig, "cross_contrast")

# ======================================================================
# LOGS TAB
# ======================================================================
with tabs[6]:
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
