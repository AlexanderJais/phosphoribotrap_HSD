"""Regression tests for ``app.py`` structure and widget-key conventions.

The HIGH-priority bugs that caused the first live pipeline crash all
shared one root cause: Streamlit text_inputs / checkboxes / selectboxes
without explicit ``key=`` parameters cache their own session state and
silently overwrite ``cfg`` updates from other tabs.

These are static-analysis tests over ``app.py`` — they don't import or
execute Streamlit. The point is to fail loudly if someone refactors
any of the tabs and accidentally drops a ``key=`` from a widget that
needs to be programmatically updatable, or reorders / renames / deletes
a tab, or guts a panel block from the Figures tab.

Three regression guards live here:

* ``REQUIRED_KEYS`` — widget keys that MUST stay explicit.
* ``EXPECTED_TABS`` — tab names in their positional order (the
  ``with tabs[N]:`` blocks are index-based, so a reorder without
  matching index updates would silently mis-route rendering).
* A source-scan smoke test for the Figures tab body, checking that
  the five publication panels (A-E) are still wired up and the
  download-row helper is still called.

If a test fails after a legitimate refactor, update the matching
constant below — but think twice before doing so. Each entry was
added because losing the invariant would resurrect a class of bug
we already paid for once.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_PY = Path(__file__).resolve().parent.parent / "app.py"


# Widget keys that MUST exist somewhere in app.py with the explicit
# ``key="..."`` form. Each one corresponds to a cfg field that is
# (a) updated programmatically from another tab, or (b) at risk of
# being silently overwritten by stale widget session state if the
# explicit-key pattern is dropped.
REQUIRED_KEYS = (
    # Path fields — fixed in d3b838f after the tx2gene-as-directory bug.
    # The Reference tab's "Use these paths in Config" button writes to
    # widget_salmon_index and widget_tx2gene_tsv directly via session
    # state, so dropping the keys would resurrect the original crash.
    "widget_fastq_dir",
    "widget_salmon_index",
    "widget_tx2gene_tsv",
    "widget_output_dir",
    "widget_report_dir",
    "widget_rscript_path",
    # Runtime fields — fixed in 3912527 (HIGH #1).
    "widget_threads",
    "widget_run_fastp",
    "widget_force_rerun",
    "widget_salmon_libtype",
    # Reference group selectbox — fixed in 7803e36 (HIGH #2).
    "widget_reference_group",
    # anota2seq threshold widgets — original explicit-key adopters.
    # The "Apply chronic-stimulus preset" button writes to these
    # session-state keys, so they MUST stay explicit.
    "widget_anota_delta_pt",
    "widget_anota_delta_tp",
    "widget_anota_max_padj",
    "widget_anota_min_slope_trans",
    "widget_anota_max_slope_trans",
    "widget_min_fpkm",
    # Reference tab — fixed in 120f50c (MEDIUM #8).
    "ref_release",
    "ref_dest",
    "ref_threads",
    "ref_force",
    # Figures tab — added alongside the tab itself.
    "widget_fig_primary_highlights",
    "widget_fig_custom_highlights",
    "widget_fig_font_size",
    "widget_fig_alpha",
    "widget_fig_heatmap_norm",
    # Volcano background filter (GO-term curated gene set).
    "widget_fig_volcano_filter_enabled",
    "widget_fig_volcano_filter_text",
    "widget_fig_volcano_go_id",
    "widget_fig_volcano_go_taxon",
)


def _app_source() -> str:
    return APP_PY.read_text()


def test_app_py_exists():
    assert APP_PY.exists(), f"app.py not found at {APP_PY}"


def test_all_required_widget_keys_present():
    """Every key in REQUIRED_KEYS must appear with the ``key="X"`` form."""
    src = _app_source()
    missing: list[str] = []
    for key in REQUIRED_KEYS:
        # Match either single or double quotes around the key value.
        pattern = rf'key=["\']{re.escape(key)}["\']'
        if not re.search(pattern, src):
            missing.append(key)
    assert not missing, (
        "Widget keys missing the explicit ``key=`` form in app.py: "
        + ", ".join(missing)
        + "\n\nThis is the same class of bug that caused the "
        "tx2gene-as-directory crash on the first live pipeline run "
        "(d3b838f). See the comment block above ``_PATH_KEYS`` in "
        "app.py and the audit notes in commit 3912527 / 7803e36."
    )


def test_no_widget_uses_value_and_key_together():
    """Mixing ``value=`` and ``key=`` on the same widget is a Streamlit
    anti-pattern: ``value=`` only takes effect on first render, then
    Streamlit ignores it and uses session state. Recent Streamlit
    versions raise ``StreamlitAPIException`` for this combination.

    This is a heuristic check — it scans for st.<widget>(...) blocks
    that contain BOTH ``value=`` and ``key=`` between matching parens.
    Not a perfect parse, but covers the common cases.
    """
    src = _app_source()
    # Match st.text_input/number_input/checkbox/selectbox(...) with
    # balanced parens. Use the DOTALL flag so multi-line widget calls
    # are captured.
    widget_re = re.compile(
        r"st\.(?:text_input|number_input|checkbox|selectbox)\s*\((.*?)\)\s*$",
        re.DOTALL | re.MULTILINE,
    )
    # Match the BARE ``value=`` parameter, not ``min_value=`` /
    # ``max_value=`` / ``default_value=``. The negative lookbehind
    # asserts that the character before ``value`` is not an identifier
    # character (letter / digit / underscore).
    bare_value_re = re.compile(r"(?<![A-Za-z0-9_])value\s*=")
    bare_key_re = re.compile(r"(?<![A-Za-z0-9_])key\s*=")

    offenders: list[str] = []
    for match in widget_re.finditer(src):
        body = match.group(1)
        if bare_value_re.search(body) and bare_key_re.search(body):
            # Compute the line number for a useful failure message.
            line_no = src[: match.start()].count("\n") + 1
            preview = body.replace("\n", " ").strip()[:80]
            offenders.append(f"app.py:{line_no} — {preview}...")

    assert not offenders, (
        "Widgets passing both ``value=`` and ``key=`` (Streamlit "
        "anti-pattern, breaks programmatic updates):\n"
        + "\n".join("  " + o for o in offenders)
        + "\n\nUse st.session_state.setdefault(key, default) before "
        "the widget instead, then render with key=... only."
    )


# ----------------------------------------------------------------------
# Tab order + Figures-tab smoke test (TEST #13 + #10 from audit)
# ----------------------------------------------------------------------

# The seven tabs in their positional order. ``with tabs[N]:`` blocks
# in app.py are index-based, so a reorder without matching index
# updates silently mis-routes rendering (e.g. Logs content rendering
# into the Figures tab slot). This tuple is the single source of
# truth — update it only alongside a deliberate tab reshuffle.
EXPECTED_TABS = (
    "Config",
    "Reference",
    "Samples",
    "Pipeline",
    "Analysis",
    "Figures",
    "Logs",
)


def test_tab_names_and_order_match_expected():
    """``st.tabs([...])`` must contain the seven expected tab names
    in the expected positional order. A mismatch almost always
    indicates a refactor that reordered tabs without updating the
    matching ``with tabs[N]:`` indices below."""
    src = _app_source()
    # Match the st.tabs([...]) call — accepts multi-line list.
    m = re.search(r"st\.tabs\(\s*\[([^\]]+)\]\s*\)", src)
    assert m, "No st.tabs([...]) call found in app.py"
    inside = m.group(1)
    # Extract the quoted tab names in order.
    names = re.findall(r'["\']([^"\']+)["\']', inside)
    assert tuple(names) == EXPECTED_TABS, (
        f"Tab names / order diverged from EXPECTED_TABS:\n"
        f"  expected: {EXPECTED_TABS}\n"
        f"  got:      {tuple(names)}\n\n"
        "If this is a deliberate reshuffle, update EXPECTED_TABS "
        "AND every ``with tabs[N]:`` index in app.py."
    )


def test_each_tab_index_has_a_with_block():
    """Every tab index 0..N-1 must have a matching ``with tabs[N]:``
    block. Otherwise the tab renders as a blank panel and the user
    clicks it once, sees nothing, and closes the app."""
    src = _app_source()
    indices = set(
        int(m.group(1))
        for m in re.finditer(r"with\s+tabs\[(\d+)\]\s*:", src)
    )
    expected_indices = set(range(len(EXPECTED_TABS)))
    missing = expected_indices - indices
    assert not missing, (
        f"Tab indices present in st.tabs([...]) but missing a "
        f"matching ``with tabs[N]:`` body: {sorted(missing)}. "
        "Every tab index needs a body block — otherwise the tab "
        "renders as a blank panel."
    )
    extra = indices - expected_indices
    assert not extra, (
        f"``with tabs[N]:`` blocks found for indices beyond the "
        f"declared tab count: {sorted(extra)}. These would render "
        "no content and are almost always leftover from a deleted "
        "tab whose index wasn't cleaned up."
    )


# Required structural elements of the Figures tab body. These are
# the five publication panels plus the download-row helper call.
# The subheader strings are exactly what ``st.subheader(...)``
# renders, so if someone silently deletes one they fail this test
# instead of producing a Figures tab with four panels instead of
# five.
FIGURES_TAB_REQUIRED_MARKERS = (
    # Panel subheaders.
    "A — Volcano plots",
    "B — Per-gene log₂(IP/Input)",
    "C — Expression heatmap",
    "D — anota2seq regulatory mode",
    "E — Cross-contrast consistency",
    # Plot-helper call sites — one per panel except panel D (which
    # uses a DataFrame/st.dataframe, not a plotly chart).
    "volcano_plot(",
    "per_gene_strip(",
    "expression_heatmap(",
    "regmode_classification(",
    "cross_contrast_scatter(",
    # The download-row helper must be invoked for every chart panel
    # so users can grab HTML / SVG / PNG copies.
    "_figure_download_row(",
)


def test_figures_tab_body_has_all_five_panels():
    """TEST #10 from the audit: smoke test that the Figures tab body
    still contains the A-E panel subheaders, the plot-helper calls,
    and at least one _figure_download_row invocation. Guards against
    accidentally gutting the tab during a future refactor."""
    src = _app_source()
    missing = [
        marker for marker in FIGURES_TAB_REQUIRED_MARKERS
        if marker not in src
    ]
    assert not missing, (
        "Figures tab body is missing required panel markers / "
        "helper calls:\n  "
        + "\n  ".join(missing)
        + "\n\nIf this is a deliberate refactor, update "
        "FIGURES_TAB_REQUIRED_MARKERS to match. Otherwise the "
        "Figures tab has lost a publication panel and the "
        "manuscript figure is incomplete."
    )


def test_figures_tab_download_row_called_for_every_chart_panel():
    """Four of the five panels (A, B, C, E) use a plotly chart and
    should each feed _figure_download_row. Panel D is a DataFrame
    so it uses a direct st.download_button instead. The chart count
    on panel A varies (one volcano per contrast) so we just require
    at least four _figure_download_row calls total — one per
    non-tabular panel plus at least one per-contrast volcano."""
    src = _app_source()
    # Grab just the Figures-tab body to avoid counting matches in
    # the helper's own docstring.
    body_match = re.search(
        r"# FIGURES TAB\s*\n# ={10,}\s*\nwith tabs\[\d+\]:(.*?)"
        r"(?=\n# ={10,}\s*\n# [A-Z]+ TAB)",
        src,
        flags=re.DOTALL,
    )
    assert body_match, "Could not locate Figures tab body in app.py"
    body = body_match.group(1)
    n_calls = len(re.findall(r"_figure_download_row\(", body))
    assert n_calls >= 4, (
        f"Expected at least 4 _figure_download_row calls in the "
        f"Figures tab body (one per chart panel: volcano, strip, "
        f"heatmap, cross-contrast), found {n_calls}."
    )
