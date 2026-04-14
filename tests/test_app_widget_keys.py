"""Regression tests for app.py widget-key conventions.

The HIGH-priority bugs that caused the first live pipeline crash all
shared one root cause: Streamlit text_inputs / checkboxes / selectboxes
without explicit ``key=`` parameters cache their own session state and
silently overwrite ``cfg`` updates from other tabs.

These are static-analysis tests over ``app.py`` — they don't import or
execute Streamlit. The point is to fail loudly if someone refactors
the Config or Reference tab and accidentally drops a ``key=`` from one
of the widgets that needs to be programmatically updatable.

If the test fails after a legitimate refactor, update ``REQUIRED_KEYS``
below to match — but think twice before doing so. Each entry was added
because losing the key would resurrect a class of bug we already paid
for once.
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
    "widget_fig_custom_highlights",
    "widget_fig_font_size",
    "widget_fig_alpha",
    "widget_fig_heatmap_norm",
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
