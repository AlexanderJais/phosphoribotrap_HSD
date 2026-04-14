"""Tests for phosphotrap.figures.

Chunk 1 — scaffolding: constants, theme, gene symbol resolution.
Plot-function tests land in subsequent chunks as those functions are
added, to keep each commit individually reviewable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from phosphotrap import figures as fig_mod


# ----------------------------------------------------------------------
# Galanin set + theme
# ----------------------------------------------------------------------
def test_galanin_genes_contains_canonical_five():
    assert fig_mod.GALANIN_GENES == ("Gal", "Galp", "Galr1", "Galr2", "Galr3")


def test_nature_theme_has_expected_keys():
    theme = fig_mod.nature_theme()
    # Core plotly layout keys we rely on everywhere.
    assert theme["template"] == "plotly_white"
    assert theme["paper_bgcolor"] == "white"
    assert theme["plot_bgcolor"] == "white"
    assert theme["font"]["family"] == fig_mod.NATURE_FONT_FAMILY
    assert theme["font"]["size"] == fig_mod.DEFAULT_FONT_SIZE


def test_nature_theme_honours_custom_font_size():
    theme = fig_mod.nature_theme(base_font_size=20)
    assert theme["font"]["size"] == 20
    # Title font tracks base + 2 for a visible hierarchy.
    assert theme["title"]["font"]["size"] == 22


# ----------------------------------------------------------------------
# load_gene_symbol_map
# ----------------------------------------------------------------------
def _write_tx2gene(path: Path, rows: list[tuple[str, ...]]) -> Path:
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return path


def test_load_gene_symbol_map_three_column(tmp_path: Path):
    tx2 = _write_tx2gene(
        tmp_path / "tx2gene.tsv",
        [
            ("ENSMUST_GAL_1", "ENSMUSG_GAL",   "Gal"),
            ("ENSMUST_GAL_2", "ENSMUSG_GAL",   "Gal"),     # dup symbol -> same gene_id
            ("ENSMUST_GLR1",  "ENSMUSG_GALR1", "Galr1"),
            ("ENSMUST_GLR2",  "ENSMUSG_GALR2", "Galr2"),
        ],
    )
    m = fig_mod.load_gene_symbol_map(tx2)
    # Lookup is lowercase.
    assert m == {
        "gal":   "ENSMUSG_GAL",
        "galr1": "ENSMUSG_GALR1",
        "galr2": "ENSMUSG_GALR2",
    }


def test_load_gene_symbol_map_empty_on_two_column(tmp_path: Path):
    """2-column tx2gene has no symbols -> empty map, caller falls back."""
    tx2 = _write_tx2gene(
        tmp_path / "tx2g_2col.tsv",
        [
            ("ENSMUST_A", "ENSMUSG_A"),
            ("ENSMUST_B", "ENSMUSG_B"),
        ],
    )
    assert fig_mod.load_gene_symbol_map(tx2) == {}


def test_load_gene_symbol_map_skips_malformed_rows(tmp_path: Path):
    tx2 = tmp_path / "tx2gene.tsv"
    tx2.write_text(
        "\n".join(
            [
                "ENSMUST_GAL\tENSMUSG_GAL\tGal",
                "",                                # blank line
                "ENSMUST_EMPTY_GENE\t\tOrphan",    # empty gene_id
                "ENSMUST_EMPTY_SYM\tENSMUSG_X\t",  # empty symbol
                "ENSMUST_BDNF\tENSMUSG_BDNF\tBdnf",
            ]
        )
        + "\n"
    )
    m = fig_mod.load_gene_symbol_map(tx2)
    assert m == {"gal": "ENSMUSG_GAL", "bdnf": "ENSMUSG_BDNF"}


def test_load_gene_symbol_map_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        fig_mod.load_gene_symbol_map(tmp_path / "nope.tsv")


# ----------------------------------------------------------------------
# resolve_symbols
# ----------------------------------------------------------------------
def test_resolve_symbols_happy_path():
    symbol_map = {
        "gal":   "ENSMUSG_GAL",
        "galp":  "ENSMUSG_GALP",
        "galr1": "ENSMUSG_GALR1",
    }
    resolved, missing = fig_mod.resolve_symbols(
        fig_mod.GALANIN_GENES, symbol_map
    )
    assert resolved == {
        "ENSMUSG_GAL":   "Gal",
        "ENSMUSG_GALP":  "Galp",
        "ENSMUSG_GALR1": "Galr1",
    }
    # Galr2 and Galr3 are not in the map.
    assert missing == ["Galr2", "Galr3"]


def test_resolve_symbols_case_insensitive_with_original_casing_preserved():
    symbol_map = {"bdnf": "ENSMUSG_BDNF"}
    resolved, missing = fig_mod.resolve_symbols(["BDNF"], symbol_map)
    # Lookup is case-insensitive but the DISPLAY label keeps the
    # user's original capitalisation so plot labels look natural.
    assert resolved == {"ENSMUSG_BDNF": "BDNF"}
    assert missing == []


def test_resolve_symbols_deduplicates():
    symbol_map = {"gal": "ENSMUSG_GAL"}
    resolved, missing = fig_mod.resolve_symbols(
        ["Gal", "GAL", "gal"], symbol_map
    )
    assert resolved == {"ENSMUSG_GAL": "Gal"}  # first casing wins
    assert missing == []


def test_resolve_symbols_skips_blank_and_none_inputs():
    symbol_map = {"gal": "ENSMUSG_GAL"}
    resolved, missing = fig_mod.resolve_symbols(
        ["Gal", "", "   ", None], symbol_map  # type: ignore[list-item]
    )
    assert resolved == {"ENSMUSG_GAL": "Gal"}
    assert missing == []


def test_resolve_symbols_collects_missing_in_order():
    symbol_map = {"gal": "ENSMUSG_GAL"}
    resolved, missing = fig_mod.resolve_symbols(
        ["Bdnf", "Gal", "Npy", "Galr3"], symbol_map
    )
    assert resolved == {"ENSMUSG_GAL": "Gal"}
    assert missing == ["Bdnf", "Npy", "Galr3"]


# ----------------------------------------------------------------------
# parse_highlight_text
# ----------------------------------------------------------------------
def test_parse_highlight_text_empty():
    assert fig_mod.parse_highlight_text("") == []
    assert fig_mod.parse_highlight_text("   \n\n  ") == []


def test_parse_highlight_text_comma_separated():
    assert fig_mod.parse_highlight_text("Gal, Galp, Galr1") == [
        "Gal", "Galp", "Galr1"
    ]


def test_parse_highlight_text_whitespace_separated():
    assert fig_mod.parse_highlight_text("Bdnf  Npy\nPomc") == [
        "Bdnf", "Npy", "Pomc"
    ]


def test_parse_highlight_text_mixed():
    text = """
    Gal, Galp
    Galr1 Galr2
      Bdnf,Npy,  Pomc
    """
    assert fig_mod.parse_highlight_text(text) == [
        "Gal", "Galp", "Galr1", "Galr2", "Bdnf", "Npy", "Pomc"
    ]


# ----------------------------------------------------------------------
# volcano_plot
# ----------------------------------------------------------------------
def _fake_contrast_table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic contrast table indexed by gene_id.

    Roughly volcano-shaped: centered at 0, a handful of genes out in
    the tails with small p-values. Columns match what
    ``phosphotrap.fpkm.ContrastResult.table`` produces.
    """
    rng = np.random.default_rng(seed)
    gene_ids = [f"ENSMUSG{str(i).zfill(11)}" for i in range(n)]
    delta = rng.normal(0, 1.0, size=n)
    p = rng.uniform(0, 1, size=n)
    # Force a few genes into the "significant" zone.
    delta[:5] = [2.5, -2.2, 1.9, -1.8, 3.0]
    p[:5] = [1e-4, 1e-3, 5e-3, 2e-3, 7e-5]
    padj = np.clip(p * 2, 0, 1)
    df = pd.DataFrame(
        {
            "mean_log2_alt": delta + 0.1,
            "mean_log2_ref": 0.1,
            "delta_log2": delta,
            "mannwhitney_p": p,
            "mannwhitney_padj": padj,
        },
        index=gene_ids,
    )
    df.index.name = "gene_id"
    return df


def test_volcano_plot_returns_figure_with_four_trace_layers():
    df = _fake_contrast_table()
    primary = {df.index[0]: "Gal", df.index[1]: "Galp"}
    secondary = {df.index[2]: "Bdnf"}
    fig = fig_mod.volcano_plot(
        df,
        title="HSD1_vs_NCD",
        highlight_primary=primary,
        highlight_secondary=secondary,
        alpha=0.1,
    )
    assert isinstance(fig, go.Figure)
    # Exactly 4 layers: background, significant, primary, secondary.
    assert len(fig.data) == 4
    trace_names = [t.name for t in fig.data]
    assert "all genes" in trace_names
    assert any("padj" in n for n in trace_names)
    assert "Galanin signaling" in trace_names
    assert "Custom highlights" in trace_names


def test_volcano_plot_title_in_layout():
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="HSD3_vs_NCD")
    # Plotly wraps string titles in a dict when we set font via theme.
    assert fig.layout.title.text == "HSD3_vs_NCD"


def test_volcano_plot_highlight_trace_uses_labels_as_text():
    df = _fake_contrast_table()
    primary = {df.index[0]: "Gal", df.index[1]: "Galp"}
    fig = fig_mod.volcano_plot(
        df, title="C", highlight_primary=primary
    )
    galanin_traces = [t for t in fig.data if t.name == "Galanin signaling"]
    assert len(galanin_traces) == 1
    trace = galanin_traces[0]
    assert "markers+text" in trace.mode
    assert list(trace.text) == ["Gal", "Galp"]


def test_volcano_plot_primary_color_override():
    df = _fake_contrast_table()
    primary = {df.index[0]: "Gal"}
    fig = fig_mod.volcano_plot(
        df,
        title="C",
        highlight_primary=primary,
        primary_color="#00aa00",
    )
    galanin_traces = [t for t in fig.data if t.name == "Galanin signaling"]
    assert galanin_traces[0].marker.color == "#00aa00"


def test_volcano_plot_empty_table_returns_placeholder_figure():
    fig = fig_mod.volcano_plot(pd.DataFrame(), title="empty")
    assert isinstance(fig, go.Figure)
    # Should carry the "No contrast data loaded." annotation.
    annotations = fig.layout.annotations or ()
    assert any(
        "No contrast data loaded" in (a.text or "") for a in annotations
    )


def test_volcano_plot_skips_unknown_highlight_ids():
    """A gene_id in the highlight dict that isn't in the table is a no-op."""
    df = _fake_contrast_table()
    primary = {
        df.index[0]: "Gal",
        "ENSMUSG_NOT_IN_TABLE": "Ghost",
    }
    fig = fig_mod.volcano_plot(
        df, title="C", highlight_primary=primary, highlight_secondary=None
    )
    galanin_traces = [t for t in fig.data if t.name == "Galanin signaling"]
    assert len(galanin_traces) == 1
    # Only one valid highlight point drawn.
    assert len(galanin_traces[0].x) == 1
    assert list(galanin_traces[0].text) == ["Gal"]


def test_volcano_plot_font_size_propagates_to_layout():
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="C", font_size=20)
    assert fig.layout.font.size == 20


def test_volcano_plot_handles_missing_padj_column():
    df = _fake_contrast_table().drop(columns=["mannwhitney_padj"])
    fig = fig_mod.volcano_plot(df, title="C", alpha=0.1)
    # With padj missing, nothing qualifies as "significant" except
    # explicitly via the threshold comparison which is all-NaN -> False.
    # We should still get the background trace and NO sig trace points.
    assert isinstance(fig, go.Figure)
    sig_traces = [t for t in fig.data if t.name and "padj" in t.name]
    assert len(sig_traces) == 1
    assert len(sig_traces[0].x) == 0


def test_neg_log10_handles_zero_and_negative():
    s = pd.Series([0.001, 0.0, -1, np.nan, 1.0])
    out = fig_mod._neg_log10(s)
    assert pytest.approx(out[0], abs=1e-9) == 3.0
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert pytest.approx(out[4], abs=1e-9) == 0.0
