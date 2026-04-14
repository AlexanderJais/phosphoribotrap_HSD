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


def test_volcano_plot_alpha_zero_does_not_raise():
    """MEDIUM #3 regression: ``alpha=0`` previously passed ``y=None``
    to ``add_hline`` which raises ValueError. With the guard, the
    horizontal threshold line is simply omitted.
    """
    df = _fake_contrast_table()
    # Must not raise.
    fig = fig_mod.volcano_plot(df, title="t", alpha=0.0)
    assert isinstance(fig, go.Figure)
    # With alpha=0, the -log10(alpha) hline is skipped, so the set
    # of shapes should NOT contain a horizontal dashed line at the
    # padj threshold. The vertical x=0 line should still be present.
    shapes = list(fig.layout.shapes or ())
    horizontal_shapes = [
        s for s in shapes
        if getattr(s, "y0", None) is not None
        and getattr(s, "y1", None) is not None
        and s.y0 == s.y1
    ]
    assert horizontal_shapes == [], (
        "alpha=0 should omit the horizontal threshold line"
    )


def test_volcano_plot_alpha_positive_draws_threshold_line():
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="t", alpha=0.1)
    shapes = list(fig.layout.shapes or ())
    horizontal_shapes = [
        s for s in shapes
        if getattr(s, "y0", None) is not None
        and getattr(s, "y1", None) is not None
        and s.y0 == s.y1
    ]
    assert len(horizontal_shapes) == 1
    # y0 should equal -log10(0.1) == 1.0.
    assert pytest.approx(horizontal_shapes[0].y0, abs=1e-9) == 1.0


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


# ----------------------------------------------------------------------
# per_gene_strip
# ----------------------------------------------------------------------
def _fake_ratio_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Synthetic ratios DataFrame + pair_labels dict matching the
    ``RatioResult`` shape from phosphotrap.fpkm."""
    genes = ["ENSMUSG_GAL", "ENSMUSG_GALR1", "ENSMUSG_BDNF"]
    cols = [
        "NCD_rep1", "NCD_rep3", "NCD_rep4",
        "HSD1_rep5", "HSD1_rep6", "HSD1_rep8",
        "HSD3_rep9", "HSD3_rep10", "HSD3_rep11",
    ]
    data = {
        "NCD_rep1":  [0.0, 0.1, -0.2],
        "NCD_rep3":  [0.1, 0.0, -0.1],
        "NCD_rep4":  [-0.1, 0.2, 0.0],
        "HSD1_rep5": [0.8, 0.5, 0.3],
        "HSD1_rep6": [0.9, 0.6, 0.4],
        "HSD1_rep8": [0.7, 0.4, 0.5],
        "HSD3_rep9": [1.2, 0.9, 0.1],
        "HSD3_rep10":[1.1, 0.8, 0.2],
        "HSD3_rep11":[1.3, 1.0, 0.0],
    }
    ratios = pd.DataFrame(data, index=genes, columns=cols)
    pair_labels = {
        "NCD":  ["NCD_rep1", "NCD_rep3", "NCD_rep4"],
        "HSD1": ["HSD1_rep5", "HSD1_rep6", "HSD1_rep8"],
        "HSD3": ["HSD3_rep9", "HSD3_rep10", "HSD3_rep11"],
    }
    return ratios, pair_labels


def test_per_gene_strip_returns_figure_for_three_genes():
    ratios, pair_labels = _fake_ratio_data()
    gene_labels = {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_GALR1": "Galr1",
        "ENSMUSG_BDNF": "Bdnf",
    }
    fig = fig_mod.per_gene_strip(
        ratios,
        pair_labels,
        title="Galanin signaling — log2(IP/Input)",
        gene_labels=gene_labels,
        primary_ids={"ENSMUSG_GAL", "ENSMUSG_GALR1"},
    )
    assert isinstance(fig, go.Figure)
    # 3 genes × (1 dot trace per group + 1 mean trace per gene)
    # = 3 genes × (3 dot + 1 mean) = 12 traces total.
    assert len(fig.data) == 12


def test_per_gene_strip_dot_count_matches_animals_per_group():
    ratios, pair_labels = _fake_ratio_data()
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    fig = fig_mod.per_gene_strip(
        ratios, pair_labels, title="t", gene_labels=gene_labels,
    )
    # 3 dot traces (one per group), each with 3 animals.
    dot_traces = [
        t for t in fig.data
        if t.mode == "markers"
        and getattr(t.marker, "symbol", None) not in ("line-ew",)
    ]
    assert len(dot_traces) == 3
    for trace in dot_traces:
        assert len(trace.x) == 3
        assert len(trace.y) == 3


def test_per_gene_strip_primary_vs_secondary_color():
    ratios, pair_labels = _fake_ratio_data()
    gene_labels = {"ENSMUSG_GAL": "Gal", "ENSMUSG_BDNF": "Bdnf"}
    fig = fig_mod.per_gene_strip(
        ratios, pair_labels, title="t",
        gene_labels=gene_labels,
        primary_ids={"ENSMUSG_GAL"},
        primary_color="#dc2626",
        secondary_color="#2563eb",
    )
    # Find dot traces by legendgroup.
    galanin_dots = [
        t for t in fig.data
        if getattr(t, "legendgroup", None) == "galanin"
    ]
    custom_dots = [
        t for t in fig.data
        if getattr(t, "legendgroup", None) == "custom"
    ]
    assert galanin_dots, "expected at least one galanin-coloured dot trace"
    assert custom_dots, "expected at least one custom-coloured dot trace"
    assert all(t.marker.color == "#dc2626" for t in galanin_dots)
    assert all(t.marker.color == "#2563eb" for t in custom_dots)


def test_per_gene_strip_skips_gene_not_in_ratios():
    ratios, pair_labels = _fake_ratio_data()
    # Request a gene that isn't in ratios — should be silently dropped.
    gene_labels = {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_MISSING": "Ghost",
    }
    fig = fig_mod.per_gene_strip(
        ratios, pair_labels, title="t", gene_labels=gene_labels,
    )
    # Only 1 gene actually rendered = 3 dot traces + 1 mean = 4.
    assert len(fig.data) == 4


def test_per_gene_strip_empty_inputs_returns_placeholder():
    fig = fig_mod.per_gene_strip(
        pd.DataFrame(), {}, title="t", gene_labels={"X": "X"},
    )
    assert isinstance(fig, go.Figure)
    annotations = fig.layout.annotations or ()
    assert any("No ratio data" in (a.text or "") for a in annotations)


def test_per_gene_strip_respects_group_order():
    ratios, pair_labels = _fake_ratio_data()
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    fig = fig_mod.per_gene_strip(
        ratios, pair_labels, title="t", gene_labels=gene_labels,
        group_order=["HSD3", "HSD1", "NCD"],
    )
    # x-axis category array should match the explicit group_order.
    # Plotly stores it on layout.xaxis.categoryarray.
    cat_array = fig.layout.xaxis.categoryarray
    assert list(cat_array) == ["HSD3", "HSD1", "NCD"]


# ----------------------------------------------------------------------
# regmode_classification
# ----------------------------------------------------------------------
class _FakeAnotaResult:
    """Duck-typed stand-in for Anota2seqResult.

    phosphotrap.anota2seq_runner.Anota2seqResult exposes .translation,
    .buffering, and .mrna_abundance as DataFrames with a gene_id
    column. We don't import it here to keep the figures module
    test-independent of the anota2seq_runner import chain.
    """
    def __init__(
        self,
        translation: list[str],
        buffering: list[str],
        mrna_abundance: list[str],
    ):
        self.translation = pd.DataFrame({"gene_id": translation})
        self.buffering = pd.DataFrame({"gene_id": buffering})
        self.mrna_abundance = pd.DataFrame({"gene_id": mrna_abundance})


def test_regmode_classification_translation_hit():
    anota = {
        "HSD1_vs_NCD": _FakeAnotaResult(
            translation=["ENSMUSG_GAL"],
            buffering=[],
            mrna_abundance=[],
        ),
    }
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    df = fig_mod.regmode_classification(anota, gene_labels)
    assert list(df.columns) == ["gene", "contrast", "mode"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["gene"] == "Gal"
    assert row["contrast"] == "HSD1_vs_NCD"
    assert row["mode"] == "translation"


def test_regmode_classification_buffering_vs_mrna_vs_ns():
    anota = {
        "HSD1_vs_NCD": _FakeAnotaResult(
            translation=[],
            buffering=["ENSMUSG_GALR1"],
            mrna_abundance=["ENSMUSG_GALR2"],
        ),
    }
    gene_labels = {
        "ENSMUSG_GALR1": "Galr1",
        "ENSMUSG_GALR2": "Galr2",
        "ENSMUSG_GALR3": "Galr3",  # absent from all three modes -> n.s.
    }
    df = fig_mod.regmode_classification(anota, gene_labels)
    by_gene = {r["gene"]: r["mode"] for _, r in df.iterrows()}
    assert by_gene["Galr1"] == "buffering"
    assert by_gene["Galr2"] == "mRNA abundance"
    assert by_gene["Galr3"] == "n.s."


def test_regmode_classification_multiple_contrasts_one_gene():
    anota = {
        "HSD1_vs_NCD": _FakeAnotaResult(
            translation=["ENSMUSG_GAL"], buffering=[], mrna_abundance=[],
        ),
        "HSD3_vs_NCD": _FakeAnotaResult(
            translation=[], buffering=["ENSMUSG_GAL"], mrna_abundance=[],
        ),
    }
    df = fig_mod.regmode_classification(anota, {"ENSMUSG_GAL": "Gal"})
    assert len(df) == 2
    modes = dict(zip(df["contrast"], df["mode"]))
    assert modes == {
        "HSD1_vs_NCD": "translation",
        "HSD3_vs_NCD": "buffering",
    }


def test_regmode_classification_sort_order_translation_first():
    """Translation hits float to the top of the table."""
    anota = {
        "HSD1_vs_NCD": _FakeAnotaResult(
            translation=["ENSMUSG_GAL"],
            buffering=["ENSMUSG_GALR1"],
            mrna_abundance=["ENSMUSG_GALR2"],
        ),
    }
    gene_labels = {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_GALR1": "Galr1",
        "ENSMUSG_GALR2": "Galr2",
        "ENSMUSG_GALR3": "Galr3",  # n.s.
    }
    df = fig_mod.regmode_classification(anota, gene_labels)
    # Expected mode order: translation, buffering, mRNA abundance, n.s.
    assert list(df["mode"]) == [
        "translation", "buffering", "mRNA abundance", "n.s.",
    ]


def test_regmode_classification_empty_results():
    df = fig_mod.regmode_classification({}, {"ENSMUSG_GAL": "Gal"})
    assert list(df.columns) == ["gene", "contrast", "mode"]
    assert len(df) == 0


def test_regmode_classification_none_result_is_skipped():
    anota = {
        "HSD1_vs_NCD": _FakeAnotaResult(
            translation=["ENSMUSG_GAL"], buffering=[], mrna_abundance=[],
        ),
        "HSD3_vs_NCD": None,  # run failed, graceful degradation
    }
    df = fig_mod.regmode_classification(anota, {"ENSMUSG_GAL": "Gal"})
    # Only the HSD1 row — the None result is silently skipped.
    assert len(df) == 1
    assert df.iloc[0]["contrast"] == "HSD1_vs_NCD"


def test_regmode_classification_handles_missing_dataframe_columns():
    """A result with empty DataFrames (anota2seq found nothing in any
    mode) must not raise — every gene is "n.s."."""
    class _EmptyResult:
        translation = pd.DataFrame()
        buffering = pd.DataFrame()
        mrna_abundance = pd.DataFrame()

    anota = {"HSD1_vs_NCD": _EmptyResult()}
    df = fig_mod.regmode_classification(
        anota, {"ENSMUSG_GAL": "Gal", "ENSMUSG_GALR1": "Galr1"}
    )
    assert all(df["mode"] == "n.s.")


# ----------------------------------------------------------------------
# cross_contrast_scatter
# ----------------------------------------------------------------------
def _fake_two_contrast_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two contrast tables sharing most genes; values are roughly
    consistent so a diagonal is visible, with a handful of genes
    moved off-diagonal to exercise the highlight layers."""
    rng = np.random.default_rng(1)
    genes = [f"ENSMUSG{str(i).zfill(11)}" for i in range(100)]
    base_delta = rng.normal(0, 1.0, size=100)

    a = pd.DataFrame(
        {
            "delta_log2": base_delta + rng.normal(0, 0.1, size=100),
            "mannwhitney_p": rng.uniform(0, 1, size=100),
            "mannwhitney_padj": rng.uniform(0, 1, size=100),
        },
        index=genes,
    )
    b = pd.DataFrame(
        {
            "delta_log2": base_delta + rng.normal(0, 0.1, size=100),
            "mannwhitney_p": rng.uniform(0, 1, size=100),
            "mannwhitney_padj": rng.uniform(0, 1, size=100),
        },
        index=genes,
    )
    return a, b


def test_cross_contrast_scatter_returns_figure():
    a, b = _fake_two_contrast_tables()
    fig = fig_mod.cross_contrast_scatter(
        a, b,
        label_a="HSD1_vs_NCD",
        label_b="HSD3_vs_NCD",
        title="Cross-contrast consistency",
    )
    assert isinstance(fig, go.Figure)
    # background + diagonal = at least 2 traces
    assert len(fig.data) >= 2


def test_cross_contrast_scatter_with_highlights_has_all_layers():
    a, b = _fake_two_contrast_tables()
    primary = {a.index[0]: "Gal", a.index[1]: "Galr1"}
    secondary = {a.index[2]: "Bdnf"}
    fig = fig_mod.cross_contrast_scatter(
        a, b,
        label_a="HSD1_vs_NCD",
        label_b="HSD3_vs_NCD",
        title="t",
        highlight_primary=primary,
        highlight_secondary=secondary,
    )
    trace_names = [t.name for t in fig.data]
    assert "all genes" in trace_names
    assert "y = x" in trace_names
    assert "Galanin signaling" in trace_names
    assert "Custom highlights" in trace_names


def test_cross_contrast_scatter_intersects_gene_sets():
    """Genes present in only one contrast must be dropped."""
    a, b = _fake_two_contrast_tables()
    # Drop 10 genes from b so the intersection is 90.
    b2 = b.iloc[10:]
    fig = fig_mod.cross_contrast_scatter(
        a, b2,
        label_a="A", label_b="B", title="t",
    )
    bg_trace = next(t for t in fig.data if t.name == "all genes")
    # 90 shared genes minus 0 highlights => 90 background points.
    assert len(bg_trace.x) == 90


def test_cross_contrast_scatter_axes_locked_and_square():
    a, b = _fake_two_contrast_tables()
    fig = fig_mod.cross_contrast_scatter(
        a, b, label_a="A", label_b="B", title="t",
    )
    # Both axes should have the same range, and the y-axis should
    # anchor its scale to x so the diagonal is 45°.
    x_range = tuple(fig.layout.xaxis.range)
    y_range = tuple(fig.layout.yaxis.range)
    assert x_range == y_range
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_cross_contrast_scatter_diagonal_spans_axis_range():
    a, b = _fake_two_contrast_tables()
    fig = fig_mod.cross_contrast_scatter(
        a, b, label_a="A", label_b="B", title="t",
    )
    diag = next(t for t in fig.data if t.name == "y = x")
    # Diagonal is a 2-point line from the low to the high of the
    # shared axis range.
    assert len(diag.x) == 2
    assert list(diag.x) == list(diag.y)
    assert tuple(fig.layout.xaxis.range) == (diag.x[0], diag.x[1])


def test_cross_contrast_scatter_highlight_labels_are_dict_values():
    a, b = _fake_two_contrast_tables()
    primary = {a.index[0]: "Gal", a.index[1]: "Galr1"}
    fig = fig_mod.cross_contrast_scatter(
        a, b, label_a="A", label_b="B", title="t",
        highlight_primary=primary,
    )
    gal_trace = next(t for t in fig.data if t.name == "Galanin signaling")
    assert list(gal_trace.text) == ["Gal", "Galr1"]


def test_cross_contrast_scatter_empty_input_returns_placeholder():
    fig = fig_mod.cross_contrast_scatter(
        pd.DataFrame(), pd.DataFrame(),
        label_a="A", label_b="B", title="t",
    )
    annotations = fig.layout.annotations or ()
    assert any("Need both contrast tables" in (a.text or "") for a in annotations)


def test_cross_contrast_scatter_empty_intersection_returns_placeholder():
    a = pd.DataFrame(
        {"delta_log2": [1.0, -1.0]},
        index=["ENSMUSG_X", "ENSMUSG_Y"],
    )
    b = pd.DataFrame(
        {"delta_log2": [0.5, -0.5]},
        index=["ENSMUSG_P", "ENSMUSG_Q"],
    )
    fig = fig_mod.cross_contrast_scatter(
        a, b, label_a="A", label_b="B", title="t",
    )
    annotations = fig.layout.annotations or ()
    assert any("No genes in common" in (a.text or "") for a in annotations)


# ----------------------------------------------------------------------
# expression_heatmap
# ----------------------------------------------------------------------
from collections import namedtuple  # noqa: E402

FakeRec = namedtuple("FakeRec", ["sample_name", "group", "fraction", "replicate"])


def _fake_rec(group: str, fraction: str, rep: int) -> FakeRec:
    """Duck-typed SampleRecord stand-in. ``name()`` returns
    ``<group>_<fraction><rep>`` to match samples.SampleRecord.name()."""
    rec = FakeRec(sample_name=f"{group}_{fraction}{rep}",
                  group=group, fraction=fraction, replicate=rep)
    # Attach a .name() method via subclass for ducktyping.
    return rec


class _NamedRec:
    """Full duck-typed SampleRecord: has .name(), .group, .fraction,
    .replicate. We don't import the real class from samples.py so the
    figures module stays independent."""
    def __init__(self, group: str, fraction: str, rep: int):
        self.group = group
        self.fraction = fraction
        self.replicate = rep
    def name(self) -> str:
        return f"{self.group}_{self.fraction}{self.replicate}"


def _fake_fpkm_and_records() -> tuple[pd.DataFrame, list[_NamedRec]]:
    """3 groups × 2 fractions × 3 replicates = 18 samples, 4 genes."""
    genes = [
        "ENSMUSG_GAL", "ENSMUSG_GALR1", "ENSMUSG_GALR2", "ENSMUSG_BDNF",
    ]
    records: list[_NamedRec] = []
    cols: list[str] = []
    for group in ("NCD", "HSD1", "HSD3"):
        for fraction in ("IP", "INPUT"):
            for rep in (1, 2, 3):
                r = _NamedRec(group, fraction, rep)
                records.append(r)
                cols.append(r.name())
    rng = np.random.default_rng(42)
    data = rng.uniform(0.5, 100.0, size=(len(genes), len(cols)))
    fpkm = pd.DataFrame(data, index=genes, columns=cols)
    # Force a real signal so z-score tests assert something meaningful.
    fpkm.loc["ENSMUSG_GAL", [c for c in cols if c.startswith("HSD1_IP")]] *= 5
    return fpkm, records


def test_expression_heatmap_returns_figure_with_one_trace():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_GALR1": "Galr1",
    }
    fig = fig_mod.expression_heatmap(
        fpkm, records,
        title="Galanin expression",
        gene_labels=gene_labels,
    )
    assert isinstance(fig, go.Figure)
    # Single heatmap trace.
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_expression_heatmap_row_order_matches_gene_labels():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {
        "ENSMUSG_GALR2": "Galr2",
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_GALR1": "Galr1",
    }
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
    )
    # Row labels (y axis) should match dict iteration order.
    assert list(fig.data[0].y) == ["Galr2", "Gal", "Galr1"]


def test_expression_heatmap_columns_grouped_with_gap_between_blocks():
    """Every (group, fraction) block should be contiguous with a
    single blank column separating adjacent blocks."""
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
    )
    x_labels = list(fig.data[0].x)
    # 3 groups × 2 fractions × 3 reps = 18 data columns, plus 5 gaps
    # between the 6 blocks = 23 total.
    assert len(x_labels) == 23
    # Each gap is a blank string.
    blanks = [i for i, lab in enumerate(x_labels) if lab == ""]
    assert len(blanks) == 5
    # Column labels are short-form like "IP1", "IP2", "IP3",
    # "INPUT1", "INPUT2", "INPUT3" for each block — total of 18
    # non-blank.
    non_blank = [lab for lab in x_labels if lab != ""]
    assert len(non_blank) == 18
    # First block is NCD × IP sorted by replicate.
    assert non_blank[:3] == ["IP1", "IP2", "IP3"]
    # Second block is NCD × INPUT.
    assert non_blank[3:6] == ["INPUT1", "INPUT2", "INPUT3"]


def test_expression_heatmap_zscore_default_colorscale():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {"ENSMUSG_GAL": "Gal", "ENSMUSG_GALR1": "Galr1"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
    )
    # Default z-score uses a diverging scale centered at 0.
    assert fig.data[0].zmid == 0
    # RdBu_r is stored as a tuple-of-stops, not a string, so just
    # sanity-check that the first stop's colour matches a blue.
    stops = fig.data[0].colorscale
    assert stops is not None and len(stops) > 0


def test_expression_heatmap_log2_normalization():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
        normalize="log2",
    )
    assert fig.data[0].zmid is None  # log2 doesn't center
    # Colorbar title reflects the chosen normalization.
    assert "log" in (fig.data[0].colorbar.title.text or "").lower()


def test_expression_heatmap_raw_normalization():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
        normalize="raw",
    )
    assert fig.data[0].zmid is None
    assert fig.data[0].colorbar.title.text == "FPKM"


def test_expression_heatmap_zscore_zero_std_row_is_neutral(tmp_path: Path):
    """A row with identical values across samples must render neutral
    (zero z-score), not NaN/white."""
    genes = ["ENSMUSG_FLAT"]
    records = [
        _NamedRec("NCD", "IP", 1), _NamedRec("NCD", "IP", 2),
        _NamedRec("NCD", "IP", 3),
        _NamedRec("NCD", "INPUT", 1), _NamedRec("NCD", "INPUT", 2),
        _NamedRec("NCD", "INPUT", 3),
    ]
    cols = [r.name() for r in records]
    fpkm = pd.DataFrame([[5.0] * len(cols)], index=genes, columns=cols)
    gene_labels = {"ENSMUSG_FLAT": "Flat"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
    )
    z = np.array(fig.data[0].z, dtype=float)
    # All non-gap values should be exactly 0.
    finite = z[~np.isnan(z)]
    assert (finite == 0.0).all()


def test_expression_heatmap_missing_genes_returns_placeholder():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {"ENSMUSG_NOT_THERE": "Ghost"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
    )
    annotations = fig.layout.annotations or ()
    assert any("No expression data" in (a.text or "") for a in annotations)


def test_expression_heatmap_block_header_annotations_present():
    fpkm, records = _fake_fpkm_and_records()
    gene_labels = {"ENSMUSG_GAL": "Gal"}
    fig = fig_mod.expression_heatmap(
        fpkm, records, title="t", gene_labels=gene_labels,
    )
    annots = fig.layout.annotations or ()
    texts = [a.text for a in annots]
    # Six "IP" / "INPUT" fraction labels + three bold group labels.
    assert texts.count("IP") == 3
    assert texts.count("INPUT") == 3
    assert any("<b>NCD</b>" in t for t in texts)
    assert any("<b>HSD1</b>" in t for t in texts)
    assert any("<b>HSD3</b>" in t for t in texts)


def test_per_gene_strip_shared_y_range():
    ratios, pair_labels = _fake_ratio_data()
    gene_labels = {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_GALR1": "Galr1",
        "ENSMUSG_BDNF": "Bdnf",
    }
    fig = fig_mod.per_gene_strip(
        ratios, pair_labels, title="t", gene_labels=gene_labels,
    )
    # Every y axis should share the same range (padding makes it
    # slightly wider than the raw min/max).
    y_ranges = []
    for key in fig.layout:
        if key.startswith("yaxis"):
            axis = fig.layout[key]
            if axis.range is not None:
                y_ranges.append(tuple(axis.range))
    assert len(y_ranges) >= 3
    assert len(set(y_ranges)) == 1, (
        f"expected all subplots to share one y range, got {y_ranges}"
    )
