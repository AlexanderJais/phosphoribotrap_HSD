"""Tests for phosphotrap.figures.

Chunk 1 — scaffolding: constants, theme, gene symbol resolution.
Plot-function tests land in subsequent chunks as those functions are
added, to keep each commit individually reviewable.
"""

from __future__ import annotations

from pathlib import Path

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
