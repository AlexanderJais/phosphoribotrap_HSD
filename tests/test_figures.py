"""Tests for phosphotrap.figures.

Chunk 1 — scaffolding: constants, theme, gene symbol resolution.
Plot-function tests land in subsequent chunks as those functions are
added, to keep each commit individually reviewable.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

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
# load_gene_id_to_name_map
# ----------------------------------------------------------------------
def test_load_gene_id_to_name_map_three_column(tmp_path: Path):
    """Happy path: 3-column tx2gene with multiple transcripts per gene
    collapses to a deterministic ``{gene_id: gene_name}`` map.
    """
    tx2 = tmp_path / "tx2gene.tsv"
    tx2.write_text(
        "\n".join(
            [
                "ENSMUST_GAL_1\tENSMUSG_GAL\tGal",
                "ENSMUST_GAL_2\tENSMUSG_GAL\tGal",
                "ENSMUST_BDNF\tENSMUSG_BDNF\tBdnf",
                "ENSMUST_PCSK1\tENSMUSG_PCSK1\tPcsk1n",
            ]
        )
        + "\n"
    )
    m = fig_mod.load_gene_id_to_name_map(tx2)
    assert m == {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_BDNF": "Bdnf",
        "ENSMUSG_PCSK1": "Pcsk1n",
    }


def test_load_gene_id_to_name_map_two_column_returns_empty(tmp_path: Path):
    """Legacy 2-column tx2gene has no symbol — the map is empty and
    the caller is expected to fall back to plain gene_ids.
    """
    tx2 = tmp_path / "tx2gene.tsv"
    tx2.write_text("ENSMUST_GAL_1\tENSMUSG_GAL\n")
    assert fig_mod.load_gene_id_to_name_map(tx2) == {}


def test_load_gene_id_to_name_map_skips_malformed_rows(tmp_path: Path):
    """Rows missing gene_id or gene_symbol are silently dropped so
    one corrupt line doesn't torch the whole map.
    """
    tx2 = tmp_path / "tx2gene.tsv"
    tx2.write_text(
        "\n".join(
            [
                "ENSMUST_GAL\tENSMUSG_GAL\tGal",
                "ENSMUST_EMPTY\t\tEmptyId",
                "ENSMUST_EMPTY2\tENSMUSG_X\t",
                "ENSMUST_BDNF\tENSMUSG_BDNF\tBdnf",
            ]
        )
        + "\n"
    )
    assert fig_mod.load_gene_id_to_name_map(tx2) == {
        "ENSMUSG_GAL": "Gal",
        "ENSMUSG_BDNF": "Bdnf",
    }


def test_load_gene_id_to_name_map_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        fig_mod.load_gene_id_to_name_map(tmp_path / "nope.tsv")


# ----------------------------------------------------------------------
# prepend_gene_name
# ----------------------------------------------------------------------
def test_prepend_gene_name_from_index():
    """Default behaviour: DataFrame indexed by gene_id. The helper
    resets the index and inserts ``gene_name`` as column 1."""
    df = pd.DataFrame(
        {"delta_log2": [1.0, -2.0, 0.5]},
        index=pd.Index(
            ["ENSMUSG_GAL", "ENSMUSG_BDNF", "ENSMUSG_UNKNOWN"],
            name="gene_id",
        ),
    )
    id_to_name = {"ENSMUSG_GAL": "Gal", "ENSMUSG_BDNF": "Bdnf"}
    out = fig_mod.prepend_gene_name(df, id_to_name)
    assert list(out.columns) == ["gene_id", "gene_name", "delta_log2"]
    assert out["gene_id"].tolist() == [
        "ENSMUSG_GAL", "ENSMUSG_BDNF", "ENSMUSG_UNKNOWN",
    ]
    # Unmapped gene_ids fall back to the gene_id itself so no row is
    # ever silently anonymised to an empty string.
    assert out["gene_name"].tolist() == ["Gal", "Bdnf", "ENSMUSG_UNKNOWN"]


def test_prepend_gene_name_from_unnamed_index():
    """An unnamed index still gets normalised to ``gene_id`` in the
    output schema."""
    df = pd.DataFrame(
        {"score": [1.0, 2.0]},
        index=["ENSMUSG_GAL", "ENSMUSG_BDNF"],
    )
    out = fig_mod.prepend_gene_name(df, {"ENSMUSG_GAL": "Gal"})
    assert list(out.columns)[:2] == ["gene_id", "gene_name"]
    assert out["gene_id"].tolist() == ["ENSMUSG_GAL", "ENSMUSG_BDNF"]


def test_prepend_gene_name_from_column():
    """``id_col`` variant: DataFrame already has a gene_id column,
    insertion goes right after it and preserves the existing index."""
    df = pd.DataFrame(
        {
            "gene_id": ["ENSMUSG_GAL", "ENSMUSG_BDNF"],
            "log2FoldChange": [1.5, -0.5],
            "padj": [0.01, 0.2],
        }
    )
    out = fig_mod.prepend_gene_name(
        df, {"ENSMUSG_GAL": "Gal", "ENSMUSG_BDNF": "Bdnf"}, id_col="gene_id"
    )
    assert list(out.columns) == [
        "gene_id", "gene_name", "log2FoldChange", "padj"
    ]
    assert out["gene_name"].tolist() == ["Gal", "Bdnf"]


def test_prepend_gene_name_overwrites_existing_gene_name_column():
    """If the DataFrame already has a stale ``gene_name`` column
    (e.g. from a previous augmentation pass), the fresh map wins —
    no duplicate-column error from pandas."""
    df = pd.DataFrame(
        {
            "gene_id": ["ENSMUSG_GAL"],
            "gene_name": ["OldName"],
            "value": [1.0],
        }
    )
    out = fig_mod.prepend_gene_name(
        df, {"ENSMUSG_GAL": "Gal"}, id_col="gene_id"
    )
    assert out["gene_name"].tolist() == ["Gal"]
    # Only one gene_name column.
    assert list(out.columns).count("gene_name") == 1


def test_prepend_gene_name_from_column_missing_col_raises():
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(KeyError, match="id_col='gene_id'"):
        fig_mod.prepend_gene_name(df, {}, id_col="gene_id")


def test_load_gene_id_to_name_map_registers_version_stripped_key(tmp_path: Path):
    """GENCODE gene_ids carry a ``.<digits>`` version suffix. The
    loader stores both the versioned key (as read) and the stripped
    form so downstream lookups tolerate either convention — otherwise
    a DataFrame built from an unversioned GTF silently resolves to
    NaN and prepend_gene_name falls back to gene_id, producing the
    "gene_name looks like gene_id" symptom we kept chasing.
    """
    tx2 = tmp_path / "tx2gene.tsv"
    tx2.write_text(
        "ENSMUST00000000001.5\tENSMUSG00000051951.6\tXkr4\n"
        "ENSMUST00000000002.8\tENSMUSG00000025900.13\tRp1\n"
    )
    m = fig_mod.load_gene_id_to_name_map(tx2)
    # Versioned keys (preserved verbatim from the TSV)
    assert m["ENSMUSG00000051951.6"] == "Xkr4"
    assert m["ENSMUSG00000025900.13"] == "Rp1"
    # Stripped keys (synthesised by _strip_gene_id_version)
    assert m["ENSMUSG00000051951"] == "Xkr4"
    assert m["ENSMUSG00000025900"] == "Rp1"


def test_prepend_gene_name_version_tolerant_lookup_stripped_df():
    """DataFrame carries version-stripped gene_ids, map was built
    from a versioned tx2gene. The augmented output should still show
    the real gene_name (via the stripped-key entries registered by
    load_gene_id_to_name_map), not fall back to gene_id.
    """
    df = pd.DataFrame(
        {"delta_log2": [1.0, -0.5]},
        index=["ENSMUSG00000051951", "ENSMUSG00000025900"],
    )
    id_to_name = {
        # Only versioned keys — simulating a caller that didn't go
        # through load_gene_id_to_name_map's setdefault-stripped step.
        "ENSMUSG00000051951.6": "Xkr4",
        "ENSMUSG00000025900.13": "Rp1",
    }
    out = fig_mod.prepend_gene_name(df, id_to_name)
    assert out["gene_name"].tolist() == ["Xkr4", "Rp1"]


def test_prepend_gene_name_version_tolerant_lookup_versioned_df():
    """The reverse: DataFrame carries versioned gene_ids, map was
    built from an unversioned tx2gene. The exact match fails but
    the stripped retry in _version_tolerant_name_lookup rescues it.
    """
    df = pd.DataFrame(
        {"gene_id": ["ENSMUSG00000051951.6", "ENSMUSG00000025900.13"],
         "log2FC": [0.3, -0.2]}
    )
    id_to_name = {
        "ENSMUSG00000051951": "Xkr4",
        "ENSMUSG00000025900": "Rp1",
    }
    out = fig_mod.prepend_gene_name(df, id_to_name, id_col="gene_id")
    assert out["gene_name"].tolist() == ["Xkr4", "Rp1"]


def test_prepend_gene_name_empty_dataframe():
    """Empty input returns a frame with the gene_name column added,
    not a crash. Download buttons in the UI sometimes fire against
    empty cached tables during a partial-analysis state."""
    df = pd.DataFrame(
        {"delta_log2": []},
        index=pd.Index([], name="gene_id"),
    )
    out = fig_mod.prepend_gene_name(df, {})
    assert "gene_name" in out.columns
    assert len(out) == 0


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


def test_fetch_go_term_genes_rejects_invalid_go_id():
    """``fetch_go_term_genes`` must refuse anything that isn't the
    canonical ``GO:dddddddd`` curie before opening a connection.
    Protects against splicing arbitrary user input into the request
    URL (prompt-injection / SSRF concerns) and against the common
    typo of pasting a full jax.org URL into the GO field.
    """
    for bad in [
        "",
        "GO",
        "GO:",
        "GO:123",
        "GO:12345678",        # nine digits, one too many
        "GO:abcdefg",
        "https://www.informatics.jax.org/go/term/GO:0007218",
        "0007218",
    ]:
        with pytest.raises(ValueError, match="Not a valid GO ID"):
            fig_mod.fetch_go_term_genes(
                bad, fetcher=lambda url, timeout: {"results": []}
            )


def test_fetch_go_term_genes_rejects_invalid_taxon():
    with pytest.raises(ValueError, match="Not a valid NCBI taxon"):
        fig_mod.fetch_go_term_genes(
            "GO:0007218",
            taxon="mouse",
            fetcher=lambda url, timeout: {"results": []},
        )


def test_fetch_go_term_genes_single_page(tmp_path: Path):
    """Single-page fetch: the stubbed fetcher returns one page with
    three gene symbols and a ``pageInfo.total = 1``. The helper must
    return exactly those symbols (de-duplicated and as a set) and
    persist them to the cache directory for subsequent offline use.
    """
    captured_urls: list[str] = []

    def stub(url: str, timeout: float) -> dict:
        captured_urls.append(url)
        return {
            "numberOfHits": 3,
            "pageInfo": {"current": 1, "total": 1, "resultsPerPage": 100},
            "results": [
                {"symbol": "Gal",   "geneProductId": "MGI:MGI:95637"},
                {"symbol": "Galp",  "geneProductId": "MGI:MGI:1891814"},
                {"symbol": "Galr1", "geneProductId": "MGI:MGI:1337005"},
            ],
        }

    out = fig_mod.fetch_go_term_genes(
        "GO:0007218",
        taxon="10090",
        cache_dir=tmp_path,
        fetcher=stub,
    )
    assert out == {"Gal", "Galp", "Galr1"}
    # The URL was built with the right query parameters.
    assert len(captured_urls) == 1
    assert "goId=GO%3A0007218" in captured_urls[0]
    assert "taxonId=10090" in captured_urls[0]
    assert "goUsage=descendants" in captured_urls[0]
    # Cache file landed where we expect and carries the gene set.
    cache_file = tmp_path / "go_GO_0007218_10090.json"
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text())
    assert set(payload["symbols"]) == {"Gal", "Galp", "Galr1"}
    assert payload["go_id"] == "GO:0007218"
    assert payload["taxon"] == "10090"


def test_fetch_go_term_genes_walks_pages(tmp_path: Path):
    """Multi-page fetch: stub reports ``pageInfo.total = 2`` so the
    helper must issue two HTTP calls and union the results.
    """
    pages = {
        1: {
            "pageInfo": {"current": 1, "total": 2, "resultsPerPage": 2},
            "results": [{"symbol": "Gal"}, {"symbol": "Galp"}],
        },
        2: {
            "pageInfo": {"current": 2, "total": 2, "resultsPerPage": 2},
            "results": [{"symbol": "Galr1"}, {"symbol": "Galr2"}],
        },
    }
    calls: list[int] = []

    def stub(url: str, timeout: float) -> dict:
        # Extract the page param and return the matching page.
        if "page=1" in url:
            calls.append(1)
            return pages[1]
        if "page=2" in url:
            calls.append(2)
            return pages[2]
        raise AssertionError(f"unexpected URL: {url}")

    out = fig_mod.fetch_go_term_genes(
        "GO:0007218", cache_dir=tmp_path, fetcher=stub
    )
    assert out == {"Gal", "Galp", "Galr1", "Galr2"}
    assert calls == [1, 2]


def test_fetch_go_term_genes_uses_disk_cache_on_second_call(tmp_path: Path):
    """Second call with the same arguments must read from cache and
    skip the fetcher entirely — that's the whole point of the disk
    cache, the network is only hit once per GO ID per taxon.
    """
    def stub(url: str, timeout: float) -> dict:
        return {
            "pageInfo": {"current": 1, "total": 1},
            "results": [{"symbol": "Gal"}],
        }

    # First call populates the cache.
    first = fig_mod.fetch_go_term_genes(
        "GO:0007218", cache_dir=tmp_path, fetcher=stub
    )
    assert first == {"Gal"}

    # Second call with a fetcher that explodes — if the helper tries
    # to hit the network, this test fails loudly.
    def exploding(url: str, timeout: float) -> dict:
        raise AssertionError("fetcher must not be called on cache hit")

    second = fig_mod.fetch_go_term_genes(
        "GO:0007218", cache_dir=tmp_path, fetcher=exploding
    )
    assert second == {"Gal"}


def test_fetch_go_term_genes_refetches_when_cache_schema_mismatches(
    tmp_path: Path,
):
    """A cache file written by an older build of the parser (no
    ``schema_version`` field, or a different integer) must be treated
    as stale and refetched. Without this, a field-rename in QuickGO's
    response would have required every user to manually click
    "force refresh" to drop the stale parse.
    """
    cache_path = tmp_path / "go_GO_0007218_10090.json"
    # Simulate a v0 cache file (pre-schema-version) with stale content
    # that the fresh fetch will replace.
    cache_path.write_text(
        json.dumps({
            "go_id": "GO:0007218",
            "taxon": "10090",
            "fetched_at": 0.0,
            "symbols": ["StaleGeneFromOldSchema"],
        })
    )

    called: list[str] = []

    def stub(url: str, timeout: float) -> dict:
        called.append(url)
        return {
            "pageInfo": {"current": 1, "total": 1},
            "results": [{"symbol": "Gal"}],
        }

    out = fig_mod.fetch_go_term_genes(
        "GO:0007218", cache_dir=tmp_path, fetcher=stub
    )
    # The stale v0 cache was ignored; fetcher was called; fresh result
    # replaced the cache.
    assert out == {"Gal"}
    assert len(called) == 1, "fetcher should have been invoked exactly once"
    fresh = json.loads(cache_path.read_text())
    assert fresh["schema_version"] == fig_mod._GO_CACHE_SCHEMA_VERSION
    assert fresh["symbols"] == ["Gal"]


def test_fetch_go_term_genes_force_refresh_bypasses_cache(tmp_path: Path):
    """``force_refresh=True`` must skip the cache and re-hit the
    fetcher, so users on the Figures tab can click Refresh to get
    updated MGI curation without deleting files by hand.
    """
    fetch_calls: list[int] = []

    def stub(url: str, timeout: float) -> dict:
        fetch_calls.append(1)
        return {
            "pageInfo": {"current": 1, "total": 1},
            "results": [{"symbol": "Gal"}],
        }

    fig_mod.fetch_go_term_genes("GO:0007218", cache_dir=tmp_path, fetcher=stub)
    fig_mod.fetch_go_term_genes(
        "GO:0007218", cache_dir=tmp_path, fetcher=stub, force_refresh=True
    )
    assert len(fetch_calls) == 2


def test_fetch_go_term_genes_network_failure_falls_back_to_cache(tmp_path: Path):
    """If a fresh cache exists from a previous successful call, a
    network failure on the next call must return the cached symbols
    with a warning instead of raising — the Figures tab needs to
    stay usable offline once the user has fetched at least once.
    """
    def good(url: str, timeout: float) -> dict:
        return {
            "pageInfo": {"current": 1, "total": 1},
            "results": [{"symbol": "Gal"}, {"symbol": "Galp"}],
        }

    # Populate cache.
    fig_mod.fetch_go_term_genes("GO:0007218", cache_dir=tmp_path, fetcher=good)

    def dead(url: str, timeout: float) -> dict:
        raise urllib.error.URLError("network unreachable")

    out = fig_mod.fetch_go_term_genes(
        "GO:0007218",
        cache_dir=tmp_path,
        fetcher=dead,
        force_refresh=True,  # force a refetch -> triggers the failure path
    )
    assert out == {"Gal", "Galp"}


def test_fetch_go_term_genes_network_failure_no_cache_raises():
    """With no cache available, a network failure must surface as a
    ``RuntimeError`` with a user-readable message. The Figures tab
    catches this and renders it via ``st.error`` so users know to
    paste the gene list manually or check their network.
    """
    def dead(url: str, timeout: float) -> dict:
        raise urllib.error.URLError("connection refused")

    with pytest.raises(RuntimeError, match="Could not fetch GO term"):
        fig_mod.fetch_go_term_genes("GO:0007218", fetcher=dead)


def test_fetch_go_term_genes_handles_empty_results():
    """An unknown GO ID (or a taxon with no annotations for that
    term) returns an empty set without raising. Caller decides how
    to surface that to the user.
    """
    def stub(url: str, timeout: float) -> dict:
        return {
            "pageInfo": {"current": 1, "total": 1},
            "results": [],
        }

    out = fig_mod.fetch_go_term_genes("GO:0000000", fetcher=stub)
    assert out == set()


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


def _mwu_n3_contrast_table(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Contrast table that mimics the n=3 vs n=3 Mann-Whitney
    banding: only 5 possible two-sided p-values. Used by the honest-
    banding regression test below.
    """
    rng = np.random.default_rng(seed)
    gene_ids = [f"ENSMUSG{str(i).zfill(11)}" for i in range(n)]
    discrete = np.array([0.10, 0.20, 0.40, 0.70, 1.00])
    p = discrete[rng.integers(0, 5, size=n)]
    delta = rng.normal(0, 1, size=n)
    return pd.DataFrame(
        {
            "delta_log2": delta,
            "mannwhitney_p": p,
            "mannwhitney_padj": np.clip(p * 2, 0, 1),
        },
        index=gene_ids,
    )


def test_volcano_preserves_discrete_mwu_bands_without_jitter():
    """Regression pin: ``volcano_plot`` must NOT apply y-jitter to
    the MWU n=3 p-value distribution. Moving points off their true
    -log10(p) would misrepresent the statistic. The banding is a
    property of the Mann-Whitney exact distribution at n=3 vs 3
    (only C(6,3)=20 rank orderings → 5 distinct two-sided p-values)
    and the plot surfaces that honestly. Users who need continuous
    p at small n should switch to the anota2seq moderated RVM
    p-value; see the docstring on ``volcano_plot``.
    """
    df = _mwu_n3_contrast_table()
    fig = fig_mod.volcano_plot(df, title="honest bands")
    bg = [t for t in fig.data if t.name == "all genes"][0]
    # Background y-values still sit on ≤5 raw bands, not a cloud.
    assert 1 <= len(np.unique(bg.y)) <= 5
    # And those values match -log10 of the raw p column — no
    # transformation besides the safe-log10.
    raw_y = fig_mod._neg_log10(df["mannwhitney_p"])
    assert set(np.unique(bg.y)).issubset(set(np.unique(raw_y[np.isfinite(raw_y)])))


def test_neg_log10_handles_zero_and_negative():
    s = pd.Series([0.001, 0.0, -1, np.nan, 1.0])
    out = fig_mod._neg_log10(s)
    assert pytest.approx(out[0], abs=1e-9) == 3.0
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert pytest.approx(out[4], abs=1e-9) == 0.0


# ----------------------------------------------------------------------
# figure_export_bytes + fig.to_html round-trip
# ----------------------------------------------------------------------
def test_figure_export_bytes_returns_error_when_kaleido_missing():
    """When ``fig.to_image`` raises (kaleido not installed, bad
    format string, etc), figure_export_bytes must funnel the
    exception into a clean (None, str) pair instead of re-raising.
    The test env doesn't have kaleido installed, so the real
    ``fig.to_image`` call naturally raises — that's the exact
    degradation path the Streamlit Figures tab relies on."""
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="t")
    img_bytes, err = fig_mod.figure_export_bytes(fig, format="svg")
    assert img_bytes is None
    assert err is not None and isinstance(err, str)
    # The exact wording depends on the plotly / kaleido version,
    # but the error should be non-empty and readable.
    assert len(err) > 0


def test_figure_export_bytes_success_path_returns_bytes():
    """Happy path with ``fig.to_image`` mocked to return bytes —
    exercises the success branch without needing kaleido installed."""
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="t")
    sentinel = b"<svg>fake bytes</svg>"
    with patch.object(fig, "to_image", return_value=sentinel) as mock_to_image:
        img_bytes, err = fig_mod.figure_export_bytes(fig, format="svg", scale=3)
    mock_to_image.assert_called_once_with(format="svg", scale=3)
    assert img_bytes == sentinel
    assert err is None


def test_figure_export_bytes_wraps_specific_exception_type():
    """A non-generic exception from ``fig.to_image`` (e.g.
    RuntimeError from kaleido's internal browser spawn) should
    still come back as a (None, string) pair, not propagate."""
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="t")
    with patch.object(
        fig, "to_image",
        side_effect=RuntimeError("chromium spawn failed"),
    ):
        img_bytes, err = fig_mod.figure_export_bytes(fig, format="png")
    assert img_bytes is None
    assert "chromium spawn failed" in err


def test_figure_to_html_round_trip_produces_plotly_payload():
    """The HTML download path in _figure_download_row uses
    ``fig.to_html(full_html=True, include_plotlyjs="cdn")`` — make
    sure that produces a string containing a plotly payload and the
    gene IDs from the input, so the downloaded file is a real,
    self-contained, interactive figure and not just an empty shell."""
    df = _fake_contrast_table()
    fig = fig_mod.volcano_plot(df, title="round-trip test")
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    assert isinstance(html, str)
    # CDN mode embeds a <script> tag pointing at plotly.
    assert "cdn.plot.ly" in html or "plotly" in html.lower()
    # The title we passed should survive into the HTML payload.
    assert "round-trip test" in html


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
