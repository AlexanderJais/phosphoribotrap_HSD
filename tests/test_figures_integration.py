"""End-to-end integration tests for phosphotrap.figures with real shapes.

Chunks 1-6 of the Figures-tab series tested each plot function
against synthetic fixtures built directly in the test file. Those
fixtures happen to match the production shapes (gene_id index,
.name() method on records, etc.) but only because the tests were
written against the same spec.

This file plugs the gap by running real code from ``phosphotrap.fpkm``
and ``phosphotrap.samples`` against tiny on-disk ``quant.genes.sf``
fixtures, building genuine :class:`SalmonLoadResult` /
:class:`RatioResult` / :class:`ContrastResult` objects, and then
passing those *real dataclasses* into every plot function. Catches
off-by-one errors in column naming, attribute access, or DataFrame
layout that a hand-built fixture would paper over.

The fixtures are tiny — 10 genes × 18 samples (3 groups × 3
replicates × IP + INPUT) — so the whole suite runs in well under a
second. No fastp / salmon / R binaries are touched.

The ``quant.genes.sf`` schema (from the real phosphotrap pipeline):

    Name    Length  EffectiveLength TPM     NumReads

``phosphotrap.fpkm._read_quant_genes`` reads it, renames ``Name`` ->
``gene_id``, and returns the 4-column subset the pipeline needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from phosphotrap import figures as fig_mod
from phosphotrap.fpkm import (
    between_group_contrast,
    load_salmon_matrix,
    pair_ratios,
)
from phosphotrap.samples import SampleRecord


# ----------------------------------------------------------------------
# On-disk fixture — mini quant.genes.sf files under a tmp_path
# ----------------------------------------------------------------------
# 10 genes: first 5 are the galanin set + a handful of neighbours so
# the volcano cloud has some background points. Values are chosen so
# galanin genes have a deliberate 2x enrichment in HSD IPs and the
# background is roughly flat — enough signal for the contrast panels
# to assert non-trivially.
_MOCK_GENES = [
    "ENSMUSG00000004366",  # Gal
    "ENSMUSG00000033128",  # Galp
    "ENSMUSG00000006542",  # Galr1
    "ENSMUSG00000037844",  # Galr2
    "ENSMUSG00000043964",  # Galr3
    "ENSMUSG00000000001",  # filler 1
    "ENSMUSG00000000002",  # filler 2
    "ENSMUSG00000000003",  # filler 3
    "ENSMUSG00000000004",  # filler 4
    "ENSMUSG00000000005",  # filler 5
]
_GALANIN_IDS = set(_MOCK_GENES[:5])
_GENE_LENGTHS = [1500, 1200, 2400, 2100, 2700, 1000, 1100, 1300, 1400, 1600]


def _build_records() -> list[SampleRecord]:
    """18 records matching the default phosphotrap design:
    3 groups × 3 replicates × {IP, INPUT}."""
    records: list[SampleRecord] = []
    for group, reps in (
        ("NCD", (1, 3, 4)),
        ("HSD1", (5, 6, 8)),
        ("HSD3", (9, 10, 11)),
    ):
        for fraction in ("IP", "INPUT"):
            for rep in reps:
                records.append(
                    SampleRecord(
                        ccg_id=f"ccg_{group}_{fraction}_{rep}",
                        sample=f"{fraction}{rep}",
                        comment="",
                        replicate=rep,
                        group=group,
                        fraction=fraction,
                    )
                )
    assert len(records) == 18
    return records


def _fake_counts(
    group: str, fraction: str, rep: int, seed_base: int = 0
) -> np.ndarray:
    """Return 10 NumReads values for a single (group, fraction, rep).

    The filler baseline is deliberately ~20x the galanin baseline so
    that galanin reads stay a small fraction of library size — if we
    balance them evenly, FPKM's per-sample library-size normalisation
    cancels most of the galanin enrichment and the fixture loses the
    signal the tests assert on.

    Signal structure:
    * Galanin genes (indices 0-4) start at 200 reads.
    * Filler genes sit at 4000 reads (20x baseline).
    * IP enrichment: galanin 3x over INPUT.
    * HSD effect: further 2x bump on galanin in HSD IPs.
    * Noise: ±10% per gene per sample.
    """
    rng = np.random.default_rng(seed_base + hash((group, fraction, rep)) % 1000)
    counts = np.array(
        [200, 200, 200, 200, 200, 4000, 4000, 4000, 4000, 4000],
        dtype=float,
    )
    # IP enrichment: galanin 3x.
    if fraction == "IP":
        counts[:5] *= 3.0
    # HSD effect: further 2x bump on galanin in HSD IPs.
    if fraction == "IP" and group in ("HSD1", "HSD3"):
        counts[:5] *= 2.0
    # Noise: ±10%.
    counts *= rng.uniform(0.9, 1.1, size=10)
    return counts


def _write_quant_genes(
    salmon_root: Path, rec: SampleRecord, counts: np.ndarray
) -> None:
    """Write a real phosphotrap-format quant.genes.sf for one record."""
    sample_dir = salmon_root / rec.name()
    sample_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "Name": _MOCK_GENES,
            "Length": _GENE_LENGTHS,
            "EffectiveLength": [l - 50 for l in _GENE_LENGTHS],
            "TPM": counts / counts.sum() * 1e6,
            "NumReads": counts,
        }
    )
    df.to_csv(sample_dir / "quant.genes.sf", sep="\t", index=False)
    # ``fpkm._read_quant_genes`` checks quant.genes.sf first so
    # quant.sf isn't strictly required, but the pipeline contract
    # says it's present, so write a dummy one for completeness.
    (sample_dir / "quant.sf").write_text("")


@pytest.fixture
def real_analysis_state(tmp_path: Path):
    """Build real SalmonLoadResult + RatioResult + two ContrastResults
    via the production fpkm/samples code paths against mini fixtures.

    Returns a dict with keys:
      - records:        list[SampleRecord] (18 rows)
      - salmon_result:  SalmonLoadResult (10 genes × 18 samples)
      - ratios:         RatioResult
      - contrast_hsd1:  ContrastResult for HSD1_vs_NCD
      - contrast_hsd3:  ContrastResult for HSD3_vs_NCD
    """
    records = _build_records()
    salmon_root = tmp_path / "salmon"
    for rec in records:
        _write_quant_genes(salmon_root, rec, _fake_counts(
            rec.group, rec.fraction, rec.replicate
        ))

    salmon_result = load_salmon_matrix(records, salmon_root)
    assert not salmon_result.fpkm.empty
    assert salmon_result.fpkm.shape == (10, 18)
    assert salmon_result.loaded == records
    assert salmon_result.missing == []

    ratios = pair_ratios(salmon_result.fpkm, salmon_result.loaded)
    assert not ratios.ratios.empty
    # 9 matched animal pairs total (3 per group).
    assert ratios.ratios.shape == (10, 9)

    contrast_hsd1 = between_group_contrast(
        ratios, alt_group="HSD1", ref_group="NCD"
    )
    contrast_hsd3 = between_group_contrast(
        ratios, alt_group="HSD3", ref_group="NCD"
    )
    assert not contrast_hsd1.table.empty
    assert not contrast_hsd3.table.empty
    # Sanity check on the FIXTURE (not the plot functions): the
    # galanin set has an HSD IP enrichment baked in, so the mean
    # delta across all 5 galanin genes should be positive in each
    # contrast. Per-gene values can dip slightly negative due to
    # the ~10% per-sample noise, which is why we check the mean.
    galanin_mean_hsd1 = contrast_hsd1.table.loc[
        list(_GALANIN_IDS), "delta_log2"
    ].mean()
    galanin_mean_hsd3 = contrast_hsd3.table.loc[
        list(_GALANIN_IDS), "delta_log2"
    ].mean()
    assert galanin_mean_hsd1 > 0.2, (
        f"Fixture drift: expected positive galanin enrichment in "
        f"HSD1, got mean delta {galanin_mean_hsd1:.3f}"
    )
    assert galanin_mean_hsd3 > 0.2, (
        f"Fixture drift: expected positive galanin enrichment in "
        f"HSD3, got mean delta {galanin_mean_hsd3:.3f}"
    )

    return {
        "records": records,
        "salmon_result": salmon_result,
        "ratios": ratios,
        "contrast_hsd1": contrast_hsd1,
        "contrast_hsd3": contrast_hsd3,
    }


# ----------------------------------------------------------------------
# Plot functions against real dataclass shapes
# ----------------------------------------------------------------------
def _gene_labels() -> dict[str, str]:
    return dict(zip(_MOCK_GENES[:5], ["Gal", "Galp", "Galr1", "Galr2", "Galr3"]))


def test_volcano_plot_against_real_contrast_result(real_analysis_state):
    """The volcano must accept ``ContrastResult.table`` directly
    (gene_id index, delta_log2/mannwhitney_p/mannwhitney_padj columns)
    and produce a 4-layer figure with galanin labels."""
    cr = real_analysis_state["contrast_hsd1"]
    fig = fig_mod.volcano_plot(
        cr.table,
        title="HSD1_vs_NCD (integration)",
        highlight_primary=_gene_labels(),
    )
    assert isinstance(fig, go.Figure)
    # Background + sig + galanin = 3 traces (no secondary highlights).
    assert len(fig.data) == 3
    galanin_trace = next(t for t in fig.data if t.name == "Galanin signaling")
    # All 5 galanin genes should be present in the labeled trace.
    # Order reflects the contrast-table index order (which
    # ``between_group_contrast`` sorts by mannwhitney_p then
    # |delta_log2|), not dict-insertion order of the highlight
    # dict — so we assert on the SET of labels, not the sequence.
    assert set(galanin_trace.text) == {"Gal", "Galp", "Galr1", "Galr2", "Galr3"}


def test_per_gene_strip_against_real_ratio_result(real_analysis_state):
    """Strip plot must accept ``RatioResult.ratios`` + ``.pair_labels``
    directly (real DataFrame layout, real pair_labels dict keyed by
    group) and produce one subplot per gene."""
    ratios = real_analysis_state["ratios"]
    fig = fig_mod.per_gene_strip(
        ratios.ratios,
        ratios.pair_labels,
        title="Per-gene log₂(IP/Input) — integration",
        gene_labels=_gene_labels(),
        primary_ids=_GALANIN_IDS,
        group_order=["NCD", "HSD1", "HSD3"],
    )
    assert isinstance(fig, go.Figure)
    # 5 genes × (3 groups dot traces + 1 mean trace) = 20 traces.
    assert len(fig.data) == 20
    # Every galanin dot trace should show 3 animals per group.
    dot_traces = [
        t for t in fig.data
        if t.mode == "markers"
        and getattr(t.marker, "symbol", None) != "line-ew"
    ]
    # 5 genes × 3 groups = 15 dot traces.
    assert len(dot_traces) == 15
    for trace in dot_traces:
        assert len(trace.y) == 3


def test_expression_heatmap_against_real_salmon_result(real_analysis_state):
    """Heatmap must accept ``SalmonLoadResult.fpkm`` (gene_id index,
    sample columns) and ``SalmonLoadResult.loaded`` (SampleRecord
    list) directly."""
    sr = real_analysis_state["salmon_result"]
    fig = fig_mod.expression_heatmap(
        sr.fpkm,
        sr.loaded,
        title="Galanin expression — integration",
        gene_labels=_gene_labels(),
        group_order=["NCD", "HSD1", "HSD3"],
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    # 5 genes rendered in the requested order.
    assert list(fig.data[0].y) == ["Gal", "Galp", "Galr1", "Galr2", "Galr3"]
    # 18 real columns + 5 gap columns across 6 (group × fraction) blocks.
    assert len(fig.data[0].x) == 23


def test_cross_contrast_scatter_against_real_contrast_results(
    real_analysis_state,
):
    """Cross-contrast scatter must accept two ``ContrastResult.table``
    DataFrames directly and produce a square figure with the
    galanin genes highlighted on the diagonal."""
    hsd1 = real_analysis_state["contrast_hsd1"]
    hsd3 = real_analysis_state["contrast_hsd3"]
    fig = fig_mod.cross_contrast_scatter(
        hsd1.table,
        hsd3.table,
        label_a="HSD1_vs_NCD",
        label_b="HSD3_vs_NCD",
        title="Integration — cross-contrast",
        highlight_primary=_gene_labels(),
    )
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "all genes" in trace_names
    assert "y = x" in trace_names
    assert "Galanin signaling" in trace_names
    # Axes locked square.
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
    # Intersection should be all 10 genes (same fixture).
    bg_trace = next(t for t in fig.data if t.name == "all genes")
    galanin_trace = next(t for t in fig.data if t.name == "Galanin signaling")
    assert len(bg_trace.x) == 5  # 10 shared - 5 highlighted
    assert len(galanin_trace.x) == 5


def test_regmode_classification_against_empty_anota(real_analysis_state):
    """Full anota2seq integration needs R, which isn't in the test env.
    Instead feed an empty anota_results dict and verify the helper
    returns an empty DataFrame with the correct schema — the same
    code path the Figures tab hits before anota2seq has been run."""
    df = fig_mod.regmode_classification({}, _gene_labels())
    assert list(df.columns) == ["gene", "contrast", "mode"]
    assert len(df) == 0


def test_full_figure_pipeline_survives_real_data_end_to_end(
    real_analysis_state,
):
    """Belt-and-braces: run all five plot functions back-to-back
    against the real state. The point isn't to re-check what the
    per-function tests already assert; it's to catch cross-
    function state bleed or accidental DataFrame mutation."""
    state = real_analysis_state
    labels = _gene_labels()

    # Panel A: both contrasts.
    fig_mod.volcano_plot(
        state["contrast_hsd1"].table, title="A1",
        highlight_primary=labels,
    )
    fig_mod.volcano_plot(
        state["contrast_hsd3"].table, title="A2",
        highlight_primary=labels,
    )
    # Panel B.
    fig_mod.per_gene_strip(
        state["ratios"].ratios,
        state["ratios"].pair_labels,
        title="B",
        gene_labels=labels,
        primary_ids=_GALANIN_IDS,
    )
    # Panel C.
    fig_mod.expression_heatmap(
        state["salmon_result"].fpkm,
        state["salmon_result"].loaded,
        title="C",
        gene_labels=labels,
    )
    # Panel D.
    fig_mod.regmode_classification({}, labels)
    # Panel E.
    fig_mod.cross_contrast_scatter(
        state["contrast_hsd1"].table,
        state["contrast_hsd3"].table,
        label_a="HSD1_vs_NCD",
        label_b="HSD3_vs_NCD",
        title="E",
        highlight_primary=labels,
    )

    # None of the plot functions should have mutated the input
    # DataFrames. Spot-check that the raw contrast table still has
    # the original column set.
    assert "delta_log2" in state["contrast_hsd1"].table.columns
    assert "mannwhitney_p" in state["contrast_hsd1"].table.columns
    assert state["salmon_result"].fpkm.shape == (10, 18)
