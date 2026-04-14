"""Smoke tests for FPKM math, sign-consistency, and between-group MWU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phosphotrap.fpkm import (
    between_group_contrast,
    compute_fpkm,
    pair_ratios,
)
from phosphotrap.samples import SampleRecord


def test_compute_fpkm_matches_mortazavi_formula():
    # One gene, two samples, known FPKMs.
    counts = pd.DataFrame(
        {"s1": [100.0, 900.0], "s2": [200.0, 800.0]},
        index=["g1", "g2"],
    )
    eff = pd.DataFrame(
        {"s1": [1000.0, 2000.0], "s2": [1000.0, 2000.0]},
        index=["g1", "g2"],
    )
    fpkm = compute_fpkm(counts, eff)

    # s1 total = 1000, s2 total = 1000.
    # g1 s1: 100 * 1e9 / (1000 * 1000) = 1e5
    # g2 s1: 900 * 1e9 / (2000 * 1000) = 4.5e5
    # g1 s2: 200 * 1e9 / (1000 * 1000) = 2e5
    # g2 s2: 800 * 1e9 / (2000 * 1000) = 4e5
    assert fpkm.loc["g1", "s1"] == pytest.approx(1e5)
    assert fpkm.loc["g2", "s1"] == pytest.approx(4.5e5)
    assert fpkm.loc["g1", "s2"] == pytest.approx(2e5)
    assert fpkm.loc["g2", "s2"] == pytest.approx(4e5)


def _make_records_and_fpkm(n_genes: int = 20):
    """Build a 6-sample synthetic FPKM matrix (NCD and HSD1, 3 pairs each).

    The ``up_in_hsd1`` genes have higher IP/Input in HSD1 than NCD;
    the rest are flat.
    """
    records = []
    cols = []
    for grp in ("NCD", "HSD1"):
        for rep in range(1, 4):
            ip = SampleRecord(
                ccg_id=f"c_{grp}_ip_{rep}",
                sample=f"IP_{grp}_{rep}",
                comment="",
                replicate=rep,
                group=grp,
                fraction="IP",
            )
            inp = SampleRecord(
                ccg_id=f"c_{grp}_in_{rep}",
                sample=f"IN_{grp}_{rep}",
                comment="",
                replicate=rep,
                group=grp,
                fraction="INPUT",
            )
            records.extend([ip, inp])
            cols.extend([ip.name(), inp.name()])

    rng = np.random.default_rng(42)
    data = rng.uniform(5, 10, size=(n_genes, len(cols)))
    fpkm = pd.DataFrame(data, columns=cols, index=[f"g{i}" for i in range(n_genes)])

    # For genes 0..4, amplify HSD1 IP columns and shrink HSD1 INPUT columns.
    for g in range(5):
        for rep in range(1, 4):
            ip_col = f"HSD1_IP{rep}"
            in_col = f"HSD1_INPUT{rep}"
            fpkm.loc[f"g{g}", ip_col] = 80.0
            fpkm.loc[f"g{g}", in_col] = 5.0

    return records, fpkm


def test_sign_consistency_three_of_three_on_enriched_genes():
    records, fpkm = _make_records_and_fpkm(n_genes=20)
    ratios = pair_ratios(fpkm, records, min_fpkm=0.1)

    hsd1 = ratios.per_group["HSD1"]
    for g in [f"g{i}" for i in range(5)]:
        row = hsd1.loc[g]
        assert row["n_up"] == 3
        assert row["n_down"] == 0
        assert row["consistency"] == 3
        assert row["log2_geomean_ratio"] > 2  # log2(80/5) ≈ 4


def test_between_group_mann_whitney_detects_shifted_group():
    records, fpkm = _make_records_and_fpkm(n_genes=20)
    ratios = pair_ratios(fpkm, records, min_fpkm=0.1)
    cr = between_group_contrast(ratios, alt_group="HSD1", ref_group="NCD")

    assert cr.alt_group == "HSD1"
    assert cr.ref_group == "NCD"
    # Enriched genes should have the largest delta_log2 values.
    top = cr.table.sort_values("delta_log2", ascending=False).head(5).index.tolist()
    assert set(top) == {f"g{i}" for i in range(5)}
    # Ranked output for preranked GSEA.
    assert list(cr.ranked.columns) == ["gene_id", "score"]
    assert cr.ranked.iloc[0]["score"] > cr.ranked.iloc[-1]["score"]
