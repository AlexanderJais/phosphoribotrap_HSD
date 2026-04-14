"""Smoke tests for FPKM math, sign-consistency, and between-group MWU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phosphotrap.fpkm import (
    SalmonLoadResult,
    between_group_contrast,
    compute_fpkm,
    load_salmon_matrix,
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


def test_between_group_mann_whitney_tie_breaks_by_abs_delta():
    """At n=3-vs-3 the Mann-Whitney p-value is extremely discrete —
    every enriched gene ends up tied at the minimum. The top of the
    ranking should then be ordered by ``|delta_log2|`` descending,
    not by arbitrary index order.
    """
    records, fpkm = _make_records_and_fpkm(n_genes=20)

    # Give the 5 enriched genes different amplitudes so the tie-break
    # has something to sort on. g0 is the largest, g4 the smallest.
    amplitudes = [200.0, 150.0, 100.0, 60.0, 20.0]
    for g, amp in zip(range(5), amplitudes):
        for rep in range(1, 4):
            fpkm.loc[f"g{g}", f"HSD1_IP{rep}"] = amp
            fpkm.loc[f"g{g}", f"HSD1_INPUT{rep}"] = 5.0

    ratios = pair_ratios(fpkm, records, min_fpkm=0.1)
    cr = between_group_contrast(ratios, alt_group="HSD1", ref_group="NCD")

    # All five enriched genes are tied at the minimum Mann-Whitney p;
    # the tie-break must sort them by |delta_log2| desc, so g0 is on
    # top and g4 at the bottom of that tier.
    head = cr.table.head(5).index.tolist()
    assert head == ["g0", "g1", "g2", "g3", "g4"]


def _write_quant_genes(directory: Path, gene_ids, counts, eff_lens) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "Name": gene_ids,
        "Length": [length * 2 for length in eff_lens],
        "EffectiveLength": eff_lens,
        "NumReads": counts,
    })
    df.to_csv(directory / "quant.genes.sf", sep="\t", index=False)


def test_load_salmon_matrix_skips_missing_samples(tmp_path: Path):
    """Missing quant files must not crash the loader; they come back
    in ``missing`` so the UI can tell the user which samples still
    need salmon.
    """
    salmon_root = tmp_path / "salmon"

    records = [
        SampleRecord(ccg_id="a", sample="A", comment="", replicate=1,
                     group="NCD", fraction="IP"),
        SampleRecord(ccg_id="b", sample="B", comment="", replicate=1,
                     group="NCD", fraction="INPUT"),
        SampleRecord(ccg_id="c", sample="C", comment="", replicate=3,
                     group="NCD", fraction="IP"),
    ]

    # Write quant.genes.sf for the first two only.
    _write_quant_genes(
        salmon_root / records[0].name(),
        gene_ids=["g1", "g2"],
        counts=[100.0, 50.0],
        eff_lens=[1000.0, 500.0],
    )
    _write_quant_genes(
        salmon_root / records[1].name(),
        gene_ids=["g1", "g2"],
        counts=[80.0, 60.0],
        eff_lens=[1000.0, 500.0],
    )
    # records[2] intentionally has no salmon output.

    result = load_salmon_matrix(records, salmon_root)

    assert isinstance(result, SalmonLoadResult)
    assert [r.name() for r in result.loaded] == [records[0].name(), records[1].name()]
    assert [r.name() for r in result.missing] == [records[2].name()]
    assert list(result.fpkm.columns) == [records[0].name(), records[1].name()]
    assert result.fpkm.shape[0] == 2  # 2 genes survived the intersection


def test_load_salmon_matrix_all_missing_returns_empty_but_does_not_crash(tmp_path: Path):
    records = [
        SampleRecord(ccg_id="x", sample="X", comment="", replicate=1,
                     group="NCD", fraction="IP"),
    ]
    result = load_salmon_matrix(records, tmp_path / "nonexistent_salmon_root")
    assert result.fpkm.empty
    assert result.loaded == []
    assert len(result.missing) == 1
