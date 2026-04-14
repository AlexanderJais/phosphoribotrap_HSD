"""FPKM, sign-consistency, and between-group Mann-Whitney analysis.

This is the Python-side secondary analysis — it has **no R dependency**
and is always available. It is a cross-check for the primary anota2seq
output, not a replacement.

For a 3-group design the meaningful statistics are:

* **Per-group sign consistency**: count how many of the 3 IP/INPUT pairs
  in a group have a log2 ratio with the same sign. A 3/3 call is the
  ceiling for ``n = 3``; binomial p-values are reported but will be
  dominated by the discrete nature of the sample size.
* **Between-group Mann-Whitney**: the three log2 ratios in the alt group
  vs the three in the ref group. This is the actual secondary
  statistic for a 3-vs-3 design, and the one that produces ranked
  lists usable for preranked GSEA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:  # scipy is defensive — the UI must still load without it
    from scipy.stats import binomtest, mannwhitneyu  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy missing
    _HAS_SCIPY = False

try:  # false_discovery_control is scipy ≥1.11
    from scipy.stats import false_discovery_control  # type: ignore
    _HAS_SCIPY_FDR = True
except Exception:  # pragma: no cover
    _HAS_SCIPY_FDR = False

from .logger import get_logger
from .samples import SampleRecord, pairs_by_group

logger = get_logger()


# ----------------------------------------------------------------------
# Salmon quant loading + FPKM computation
# ----------------------------------------------------------------------
def _read_quant_genes(sample_dir: Path) -> pd.DataFrame:
    """Read ``quant.genes.sf`` from a salmon output directory."""
    gene_path = sample_dir / "quant.genes.sf"
    if gene_path.exists():
        df = pd.read_csv(gene_path, sep="\t").rename(columns={"Name": "gene_id"})
        return df[["gene_id", "Length", "EffectiveLength", "NumReads"]]

    tx_path = sample_dir / "quant.sf"
    if not tx_path.exists():
        raise FileNotFoundError(f"No quant.sf in {sample_dir}")
    raise FileNotFoundError(
        f"No quant.genes.sf in {sample_dir} — rerun salmon with -g <tx2gene>"
    )


@dataclass
class SalmonLoadResult:
    counts: pd.DataFrame
    eff_length: pd.DataFrame
    fpkm: pd.DataFrame
    loaded: list[SampleRecord]    # records whose quant output was read
    missing: list[SampleRecord]   # records whose salmon output is absent


def load_salmon_matrix(
    records: Iterable[SampleRecord], salmon_root: Path
) -> SalmonLoadResult:
    """Load NumReads, EffectiveLength, and FPKM gene × sample matrices.

    Records without salmon output on disk are skipped (and logged) so
    a single missing ``quant.genes.sf`` does not crash the whole
    Analysis tab. The caller gets explicit ``loaded`` / ``missing``
    record lists alongside the matrices, which the UI uses to tell
    the user exactly which samples still need salmon.
    """
    salmon_root = Path(salmon_root)
    records_list = list(records)
    counts: dict[str, pd.Series] = {}
    eff_lens: dict[str, pd.Series] = {}
    shared_idx: Optional[pd.Index] = None
    loaded: list[SampleRecord] = []
    missing: list[SampleRecord] = []

    for rec in records_list:
        sdir = salmon_root / rec.name()
        try:
            df = _read_quant_genes(sdir)
        except FileNotFoundError as exc:
            logger.warning("salmon output missing for %s: %s", rec.name(), exc)
            missing.append(rec)
            continue
        df = df.set_index("gene_id")
        counts[rec.name()] = df["NumReads"]
        eff_lens[rec.name()] = df["EffectiveLength"]
        if shared_idx is None:
            shared_idx = df.index
        else:
            shared_idx = shared_idx.intersection(df.index)
        loaded.append(rec)

    if shared_idx is None or len(shared_idx) == 0:
        return SalmonLoadResult(
            counts=pd.DataFrame(),
            eff_length=pd.DataFrame(),
            fpkm=pd.DataFrame(),
            loaded=loaded,
            missing=missing,
        )

    counts_df = pd.DataFrame(counts).loc[shared_idx].fillna(0.0)
    eff_df = pd.DataFrame(eff_lens).loc[shared_idx].fillna(0.0)
    fpkm_df = compute_fpkm(counts_df, eff_df)
    return SalmonLoadResult(
        counts=counts_df,
        eff_length=eff_df,
        fpkm=fpkm_df,
        loaded=loaded,
        missing=missing,
    )


def compute_fpkm(counts: pd.DataFrame, eff_length: pd.DataFrame) -> pd.DataFrame:
    """Mortazavi FPKM: reads × 1e9 / (effective_length × library_total_reads)."""
    if counts.empty:
        return counts.copy()
    lib_totals = counts.sum(axis=0)
    denom = eff_length.replace(0, np.nan) * lib_totals
    fpkm = counts * 1.0e9 / denom
    return fpkm.fillna(0.0)


# ----------------------------------------------------------------------
# Per-pair ratios and per-group sign consistency
# ----------------------------------------------------------------------
@dataclass
class RatioResult:
    ratios: pd.DataFrame   # gene × pair-label matrix of log2(IP/Input)
    per_group: dict[str, pd.DataFrame]  # group -> summary per gene
    pair_labels: dict[str, list[str]]   # group -> list of pair labels


def pair_ratios(
    fpkm: pd.DataFrame,
    records: Iterable[SampleRecord],
    min_fpkm: float = 0.1,
) -> RatioResult:
    """Compute per-pair log2(IP/Input) ratios and per-group summaries."""
    by_group = pairs_by_group(records)
    ratio_cols: dict[str, pd.Series] = {}
    pair_labels: dict[str, list[str]] = {}

    for group, pr_list in by_group.items():
        for p in pr_list:
            ip_col = fpkm.get(p.ip.name())
            in_col = fpkm.get(p.input.name())
            if ip_col is None or in_col is None:
                logger.warning(
                    "missing FPKM column for pair %s rep %s", group, p.replicate
                )
                continue
            ip_safe = np.maximum(ip_col.values, min_fpkm)
            in_safe = np.maximum(in_col.values, min_fpkm)
            label = f"{group}_rep{p.replicate}"
            ratio_cols[label] = pd.Series(
                np.log2(ip_safe / in_safe), index=fpkm.index, name=label
            )
            pair_labels.setdefault(group, []).append(label)

    if not ratio_cols:
        return RatioResult(pd.DataFrame(index=fpkm.index), {}, {})

    ratios = pd.DataFrame(ratio_cols)

    per_group: dict[str, pd.DataFrame] = {}
    for group, labels in pair_labels.items():
        sub = ratios[labels]
        # log2(geomean of linear ratio) == arithmetic mean of the log2 values.
        geomean_log2 = sub.mean(axis=1)
        arithmetic = (2.0 ** sub).mean(axis=1)
        median_log2 = sub.median(axis=1)
        n_up = (sub > 0).sum(axis=1)
        n_down = (sub < 0).sum(axis=1)
        n = sub.shape[1]
        consistency = np.maximum(n_up, n_down)

        if _HAS_SCIPY:
            p_vals = np.array(
                [
                    binomtest(int(c), n, p=0.5, alternative="two-sided").pvalue
                    if n > 0 else np.nan
                    for c in consistency
                ],
                dtype=float,
            )
        else:
            p_vals = np.full(len(consistency), np.nan)

        per_group[group] = pd.DataFrame(
            {
                "log2_geomean_ratio": geomean_log2,
                "arith_mean_ratio": arithmetic,
                "median_log2_ratio": median_log2,
                "n_up": n_up,
                "n_down": n_down,
                "n_pairs": n,
                "consistency": consistency,
                "binom_p": p_vals,
            },
            index=ratios.index,
        )
    return RatioResult(ratios=ratios, per_group=per_group, pair_labels=pair_labels)


# ----------------------------------------------------------------------
# Between-group Mann-Whitney
# ----------------------------------------------------------------------
@dataclass
class ContrastResult:
    contrast: str
    alt_group: str
    ref_group: str
    table: pd.DataFrame
    ranked: pd.DataFrame   # two-column gene_id / score for preranked GSEA


def between_group_contrast(
    ratios: RatioResult,
    alt_group: str,
    ref_group: str,
) -> ContrastResult:
    """Mann-Whitney U between alt and ref group log2 ratios, per gene."""
    alt_labels = ratios.pair_labels.get(alt_group, [])
    ref_labels = ratios.pair_labels.get(ref_group, [])
    if not alt_labels or not ref_labels:
        empty = pd.DataFrame()
        return ContrastResult(
            f"{alt_group}_vs_{ref_group}", alt_group, ref_group, empty, empty
        )

    alt = ratios.ratios[alt_labels]
    ref = ratios.ratios[ref_labels]
    mean_alt = alt.mean(axis=1)
    mean_ref = ref.mean(axis=1)
    delta = mean_alt - mean_ref

    if _HAS_SCIPY:
        p_vals = np.empty(len(delta))
        for i in range(len(delta)):
            a = alt.iloc[i].values
            r = ref.iloc[i].values
            try:
                p_vals[i] = mannwhitneyu(a, r, alternative="two-sided").pvalue
            except ValueError:
                p_vals[i] = np.nan
    else:
        logger.warning("scipy not available — between-group Mann-Whitney p-values are NaN")
        p_vals = np.full(len(delta), np.nan)

    # Simple BH FDR
    p_adj = _bh_fdr(p_vals)

    # Tie-break by |delta_log2| desc so that at n=3-vs-3, where
    # mannwhitney_p can only take a handful of discrete values
    # (≈0.1, 0.2, 0.4, 0.7, 1.0), the most biologically interesting
    # genes in each p-tier sort to the top instead of falling out in
    # arbitrary index order.
    abs_delta = np.abs(delta.values)
    table = pd.DataFrame(
        {
            f"mean_log2_{alt_group}": mean_alt.values,
            f"mean_log2_{ref_group}": mean_ref.values,
            "delta_log2": delta.values,
            "mannwhitney_p": p_vals,
            "mannwhitney_padj": p_adj,
            "_abs_delta": abs_delta,
        },
        index=ratios.ratios.index,
    ).sort_values(
        ["mannwhitney_p", "_abs_delta"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns=["_abs_delta"])
    ranked = pd.DataFrame({"gene_id": table.index, "score": table["delta_log2"].values})
    ranked = ranked.sort_values("score", ascending=False, na_position="last")
    return ContrastResult(
        contrast=f"{alt_group}_vs_{ref_group}",
        alt_group=alt_group,
        ref_group=ref_group,
        table=table,
        ranked=ranked,
    )


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. Defers to scipy when available."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    valid = ~np.isnan(p)
    if not valid.any():
        return out

    if _HAS_SCIPY_FDR:
        out[valid] = false_discovery_control(p[valid], method="bh")
        return out

    # Hand-rolled fallback for the scipy-missing path.
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(pv)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    unordered = np.empty(m)
    unordered[order] = adj
    out[valid] = unordered
    return out
