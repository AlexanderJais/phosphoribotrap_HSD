"""Nature-grade figure helpers for the phosphoribotrap manuscript.

This module is the single source of truth for the publication figures
produced by the Streamlit **Figures** tab. Everything here is pure
data-in, plotly-figure-out — no Streamlit imports, no session state.
That keeps the plotting code trivially testable and lets a CLI
wrapper (or a future headless export job) produce the same figures
without the Streamlit runtime loaded.

Conventions across every plot function:

* Return a ``plotly.graph_objects.Figure``. Callers can preview it via
  ``st.plotly_chart`` and/or export to SVG/PNG/PDF via ``kaleido``.
* ``title: str`` is always required, passed through verbatim.
* ``font_size: int`` scales the entire figure uniformly via the
  :func:`nature_theme` helper. Screen default is 14; drop to 7–8 for
  final-print panels that need to fit Nature's 89 mm single-column
  layout.
* Gene identifiers are always Ensembl ``gene_id`` strings (e.g.
  ``ENSMUSG00000034855``). Display labels are pulled from an explicit
  ``gene_labels: dict[gene_id, symbol]`` mapping that callers build
  once via :func:`resolve_symbols` against a tx2gene TSV.

Galanin signaling is the first-class highlight set for this project
and is exported as the :data:`GALANIN_GENES` tuple so the Figures tab
can populate the field out of the box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Mouse galanin signaling core:
#
# * Gal    — galanin propeptide (ligand)
# * Galp   — galanin-like peptide (ligand)
# * Galr1  — galanin receptor 1 (Gi/o, classical signaling)
# * Galr2  — galanin receptor 2 (Gq + Gi/o)
# * Galr3  — galanin receptor 3 (Gi/o)
#
# These are the genes we want highlighted in every Figures-tab panel
# by default. Case-insensitive matching happens inside ``resolve_symbols``,
# so callers can pass these as-is regardless of how the tx2gene TSV
# capitalises the ``gene_name`` column.
GALANIN_GENES: tuple[str, ...] = ("Gal", "Galp", "Galr1", "Galr2", "Galr3")


# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------
# Nature typography leans on Arial / Helvetica at ~7 pt for final
# print. Streamlit previews need bigger text to be legible on a
# monitor, so ``DEFAULT_FONT_SIZE`` is screen-oriented; the user can
# drop it to ~8 for the actual panel export.
NATURE_FONT_FAMILY = "Arial, Helvetica, sans-serif"
DEFAULT_FONT_SIZE = 14

# Colour palette. Deliberately conservative — Nature readers want
# information, not a circus. Galanin highlights are crimson because
# that's unambiguous against both light and dark background traces;
# custom highlights are a calm blue so users can overlay a second
# gene set without the figure turning into a heat map of noise.
COLOR_BACKGROUND = "#d1d5db"   # all-genes cloud
COLOR_NONSIG = "#9ca3af"       # above-padj-threshold, slightly darker
COLOR_SIG = "#374151"          # significant, non-highlighted
COLOR_GALANIN = "#dc2626"      # galanin core — bright crimson
COLOR_CUSTOM = "#2563eb"       # user-supplied highlights — calm blue
COLOR_DIAGONAL = "#6b7280"     # reference lines (diagonal, zero axes)


def nature_theme(base_font_size: int = DEFAULT_FONT_SIZE) -> dict:
    """Return a plotly ``update_layout`` kwarg dict for Nature-style figures.

    Apply via ``fig.update_layout(**nature_theme(font_size))``. Uses the
    ``plotly_white`` template as a starting point and then tightens the
    margins, sets Arial/Helvetica, and fixes paper + plot backgrounds
    to pure white so SVG exports don't carry a faint gray fill.
    """
    return {
        "template": "plotly_white",
        "font": {
            "family": NATURE_FONT_FAMILY,
            "size": base_font_size,
            "color": "#111111",
        },
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        "title": {
            "font": {
                "family": NATURE_FONT_FAMILY,
                "size": base_font_size + 2,
                "color": "#111111",
            },
            "x": 0.02,
            "xanchor": "left",
        },
    }


def _theme_with_title(base_font_size: int, title: str) -> dict:
    """Return a ``nature_theme`` dict with ``title.text`` populated.

    Internal helper so every plot function writes the same one-liner
    rather than spelling out the ``{**theme["title"], "text": title}``
    merge inline.
    """
    theme = nature_theme(base_font_size)
    theme["title"] = {**theme["title"], "text": title}
    return theme


# ----------------------------------------------------------------------
# Gene symbol resolution
# ----------------------------------------------------------------------
def load_gene_symbol_map(tx2gene_path: Path) -> dict[str, str]:
    """Build a case-insensitive ``{gene_symbol_lower: gene_id}`` map.

    Reads the 3-column tx2gene TSV produced by the Reference tab
    (``transcript_id <TAB> gene_id <TAB> gene_name``) and collapses it
    to a symbol-to-gene_id lookup. Duplicate symbols (different
    transcripts of the same gene) collapse to the same gene_id, so the
    map is deterministic regardless of row order.

    If the tx2gene TSV has only 2 columns (transcript_id + gene_id —
    the older two-column format), the returned map is empty and the
    caller should fall back to gene_id matching. Two-column tx2gene
    files don't carry symbol names at all.

    Raises :class:`FileNotFoundError` if ``tx2gene_path`` does not
    exist. Silently skips malformed rows.
    """
    path = Path(tx2gene_path)
    if not path.exists():
        raise FileNotFoundError(f"tx2gene TSV not found: {path}")

    out: dict[str, str] = {}
    with path.open("r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                # 2-column tx2gene has no symbol column; the caller
                # will have to match on gene_id directly.
                continue
            _tid, gene_id, gene_symbol = parts[0], parts[1], parts[2]
            if not gene_symbol or not gene_id:
                continue
            out[gene_symbol.lower()] = gene_id
    return out


def resolve_symbols(
    symbols: Iterable[str],
    symbol_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Resolve user-supplied gene symbols against a tx2gene symbol map.

    Returns ``(resolved, missing)`` where:

    * ``resolved`` is ``{gene_id: display_symbol}`` — the display
      symbol preserves the user's original capitalisation (``Gal``,
      not ``gal``) so plot labels read naturally even though the
      lookup itself is case-insensitive.
    * ``missing`` is the list of input symbols (in the user's
      original casing) that had no match. The Figures tab surfaces
      this as an inline warning so the user can correct typos like
      ``GlaR1``.

    Duplicate inputs collapse: passing ``["Gal", "GAL", "gal"]`` yields
    a single entry in ``resolved`` with the display symbol taken from
    the first occurrence.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []
    seen_gene_ids: set[str] = set()

    for sym in symbols:
        if sym is None:
            continue
        display = str(sym).strip()
        if not display:
            continue
        key = display.lower()
        gene_id = symbol_map.get(key)
        if gene_id is None:
            missing.append(display)
            continue
        if gene_id in seen_gene_ids:
            # Duplicate input for the same gene; keep the first label.
            continue
        seen_gene_ids.add(gene_id)
        resolved[gene_id] = display

    return resolved, missing


# ----------------------------------------------------------------------
# Volcano plot
# ----------------------------------------------------------------------
def _neg_log10(series: pd.Series) -> np.ndarray:
    """Safe ``-log10(p)`` with NaN for non-positive / missing values."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    out[mask] = -np.log10(arr[mask])
    return out


def volcano_plot(
    contrast_table: pd.DataFrame,
    *,
    title: str,
    p_col: str = "mannwhitney_p",
    padj_col: str = "mannwhitney_padj",
    delta_col: str = "delta_log2",
    alpha: float = 0.1,
    highlight_primary: Optional[dict[str, str]] = None,
    highlight_secondary: Optional[dict[str, str]] = None,
    font_size: int = DEFAULT_FONT_SIZE,
    primary_color: str = COLOR_GALANIN,
    secondary_color: str = COLOR_CUSTOM,
) -> go.Figure:
    """Build a Nature-grade volcano plot with an explicit highlight layer.

    The x axis is ``delta_col`` (default ``delta_log2``, the mean alt-
    minus-mean-ref log2 ratio); the y axis is ``-log10(p_col)`` so
    plot orientation matches every volcano a reviewer has ever seen.

    Four trace layers, rendered bottom-to-top:

    1. **Background** — every gene in ``contrast_table`` with a
       finite p-value, drawn in ``COLOR_BACKGROUND`` at alpha 0.5.
       Provides the characteristic volcano cloud.
    2. **Significant** — genes with ``padj_col <= alpha`` that are
       NOT in either highlight set. Drawn in ``COLOR_SIG`` so they're
       visually distinct from the noise but don't compete with the
       highlights.
    3. **Primary highlights** (``highlight_primary``) — galanin core
       by default. Drawn in ``primary_color``, larger, with text
       labels. Labels use the display strings from the dict value.
    4. **Secondary highlights** (``highlight_secondary``) — user's
       additional genes. Drawn in ``secondary_color``, same size as
       primary, labeled.

    ``contrast_table`` is expected to be indexed by ``gene_id`` (as
    produced by :class:`phosphotrap.fpkm.ContrastResult.table`), so
    the highlight dicts can be keyed directly by gene_id.

    Missing genes in the highlight dicts (i.e. gene_ids the user
    resolved against tx2gene but which don't appear in this
    contrast's results) are silently skipped — callers should warn
    upstream.
    """
    if contrast_table is None or contrast_table.empty:
        fig = go.Figure()
        fig.update_layout(
            **_theme_with_title(font_size, title),
            annotations=[
                {
                    "text": "No contrast data loaded.",
                    "showarrow": False,
                    "font": {"size": font_size},
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                }
            ],
        )
        return fig

    highlight_primary = highlight_primary or {}
    highlight_secondary = highlight_secondary or {}

    df = contrast_table.copy()
    df["_x"] = pd.to_numeric(df[delta_col], errors="coerce")
    df["_y"] = _neg_log10(df[p_col])
    if padj_col in df.columns:
        df["_padj"] = pd.to_numeric(df[padj_col], errors="coerce")
    else:
        df["_padj"] = np.nan

    # Gene IDs in the index.
    df.index = df.index.astype(str)

    highlight_ids = set(highlight_primary) | set(highlight_secondary)
    is_highlight = df.index.isin(highlight_ids)
    finite = df["_x"].notna() & df["_y"].notna()
    sig_mask = (df["_padj"] <= alpha) & finite & ~is_highlight
    bg_mask = finite & ~sig_mask & ~is_highlight

    fig = go.Figure()

    # Layer 1: background cloud.
    bg = df.loc[bg_mask]
    fig.add_trace(
        go.Scattergl(
            x=bg["_x"],
            y=bg["_y"],
            mode="markers",
            marker={
                "color": COLOR_BACKGROUND,
                "size": 4,
                "opacity": 0.5,
                "line": {"width": 0},
            },
            name="all genes",
            hovertemplate=(
                "%{customdata}<br>"
                f"{delta_col}: %{{x:.2f}}<br>"
                f"-log10({p_col}): %{{y:.2f}}<extra></extra>"
            ),
            customdata=bg.index,
        )
    )

    # Layer 2: significant (non-highlighted).
    sig = df.loc[sig_mask]
    fig.add_trace(
        go.Scattergl(
            x=sig["_x"],
            y=sig["_y"],
            mode="markers",
            marker={
                "color": COLOR_SIG,
                "size": 5,
                "opacity": 0.8,
                "line": {"width": 0},
            },
            name=f"padj ≤ {alpha}",
            hovertemplate=(
                "%{customdata}<br>"
                f"{delta_col}: %{{x:.2f}}<br>"
                f"-log10({p_col}): %{{y:.2f}}<extra></extra>"
            ),
            customdata=sig.index,
        )
    )

    # Layer 3: primary highlights (galanin).
    _add_highlight_trace(
        fig,
        df=df,
        highlight=highlight_primary,
        color=primary_color,
        name="Galanin signaling",
        p_col=p_col,
        delta_col=delta_col,
    )

    # Layer 4: secondary highlights (custom).
    _add_highlight_trace(
        fig,
        df=df,
        highlight=highlight_secondary,
        color=secondary_color,
        name="Custom highlights",
        p_col=p_col,
        delta_col=delta_col,
    )

    # Reference lines: y = -log10(alpha) and x = 0.
    fig.add_hline(
        y=-np.log10(alpha) if alpha > 0 else None,
        line={"color": COLOR_DIAGONAL, "width": 1, "dash": "dash"},
        opacity=0.6,
    )
    fig.add_vline(
        x=0,
        line={"color": COLOR_DIAGONAL, "width": 1, "dash": "dot"},
        opacity=0.5,
    )

    fig.update_layout(
        **_theme_with_title(font_size, title),
        xaxis={
            "title": "log<sub>2</sub> fold change (alt − ref)",
            "zeroline": False,
            "showline": True,
            "linecolor": "#111111",
            "ticks": "outside",
        },
        yaxis={
            "title": f"−log<sub>10</sub>({p_col})",
            "zeroline": False,
            "showline": True,
            "linecolor": "#111111",
            "ticks": "outside",
        },
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#d1d5db",
            "borderwidth": 1,
        },
        showlegend=True,
    )
    return fig


def _add_highlight_trace(
    fig: go.Figure,
    *,
    df: pd.DataFrame,
    highlight: dict[str, str],
    color: str,
    name: str,
    p_col: str,
    delta_col: str,
) -> None:
    """Add a labeled scatter trace for one highlight set to ``fig``.

    Silently no-ops if no highlight gene_ids appear in ``df`` — lets
    the volcano caller pass both primary and secondary dicts without
    guarding for empty intersections.
    """
    if not highlight:
        return
    hl_df = df.loc[df.index.intersection(list(highlight.keys()))]
    if hl_df.empty:
        return
    labels = [highlight[g] for g in hl_df.index]
    fig.add_trace(
        go.Scatter(
            x=hl_df["_x"],
            y=hl_df["_y"],
            mode="markers+text",
            marker={
                "color": color,
                "size": 11,
                "line": {"color": "#111111", "width": 1},
                "opacity": 0.95,
            },
            text=labels,
            textposition="top center",
            textfont={
                "family": NATURE_FONT_FAMILY,
                "size": 12,
                "color": "#111111",
            },
            name=name,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{delta_col}: %{{x:.2f}}<br>"
                f"-log10({p_col}): %{{y:.2f}}<extra></extra>"
            ),
        )
    )


def parse_highlight_text(text: str) -> list[str]:
    """Split a comma/whitespace/newline-separated blob into gene symbols.

    The Figures tab's "Additional highlight genes" text area accepts
    anything the user is likely to paste: ``Bdnf, Npy Pomc`` or
    a newline-separated list copy-pasted from a supplementary table.
    Empty entries and whitespace are stripped; nothing else is
    normalised — capitalisation is preserved for display and
    :func:`resolve_symbols` handles the case-insensitive match.
    """
    if not text:
        return []
    # Split on commas AND whitespace so any plausible paste works.
    chunks: list[str] = []
    for comma_chunk in text.split(","):
        for ws_chunk in comma_chunk.split():
            piece = ws_chunk.strip()
            if piece:
                chunks.append(piece)
    return chunks
