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
from typing import Iterable

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
