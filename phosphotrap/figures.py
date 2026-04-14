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

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .logger import get_logger

_logger = get_logger()

# Mouse galanin signaling core (legacy seed set — kept for
# backward-compat with existing tests and as a power-user quick-
# populate option in the Figures tab). The tab no longer pre-fills
# this list by default: users supply their own primary-highlight
# set via a text area, optionally sourced from a GO term via
# :func:`fetch_go_term_genes`.
#
# * Gal    — galanin propeptide (ligand)
# * Galp   — galanin-like peptide (ligand)
# * Galr1  — galanin receptor 1 (Gi/o, classical signaling)
# * Galr2  — galanin receptor 2 (Gq + Gi/o)
# * Galr3  — galanin receptor 3 (Gi/o)
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
# GO-term gene fetch (QuickGO REST)
# ----------------------------------------------------------------------
# Default taxon ID is mouse (NCBI 10090) because this app is a mouse
# phosphoribotrap. The Figures tab exposes the taxon as a config so
# users on rat / human references can override without editing code.
GO_DEFAULT_TAXON = "10090"

# QuickGO is the EBI-hosted, publicly accessible GO annotation search
# service. We use the annotation search endpoint because it returns
# one result per annotation (with the gene product symbol already
# resolved) rather than requiring a second lookup pass. ``goUsage``
# is set to ``descendants`` so GO:0007218 (neuropeptide signaling
# pathway) pulls in its more-specific child terms as well — which
# matches how a biologist intuitively thinks about "genes in this
# pathway". The API is paginated; we honour ``pageInfo.total`` and
# walk forward until we've consumed every page.
_QUICKGO_ANNOTATION_SEARCH = (
    "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
)
_QUICKGO_PAGE_LIMIT = 100  # max allowed by the API
_QUICKGO_TIMEOUT_S = 30
_QUICKGO_USER_AGENT = "phosphoribotrap-figures/1.0"


def _valid_go_id(go_id: str) -> bool:
    """Structural sanity check on a user-supplied GO ID string.

    Accepts ``GO:`` prefix followed by exactly seven digits (the
    canonical GO curie format, e.g. ``GO:0007218``). Rejects empty
    strings, whitespace, arbitrary URLs, and SQL-injection-shaped
    inputs so we never splice unsanitised user input into an HTTPS
    request URL.
    """
    if not go_id or not isinstance(go_id, str):
        return False
    parts = go_id.strip().split(":")
    if len(parts) != 2 or parts[0] != "GO":
        return False
    return len(parts[1]) == 7 and parts[1].isdigit()


def _default_quickgo_fetcher(url: str, timeout: float) -> dict:
    """Default HTTP fetcher: urllib GET, JSON decode, dict return.

    Isolated so tests can inject a mock fetcher and avoid hitting
    the real QuickGO service. The fetcher contract is simple:
    take a URL + timeout, return the parsed JSON body as a dict,
    or raise one of ``urllib.error.URLError``, ``TimeoutError``,
    ``json.JSONDecodeError``, or ``OSError`` on failure.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _QUICKGO_USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body)


def fetch_go_term_genes(
    go_id: str,
    *,
    taxon: str = GO_DEFAULT_TAXON,
    cache_dir: Optional[Path] = None,
    fetcher: Optional[Callable[[str, float], dict]] = None,
    timeout: float = _QUICKGO_TIMEOUT_S,
    force_refresh: bool = False,
) -> set[str]:
    """Return the set of gene symbols annotated with a GO term.

    Queries QuickGO's annotation search for ``go_id`` restricted to
    ``taxon`` (default mouse = 10090), using ``goUsage=descendants``
    so child terms are included. The returned set contains the
    gene-product symbols exactly as QuickGO reports them — casing
    matches the source annotation, which is what :func:`resolve_symbols`
    can then translate into gene IDs via the tx2gene map.

    **Caching.** When ``cache_dir`` is supplied, the fetched result
    is written to ``<cache_dir>/go_<sanitised_id>_<taxon>.json`` with
    a fetch timestamp. A subsequent call with the same arguments
    reads from disk instead of hitting the network, making the
    Figures tab responsive offline once a gene list has been
    populated at least once. Pass ``force_refresh=True`` to bypass
    the cache and re-query.

    **Error handling.** On a network / parse failure:

    1. If a cache file exists, return the cached gene set and log a
       warning. The app stays usable on flaky networks.
    2. Otherwise, raise ``RuntimeError`` with a user-readable message
       so the Figures tab can surface it via ``st.error``.

    ``fetcher`` lets tests inject a stub HTTP client — it defaults
    to a urllib-based implementation that talks to QuickGO directly.

    Raises ``ValueError`` for malformed ``go_id`` strings (anything
    that isn't the canonical ``GO:dddddd`` curie) so callers can
    short-circuit before opening any connections.
    """
    if not _valid_go_id(go_id):
        raise ValueError(
            f"Not a valid GO ID: {go_id!r}. Expected format 'GO:dddddddd' "
            f"(e.g. 'GO:0007218' for neuropeptide signaling pathway)."
        )
    taxon = str(taxon).strip()
    if not taxon.isdigit():
        raise ValueError(
            f"Not a valid NCBI taxon ID: {taxon!r}. Expected a digit string "
            f"(e.g. '10090' for mouse, '10116' for rat, '9606' for human)."
        )

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        # Sanitise: GO:0007218 -> GO_0007218 (no colons in filenames).
        cache_path = cache_dir / f"go_{go_id.replace(':', '_')}_{taxon}.json"
        if cache_path.exists() and not force_refresh:
            try:
                payload = json.loads(cache_path.read_text())
                cached_symbols = payload.get("symbols")
                if isinstance(cached_symbols, list):
                    return {str(s) for s in cached_symbols if s}
            except (OSError, json.JSONDecodeError) as exc:
                _logger.warning(
                    "GO cache read failed for %s: %s — refetching",
                    cache_path, exc,
                )

    http_fetcher = fetcher or _default_quickgo_fetcher

    # Walk the QuickGO pagination until we've consumed every page.
    # ``numberOfHits`` is typically equal to ``pageInfo.total *
    # resultsPerPage`` (approximately), so we trust ``pageInfo.total``
    # as the loop bound with a hard cap of 100 pages to protect
    # against a runaway response in the unlikely event the API
    # misreports pagination metadata.
    symbols: set[str] = set()
    page = 1
    max_pages = 100
    try:
        while page <= max_pages:
            query = urllib.parse.urlencode(
                [
                    ("goId", go_id),
                    ("taxonId", taxon),
                    ("goUsage", "descendants"),
                    ("limit", str(_QUICKGO_PAGE_LIMIT)),
                    ("page", str(page)),
                ]
            )
            url = f"{_QUICKGO_ANNOTATION_SEARCH}?{query}"
            _logger.info("GO fetch: %s", url)
            payload = http_fetcher(url, timeout)
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                break
            for row in results:
                if not isinstance(row, dict):
                    continue
                sym = row.get("symbol") or row.get("geneProductSymbol")
                if sym:
                    symbols.add(str(sym))
            page_info = payload.get("pageInfo") if isinstance(payload, dict) else None
            total_pages = 1
            if isinstance(page_info, dict):
                total_pages = int(page_info.get("total") or 1)
            if page >= total_pages:
                break
            page += 1
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        # Graceful degradation: if we have an old cache, return that
        # rather than failing outright. Otherwise raise a friendly
        # RuntimeError the Figures tab can display.
        _logger.warning(
            "GO fetch failed for %s (taxon %s): %s", go_id, taxon, exc
        )
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                cached_symbols = payload.get("symbols") or []
                _logger.info(
                    "GO fetch: falling back to cached %s (%d symbols)",
                    cache_path, len(cached_symbols),
                )
                return {str(s) for s in cached_symbols if s}
            except (OSError, json.JSONDecodeError):
                pass
        raise RuntimeError(
            f"Could not fetch GO term {go_id} from QuickGO "
            f"(taxon {taxon}): {exc}. Check your network connection, "
            f"or paste the gene list manually into the volcano filter "
            f"text area."
        ) from exc

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "go_id": go_id,
                        "taxon": taxon,
                        "fetched_at": time.time(),
                        "symbols": sorted(symbols),
                    },
                    indent=2,
                )
            )
        except OSError as exc:
            _logger.warning("GO cache write failed for %s: %s", cache_path, exc)

    return symbols


# ----------------------------------------------------------------------
# anota2seq regulatory mode classification table
# ----------------------------------------------------------------------
# anota2seq writes one TSV per regulatory mode (translation, buffering,
# mRNA abundance). A gene can appear in at most one of those three,
# plus an implicit "not significant" if it's in none. These constants
# keep the mode names consistent across the helper, the tests, and the
# Figures tab. ``_REGMODE_RANK`` determines display priority when a
# gene somehow ended up in multiple modes (shouldn't happen with a
# well-behaved anota2seq run but belt-and-braces if cached TSVs drift
# out of sync).
_REGMODE_TRANSLATION = "translation"
_REGMODE_BUFFERING = "buffering"
_REGMODE_MRNA = "mRNA abundance"
_REGMODE_NS = "n.s."
_REGMODE_RANK = {
    _REGMODE_TRANSLATION: 0,
    _REGMODE_BUFFERING: 1,
    _REGMODE_MRNA: 2,
    _REGMODE_NS: 3,
}


def _regmode_for_gene(
    gene_id: str,
    translation: pd.DataFrame,
    buffering: pd.DataFrame,
    mrna_abundance: pd.DataFrame,
) -> str:
    """Return which anota2seq mode a single gene belongs to.

    Each input is one of the three per-mode TSVs written by
    ``phosphotrap.anota2seq_runner``. They carry a ``gene_id`` column
    identifying which genes fell into that regulatory mode for the
    contrast being inspected. Absence from all three means "not
    significant at the configured thresholds".

    If a gene appears in multiple modes (shouldn't happen but could
    if a stale cached TSV disagrees with a fresh one), the lowest-
    rank mode from ``_REGMODE_RANK`` wins so translation > buffering
    > mRNA abundance in display priority.
    """
    hits: list[str] = []
    for name, df in (
        (_REGMODE_TRANSLATION, translation),
        (_REGMODE_BUFFERING, buffering),
        (_REGMODE_MRNA, mrna_abundance),
    ):
        if df is None or df.empty or "gene_id" not in df.columns:
            continue
        if (df["gene_id"] == gene_id).any():
            hits.append(name)
    if not hits:
        return _REGMODE_NS
    # Lowest rank wins.
    return min(hits, key=lambda n: _REGMODE_RANK[n])


def regmode_classification(
    anota_results: dict,
    gene_labels: dict[str, str],
) -> pd.DataFrame:
    """Build a per-gene × per-contrast regulatory-mode table.

    ``anota_results`` is ``{contrast_name: Anota2seqResult}`` — the
    Figures tab builds this from ``st.session_state.analysis`` by
    filtering for entries that have a successful anota2seq run.
    ``Anota2seqResult`` is duck-typed to need only
    ``.translation``, ``.buffering``, ``.mrna_abundance`` attributes
    each exposing a DataFrame with a ``gene_id`` column — which is
    exactly what :mod:`phosphotrap.anota2seq_runner` produces.

    Returns a ``DataFrame`` with one row per
    (gene_symbol × contrast) combination, columns:

    * ``gene``     — display symbol (from ``gene_labels``)
    * ``contrast`` — contrast name (e.g. ``HSD1_vs_NCD``)
    * ``mode``     — one of "translation", "buffering",
                     "mRNA abundance", or "n.s."

    Rows are sorted so translation hits float to the top (lowest
    ``_REGMODE_RANK``), then alphabetically by gene symbol then by
    contrast name for stable display. The output is Streamlit-
    friendly — callers can pass it straight to ``st.dataframe``
    with ``hide_index=True`` for a manuscript-style table.

    If ``anota_results`` is empty or none of the requested genes
    appear anywhere, the returned DataFrame still has the three
    columns but zero rows — simpler to render than a None check.
    """
    rows: list[dict] = []
    for contrast_name, result in (anota_results or {}).items():
        if result is None:
            continue
        translation = getattr(result, "translation", None)
        buffering = getattr(result, "buffering", None)
        mrna = getattr(result, "mrna_abundance", None)
        for gene_id, symbol in gene_labels.items():
            mode = _regmode_for_gene(gene_id, translation, buffering, mrna)
            rows.append(
                {
                    "gene": symbol,
                    "contrast": contrast_name,
                    "mode": mode,
                }
            )

    df = pd.DataFrame(rows, columns=["gene", "contrast", "mode"])
    if df.empty:
        return df
    df["_rank"] = df["mode"].map(_REGMODE_RANK).fillna(999).astype(int)
    df = df.sort_values(
        ["_rank", "gene", "contrast"],
        kind="stable",
    ).drop(columns=["_rank"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Cross-contrast consistency scatter
# ----------------------------------------------------------------------
def cross_contrast_scatter(
    contrast_a: pd.DataFrame,
    contrast_b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
    title: str,
    delta_col: str = "delta_log2",
    highlight_primary: Optional[dict[str, str]] = None,
    highlight_secondary: Optional[dict[str, str]] = None,
    font_size: int = DEFAULT_FONT_SIZE,
    primary_color: str = COLOR_GALANIN,
    secondary_color: str = COLOR_CUSTOM,
    primary_label: str = "Galanin signaling",
    secondary_label: str = "Custom highlights",
) -> go.Figure:
    """Scatter of ``delta_log2`` in contrast A vs contrast B, per gene.

    A 3-vs-3 design has very little statistical power on any single
    contrast. Cross-contrast consistency is the single most useful
    piece of evidence you can put in a manuscript: a gene that moves
    in the same direction under BOTH HSD1 and HSD3 is far more
    believable than a single-contrast nominal hit. Points near the
    diagonal are consistent across both conditions; points in the
    off-diagonal quadrants are inconsistent and should be deprioritised.

    Inputs are the ``.table`` DataFrames from two
    :class:`phosphotrap.fpkm.ContrastResult` objects (one per contrast),
    indexed by ``gene_id``. Only genes present in BOTH tables are
    plotted; the intersection is silent — callers are expected to
    check matrix shapes upstream if they care about dropped genes.

    Layers, bottom-to-top:

    1. Background cloud of all shared genes in ``COLOR_BACKGROUND``.
    2. Diagonal ``y = x`` reference line (perfect reproducibility).
    3. Primary highlights (galanin) — larger, crimson, labeled.
    4. Secondary highlights (user's custom genes) — blue, labeled.

    Axes are locked to the same range (min/max of all plotted data
    plus 0.5 padding) so the diagonal is visually exactly 45°.
    """
    if contrast_a is None or contrast_b is None or contrast_a.empty or contrast_b.empty:
        fig = go.Figure()
        fig.update_layout(
            **_theme_with_title(font_size, title),
            annotations=[
                {
                    "text": "Need both contrast tables loaded.",
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

    a = contrast_a[[delta_col]].rename(columns={delta_col: "_a"}).copy()
    b = contrast_b[[delta_col]].rename(columns={delta_col: "_b"}).copy()
    a.index = a.index.astype(str)
    b.index = b.index.astype(str)
    joined = a.join(b, how="inner")
    joined = joined.dropna()

    if joined.empty:
        fig = go.Figure()
        fig.update_layout(
            **_theme_with_title(font_size, title),
            annotations=[
                {
                    "text": "No genes in common between the two contrasts.",
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
    highlight_ids = set(highlight_primary) | set(highlight_secondary)
    bg_mask = ~joined.index.isin(highlight_ids)

    fig = go.Figure()

    # Background cloud.
    bg = joined.loc[bg_mask]
    fig.add_trace(
        go.Scattergl(
            x=bg["_a"],
            y=bg["_b"],
            mode="markers",
            marker={
                "color": COLOR_BACKGROUND,
                "size": 4,
                "opacity": 0.6,
                "line": {"width": 0},
            },
            name="all genes",
            hovertemplate=(
                "%{customdata}<br>"
                f"{label_a}: %{{x:.2f}}<br>"
                f"{label_b}: %{{y:.2f}}<extra></extra>"
            ),
            customdata=bg.index,
        )
    )

    # Diagonal + zero lines. Compute the shared axis range first so
    # the diagonal covers the visible area exactly.
    x_vals = joined["_a"].to_numpy()
    y_vals = joined["_b"].to_numpy()
    lo = float(min(x_vals.min(), y_vals.min())) - 0.5
    hi = float(max(x_vals.max(), y_vals.max())) + 0.5

    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"color": COLOR_DIAGONAL, "width": 1, "dash": "dash"},
            name="y = x",
            hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=0,
        line={"color": COLOR_DIAGONAL, "width": 1, "dash": "dot"},
        opacity=0.4,
    )
    fig.add_vline(
        x=0,
        line={"color": COLOR_DIAGONAL, "width": 1, "dash": "dot"},
        opacity=0.4,
    )

    # Primary highlights.
    _add_cross_contrast_highlight(
        fig,
        joined=joined,
        highlight=highlight_primary,
        color=primary_color,
        name=primary_label,
        label_a=label_a,
        label_b=label_b,
    )
    # Secondary highlights.
    _add_cross_contrast_highlight(
        fig,
        joined=joined,
        highlight=highlight_secondary,
        color=secondary_color,
        name=secondary_label,
        label_a=label_a,
        label_b=label_b,
    )

    fig.update_layout(
        **_theme_with_title(font_size, title),
        xaxis={
            "title": f"log<sub>2</sub> FC — {label_a}",
            "range": [lo, hi],
            "zeroline": False,
            "showline": True,
            "linecolor": "#111111",
            "ticks": "outside",
        },
        yaxis={
            "title": f"log<sub>2</sub> FC — {label_b}",
            "range": [lo, hi],
            "zeroline": False,
            "showline": True,
            "linecolor": "#111111",
            "ticks": "outside",
            "scaleanchor": "x",  # lock aspect ratio so 1 x-unit == 1 y-unit
            "scaleratio": 1,
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


def _add_cross_contrast_highlight(
    fig: go.Figure,
    *,
    joined: pd.DataFrame,
    highlight: dict[str, str],
    color: str,
    name: str,
    label_a: str,
    label_b: str,
) -> None:
    """Internal helper — labeled scatter trace for one highlight set
    on the cross-contrast figure. Same shape as ``_add_highlight_trace``
    but with a two-axis hover template."""
    if not highlight:
        return
    hl = joined.loc[joined.index.intersection(list(highlight.keys()))]
    if hl.empty:
        return
    labels = [highlight[g] for g in hl.index]
    fig.add_trace(
        go.Scatter(
            x=hl["_a"],
            y=hl["_b"],
            mode="markers+text",
            marker={
                "color": color,
                "size": 12,
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
                f"{label_a}: %{{x:.2f}}<br>"
                f"{label_b}: %{{y:.2f}}<extra></extra>"
            ),
        )
    )


# ----------------------------------------------------------------------
# Expression heatmap
# ----------------------------------------------------------------------
def figure_export_bytes(
    fig: go.Figure,
    format: str,
    *,
    scale: int = 2,
) -> tuple[Optional[bytes], Optional[str]]:
    """Try to export ``fig`` as ``format`` bytes via kaleido.

    Returns a ``(bytes, error_message)`` pair. On success the bytes
    are populated and the error is ``None``; on failure the bytes
    are ``None`` and the error message is a human-readable string
    suitable for rendering to the user.

    Kaleido is the plotly backend for static image export
    (``fig.to_image``) and is NOT a hard dependency of this package
    — if it isn't installed, ``fig.to_image`` raises and this helper
    funnels that into a clean error string. Callers in the Streamlit
    Figures tab surface the error as an inline caption hint so the
    user knows which ``pip install`` / ``mamba install`` command to
    run. Callers in tests or CLI wrappers can branch on the presence
    of the error string.

    Extracted from the Streamlit ``_figure_download_row`` helper in
    ``app.py`` so the decision logic (success vs. caption fallback)
    can be unit-tested without a Streamlit runtime.
    """
    try:
        img_bytes = fig.to_image(format=format, scale=scale)
    except Exception as exc:
        return None, str(exc)
    return img_bytes, None


def _unique_preserve(items: Iterable) -> list:
    """Return unique items in first-seen order. Pure-Python stand-in
    for ``dict.fromkeys(...)`` when we want a list back. Used by
    :func:`expression_heatmap` to derive the default group order
    from a record list."""
    seen: set = set()
    out: list = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score with zero-std rows left at zero (not NaN).

    Uses the sample standard deviation (``ddof=1``), matching the
    convention used by R's ``scale()`` and scipy's default — so the
    heatmap numbers line up with what a reviewer would reproduce in
    R. For ``n=18`` samples the difference from ``ddof=0`` is only
    ``sqrt(18/17) ≈ 1.029`` (physically invisible on a colour scale),
    but matching convention means someone recomputing the z-scores in
    R gets the same values.

    Rows whose std is zero (a gene expressed identically across every
    sample) would otherwise produce NaN, which plotly renders as
    white and breaks the colour scale. Replace them with zero so
    they render as a neutral stripe.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1)
    # Avoid division by zero; zero-std rows stay at zero.
    std_safe = std.replace(0, np.nan)
    z = df.sub(mean, axis=0).div(std_safe, axis=0)
    return z.fillna(0.0)


def expression_heatmap(
    fpkm: pd.DataFrame,
    records: Iterable,
    *,
    title: str,
    gene_labels: dict[str, str],
    font_size: int = DEFAULT_FONT_SIZE,
    normalize: str = "zscore",
    group_order: Optional[list[str]] = None,
    fraction_order: Optional[list[str]] = None,
) -> go.Figure:
    """Expression heatmap of ``gene_labels`` genes × samples, grouped.

    Column order is ``(group, fraction, replicate)``: every sample
    belonging to a given ``group × fraction`` block sits contiguously,
    with a single-column visual gap between blocks so the reader sees
    the structure without squinting. The gap is implemented as a NaN
    column (plotly renders NaN as white) — simpler than stitching
    together multiple subplot heatmaps and produces the same visual.

    ``records`` is duck-typed to :class:`phosphotrap.samples.SampleRecord`:
    each entry must expose ``.name()``, ``.group``, ``.fraction``, and
    ``.replicate``. Tests pass simple namedtuples with the same shape
    so this module doesn't force a concrete samples.py import on the
    test suite.

    ``normalize``:

    * ``"zscore"`` (default) — row-wise z-score, RdBu_r diverging
      colour scale centered at zero. The right call when the question
      is "which samples deviate from the row mean" — i.e. the
      biological condition effect within each gene.
    * ``"log2"`` — ``log2(fpkm + 1)`` with Viridis. The right call
      when the question is "how much is this gene expressed overall".
    * ``"raw"`` — FPKM values as-is with Viridis. For inspection;
      skews toward a few high-expression genes dominating the scale.

    Returns a placeholder figure with a centered annotation when no
    requested gene appears in ``fpkm``.
    """
    gene_ids = [g for g in gene_labels if g in fpkm.index]
    if not gene_ids:
        fig = go.Figure()
        fig.update_layout(
            **_theme_with_title(font_size, title),
            annotations=[
                {
                    "text": "No expression data for the requested genes.",
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

    # Build the ordered column list from the records. We don't rely
    # on ``fpkm`` column ordering because it may be whatever the
    # pipeline happened to load.
    records_list = list(records)
    groups = list(group_order) if group_order else _unique_preserve(
        r.group for r in records_list
    )
    fractions = list(fraction_order) if fraction_order else ["IP", "INPUT"]

    ordered_columns: list[str] = []
    column_labels: list[str] = []
    blocks: list[tuple[str, str, list[str]]] = []

    for group in groups:
        for fraction in fractions:
            block_recs = sorted(
                (
                    r for r in records_list
                    if r.group == group
                    and r.fraction == fraction
                    and r.name() in fpkm.columns
                ),
                key=lambda r: r.replicate,
            )
            block_cols = [r.name() for r in block_recs]
            if not block_cols:
                continue
            if ordered_columns:
                # Visual gap: one NaN column between blocks. The
                # column name is a unique sentinel so plotly doesn't
                # deduplicate adjacent gaps.
                gap_name = f"__gap_{len(ordered_columns)}__"
                ordered_columns.append(gap_name)
                column_labels.append("")
            ordered_columns.extend(block_cols)
            # Short-form column labels like "IP1", "INPUT5" — the
            # group is implied by the block's position in the
            # figure-level annotations below.
            column_labels.extend(
                f"{r.fraction}{r.replicate}" for r in block_recs
            )
            blocks.append((group, fraction, block_cols))

    if not ordered_columns:
        fig = go.Figure()
        fig.update_layout(
            **_theme_with_title(font_size, title),
            annotations=[
                {
                    "text": "No samples matched the expected groups/fractions.",
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

    # Build the value matrix in gene × column order.
    gene_symbols = [gene_labels[g] for g in gene_ids]
    value_frame = pd.DataFrame(
        index=gene_ids, columns=ordered_columns, dtype=float
    )
    for col in ordered_columns:
        if col.startswith("__gap_"):
            value_frame[col] = np.nan
        else:
            value_frame[col] = fpkm.loc[gene_ids, col].astype(float).values

    if normalize == "zscore":
        real_cols = [c for c in ordered_columns if not c.startswith("__gap_")]
        real = value_frame[real_cols]
        z = _zscore_rows(real)
        # Reinsert gap columns with NaN so column positions stay put.
        plot_matrix = value_frame.copy()
        plot_matrix[real_cols] = z.values
        colorscale = "RdBu_r"
        colorbar_title = "z-score"
        zmid: Optional[float] = 0.0
    elif normalize == "log2":
        plot_matrix = np.log2(value_frame + 1.0)
        colorscale = "Viridis"
        colorbar_title = "log<sub>2</sub>(FPKM+1)"
        zmid = None
    else:
        plot_matrix = value_frame
        colorscale = "Viridis"
        colorbar_title = "FPKM"
        zmid = None

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=plot_matrix.values,
            x=column_labels,
            y=gene_symbols,
            colorscale=colorscale,
            zmid=zmid,
            colorbar={
                "title": {"text": colorbar_title, "side": "right"},
                "thickness": 14,
                "len": 0.7,
            },
            hovertemplate=(
                "%{y}<br>%{x}<br>"
                f"{colorbar_title}: %{{z:.2f}}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        )
    )

    # Block header annotations. One text label per (group, fraction)
    # block placed above its centre column, then a thicker group
    # label spanning both IP and INPUT blocks below them.
    # Implemented via paper-relative annotations because plotly
    # doesn't natively support hierarchical x-axis labels.
    annotations = list(fig.layout.annotations or [])
    block_centres: dict[str, list[float]] = {}
    running = 0
    for block_idx, (group, fraction, block_cols) in enumerate(blocks):
        # Skip gap column before every block except the first.
        if block_idx > 0:
            running += 1
        start = running
        running += len(block_cols)
        centre_idx = (start + running - 1) / 2
        # Translate column index into a paper-relative x position.
        centre_frac = (centre_idx + 0.5) / len(ordered_columns)
        annotations.append(
            {
                "text": fraction,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": centre_frac,
                "y": 1.02,
                "font": {
                    "family": NATURE_FONT_FAMILY,
                    "size": font_size,
                    "color": "#111111",
                },
            }
        )
        block_centres.setdefault(group, []).append(centre_frac)

    for group, centres in block_centres.items():
        annotations.append(
            {
                "text": f"<b>{group}</b>",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": sum(centres) / len(centres),
                "y": 1.08,
                "font": {
                    "family": NATURE_FONT_FAMILY,
                    "size": font_size + 1,
                    "color": "#111111",
                },
            }
        )

    # Heatmap needs a taller top margin for the two-row block headers.
    # Merge the override into the theme dict rather than passing a
    # separate ``margin=`` kwarg (which would collide with the
    # default margin in ``nature_theme``).
    theme = _theme_with_title(font_size, title)
    theme["margin"] = {"l": 90, "r": 60, "t": 80, "b": 90}
    fig.update_layout(
        **theme,
        xaxis={
            "tickangle": -45,
            "showline": True,
            "linecolor": "#111111",
            "ticks": "outside",
        },
        yaxis={
            "autorange": "reversed",  # first gene at the top, like a table
            "showline": True,
            "linecolor": "#111111",
            "ticks": "outside",
        },
        height=max(60 * len(gene_ids) + 160, 260),
        annotations=annotations,
    )
    return fig


# ----------------------------------------------------------------------
# Per-gene log2(IP/Input) strip plot
# ----------------------------------------------------------------------
def per_gene_strip(
    ratios: pd.DataFrame,
    pair_labels: dict[str, list[str]],
    *,
    title: str,
    gene_labels: dict[str, str],
    font_size: int = DEFAULT_FONT_SIZE,
    group_order: Optional[list[str]] = None,
    primary_color: str = COLOR_GALANIN,
    secondary_color: str = COLOR_CUSTOM,
    primary_ids: Optional[set[str]] = None,
) -> go.Figure:
    """Per-gene strip plot of log2(IP/Input) ratios grouped by diet.

    For each gene in ``gene_labels`` we render one subplot with:

    * **Individual dots** — one per animal (3 per group for this
      design), placed on a categorical x axis by group.
    * **Group means** — a short horizontal bar drawn on top of the
      dots. Makes the effect direction visible at a glance even when
      the n-per-group is small enough that the dots don't visually
      cluster.

    The dot colour is ``primary_color`` when the gene is in
    ``primary_ids`` (galanin core) and ``secondary_color`` otherwise.
    Callers typically pass ``primary_ids = set(galanin_resolved)``.

    ``ratios`` is ``RatioResult.ratios`` — a (gene_id × pair_label)
    dataframe where each column is a single animal's
    ``log2(IP_FPKM / Input_FPKM)``. ``pair_labels`` is
    ``RatioResult.pair_labels``: a mapping from group name to the list
    of pair-label column names for that group (e.g.
    ``{"NCD": ["NCD_rep1", "NCD_rep3", "NCD_rep4"], ...}``).

    Returns a placeholder figure with a centered annotation when the
    inputs don't contain a single plottable gene × group combination,
    so the Figures tab can render the panel slot before the user has
    loaded salmon output.
    """
    from plotly.subplots import make_subplots

    gene_ids = [g for g in gene_labels if g in ratios.index]
    if not gene_ids or not pair_labels:
        fig = go.Figure()
        fig.update_layout(
            **_theme_with_title(font_size, title),
            annotations=[
                {
                    "text": "No ratio data for the requested genes.",
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

    groups = list(group_order) if group_order else list(pair_labels.keys())
    primary_ids = primary_ids or set()

    # Subplot grid — up to 4 columns, rows as needed. Keeps each
    # panel large enough for individual animals to be distinguishable
    # without overflowing the page width.
    n = len(gene_ids)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[gene_labels[g] for g in gene_ids],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
        shared_yaxes=False,
    )

    # Collect all y values up front so every subplot shares the same
    # y range — lets the reader compare effect sizes across panels
    # without misreading an autoscaled axis.
    all_y: list[float] = []

    for idx, gene_id in enumerate(gene_ids):
        row = idx // cols + 1
        col = idx % cols + 1
        color = primary_color if gene_id in primary_ids else secondary_color
        show_legend = idx == 0  # one legend entry per colour group

        per_group_means: list[tuple[str, float]] = []

        for group in groups:
            labels = pair_labels.get(group, [])
            if not labels:
                continue
            values = [
                float(ratios.at[gene_id, lab])
                for lab in labels
                if lab in ratios.columns
            ]
            if not values:
                continue
            all_y.extend(values)

            # Dots: one per animal.
            fig.add_trace(
                go.Scatter(
                    x=[group] * len(values),
                    y=values,
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 9,
                        "line": {"color": "#111111", "width": 0.8},
                        "opacity": 0.95,
                    },
                    name="galanin" if gene_id in primary_ids else "custom",
                    legendgroup="galanin" if gene_id in primary_ids else "custom",
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{gene_labels[gene_id]}</b><br>"
                        f"{group}<br>"
                        "log<sub>2</sub>(IP/Input): %{y:.2f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            per_group_means.append((group, float(np.mean(values))))

        # Group-mean bars.
        if per_group_means:
            fig.add_trace(
                go.Scatter(
                    x=[g for g, _ in per_group_means],
                    y=[m for _, m in per_group_means],
                    mode="markers",
                    marker={
                        "symbol": "line-ew",
                        "size": 26,
                        "color": "#111111",
                        "line": {"color": "#111111", "width": 3},
                    },
                    name="group mean",
                    legendgroup="mean",
                    showlegend=idx == 0,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    # Shared y range with a little head/tail padding, plus a zero line
    # for every subplot so the direction of change is unambiguous.
    if all_y:
        y_min = min(all_y) - 0.3
        y_max = max(all_y) + 0.3
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                fig.update_yaxes(
                    range=[y_min, y_max],
                    row=r,
                    col=c,
                    zeroline=True,
                    zerolinecolor=COLOR_DIAGONAL,
                    zerolinewidth=1,
                    showline=True,
                    linecolor="#111111",
                    ticks="outside",
                )
                fig.update_xaxes(
                    row=r,
                    col=c,
                    categoryorder="array",
                    categoryarray=groups,
                    showline=True,
                    linecolor="#111111",
                    ticks="outside",
                )

    # One shared y-axis label, placed via layout annotation so it
    # spans the whole figure rather than repeating per-subplot.
    fig.update_layout(
        **_theme_with_title(font_size, title),
        height=max(260 * rows, 320),
        showlegend=True,
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#d1d5db",
            "borderwidth": 1,
        },
    )
    fig.add_annotation(
        text="log<sub>2</sub>(IP / Input)",
        xref="paper",
        yref="paper",
        x=-0.06,
        y=0.5,
        showarrow=False,
        textangle=-90,
        font={"family": NATURE_FONT_FAMILY, "size": font_size + 1},
    )

    return fig


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
    primary_label: str = "Galanin signaling",
    secondary_label: str = "Custom highlights",
) -> go.Figure:
    """Build a Nature-grade volcano plot with an explicit highlight layer.

    The x axis is ``delta_col`` (default ``delta_log2``, the mean alt-
    minus-mean-ref log2 ratio); the y axis is ``-log10(p_col)`` so
    plot orientation matches every volcano a reviewer has ever seen.

    **Mann-Whitney banding at small n is honest, not a bug.** With
    the default ``p_col="mannwhitney_p"`` and a 3-vs-3 comparison
    (the typical design for this app), the two-sided Mann-Whitney
    U statistic has only C(6,3)=20 possible rank orderings, which
    produces exactly five distinct p-values (≈0.10, 0.20, 0.40,
    0.70, 1.00 — the smallest two-sided p is 2/20 = 0.10). That's
    a property of the exact rank-sum distribution, not precision
    loss: p-values are stored as float64 end-to-end, no rounding,
    no permutation approximation. On ``-log10(p)`` the banding
    shows up as a staircase of horizontal stripes, and that's what
    the pipeline draws — we do *not* apply visual jitter, because
    moving points off their true p-value would misrepresent the
    statistic. If you want a continuous y-axis at small n, switch
    to a parametric, variance-borrowed test: this app runs
    ``anota2seq`` exactly for that purpose. Pass the anota2seq
    translation regMode table to ``volcano_plot`` with
    ``p_col="apvRvmP"``, ``padj_col="apvRvmPAdj"`` and
    ``delta_col="apvEff"`` (or ``"deltaPT"``) to get a
    volcano whose y-axis is not distribution-capped.

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
    produced by :class:`phosphotrap.fpkm.ContrastResult.table` or
    by the ``anota2seq`` runner's regMode TSVs), so the highlight
    dicts can be keyed directly by gene_id.

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

    # Layer 3: primary highlights.
    _add_highlight_trace(
        fig,
        df=df,
        highlight=highlight_primary,
        color=primary_color,
        name=primary_label,
        p_col=p_col,
        delta_col=delta_col,
    )

    # Layer 4: secondary highlights.
    _add_highlight_trace(
        fig,
        df=df,
        highlight=highlight_secondary,
        color=secondary_color,
        name=secondary_label,
        p_col=p_col,
        delta_col=delta_col,
    )

    # Reference lines: y = -log10(alpha) (only when alpha > 0; a
    # non-positive threshold has no defined -log10, and passing
    # ``y=None`` to ``add_hline`` raises ValueError) and x = 0.
    if alpha > 0:
        fig.add_hline(
            y=-np.log10(alpha),
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
