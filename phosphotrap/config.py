"""Persistent configuration for the phosphoribotrap Streamlit app.

``AppConfig`` is a dataclass that round-trips to ``config/phosphotrap.json``.
``load()`` coerces each field's type using :func:`dataclasses.fields` so that
a malformed JSON file (wrong types, missing keys, garbage ``contrasts``
list) falls through to defaults instead of crashing downstream Streamlit
widgets.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

DEFAULT_CONFIG_PATH = Path("config/phosphotrap.json")

DEFAULT_REFERENCE_GROUP = "NCD"
DEFAULT_CONTRASTS = ["HSD1_vs_NCD", "HSD3_vs_NCD"]
# Full enumeration of contrasts the app supports (pre-filtered later
# by the configured reference group).
ALL_CONTRASTS = ["HSD1_vs_NCD", "HSD3_vs_NCD", "HSD3_vs_HSD1"]

# Defaults for the path fields that flow into the filesystem. Declared
# at module level so the dataclass defaults and the ``effective_*``
# helper methods (which coerce blank user input) share a single
# source of truth.
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_DIR = "output/reports"
DEFAULT_RSCRIPT = "Rscript"

# Strings that should be coerced to the dataclass default rather than
# used literally. ``""`` and ``"."`` both resolve to the cwd via
# :class:`pathlib.Path`, which would silently scatter ``logs/`` /
# ``output/salmon/`` into whatever directory streamlit was launched
# from. Users who genuinely want cwd can pass an absolute path.
_BLANK_PATH_SENTINELS = {"", "."}


def _coerce_blank_path(value: str, default: str) -> str:
    """Return ``default`` when ``value`` is blank or ``"."``, else ``value``."""
    if value is None:
        return default
    stripped = str(value).strip()
    if stripped in _BLANK_PATH_SENTINELS:
        return default
    return stripped

# Safe-token regex used to validate any user-editable string that ends
# up in a filesystem path. ``reference_group`` flows into the scratch
# dir name for anota2seq/DESeq2, so a JSON-edited config with
# ``"../evil"`` would escape the output directory. ``samples.py`` uses
# the same regex for the data-editor's group/fraction/ccg_id cells, so
# this constant is the single source of truth.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
# A valid contrast is two safe tokens joined by ``_vs_``.
_CONTRAST_RE = re.compile(r"^[A-Za-z0-9_\-]+_vs_[A-Za-z0-9_\-]+$")


def is_safe_token(value: str) -> bool:
    return bool(value) and _SAFE_TOKEN_RE.fullmatch(value) is not None


def is_safe_contrast(value: str) -> bool:
    return bool(value) and _CONTRAST_RE.fullmatch(value) is not None


def resolve_rscript(cfg: "AppConfig") -> str:
    """Return the Rscript binary to invoke, falling back to PATH lookup.

    The Config tab lets the user point at a specific ``Rscript`` — e.g.
    one inside a conda env. An empty string means "use whatever's on
    PATH". This helper centralises the ``cfg.rscript_path or "Rscript"``
    idiom that was duplicated across the pipeline / anota2seq / DESeq2
    runners.
    """
    return cfg.rscript_path or "Rscript"


def contrasts_for_reference(ref_group: str, all_groups: tuple[str, ...]) -> list[str]:
    """Return the canonical contrast list for a given reference group.

    For reference ``NCD`` and groups ``(NCD, HSD1, HSD3)`` this returns
    ``["HSD1_vs_NCD", "HSD3_vs_NCD", "HSD3_vs_HSD1"]`` — every non-ref
    group compared to the reference, plus the non-ref ↔ non-ref contrast
    as an optional extra. Ordering is deterministic.
    """
    others = [g for g in all_groups if g != ref_group]
    contrasts = [f"{g}_vs_{ref_group}" for g in others]
    if len(others) >= 2:
        contrasts.append(f"{others[1]}_vs_{others[0]}")
    return contrasts


def validate_fastq_dir(fastq_dir: str) -> list[str]:
    """Return a list of human-readable errors for the fastq directory.

    Catches the HIGH #4 footgun where the Samples tab's "Auto-populate
    fastq paths" button silently returned "0 / 18 ready samples" when
    ``cfg.fastq_dir`` was a typo or pointed at a non-existent path.
    Before this helper, the only check was ``if not cfg.fastq_dir:``
    (emptiness), so a bad path was indistinguishable from "no files
    match the vendor regex" in the UI.

    An empty list means "looks OK". Empty ``fastq_dir`` is treated as
    "unconfigured" rather than "error" because the user may genuinely
    not have set it yet on a fresh install — callers that need
    non-empty should gate on that separately.
    """
    errs: list[str] = []

    if not fastq_dir:
        # Empty is "unconfigured", not "invalid" — callers decide whether
        # to hard-error or just nudge the user.
        return errs

    p = Path(fastq_dir).expanduser()
    if not p.exists():
        errs.append(f"Fastq directory does not exist: {p}")
        return errs
    if not p.is_dir():
        errs.append(f"Fastq directory path is a file, not a directory: {p}")
        return errs

    # Directory exists. Sanity-check that it contains *something*
    # matching the .fastq.gz convention. An empty directory is almost
    # always a user mistake — either the wrong path, or the files
    # haven't been copied over yet.
    try:
        has_any = any(p.glob("*.fastq.gz"))
    except OSError as exc:  # pragma: no cover - defensive
        errs.append(f"Could not list fastq directory {p}: {exc}")
        return errs
    if not has_any:
        errs.append(
            f"Fastq directory {p} contains no *.fastq.gz files. "
            f"Double-check the path, or copy the raw reads into this "
            f"directory before running the pipeline."
        )
    return errs


def validate_reference_paths(
    salmon_index: str, tx2gene_tsv: str
) -> list[str]:
    """Return a list of human-readable errors for the two reference paths.

    Catches the two footguns that v5029e4d's Reference tab was not
    defensive enough to prevent:

    1. ``salmon_index`` pointing at a file or a non-index directory.
    2. ``tx2gene_tsv`` pointing at a directory (e.g. the Reference
       tab's destination directory) instead of the ``tx2gene.tsv``
       file inside it.

    An empty list means "looks OK". Callers should surface the errors
    via ``st.error`` and refuse to proceed.
    """
    errs: list[str] = []

    if not salmon_index:
        errs.append("Salmon index path is empty.")
    else:
        p = Path(salmon_index).expanduser()
        if not p.exists():
            errs.append(f"Salmon index path does not exist: {p}")
        elif not p.is_dir():
            errs.append(
                f"Salmon index must be a directory, not a file: {p}"
            )
        elif not (p / "info.json").exists():
            errs.append(
                f"Salmon index directory {p} has no info.json — "
                f"this does not look like a built salmon index. "
                f"Rebuild via the Reference tab."
            )

    if not tx2gene_tsv:
        errs.append("tx2gene TSV path is empty.")
    else:
        t = Path(tx2gene_tsv).expanduser()
        if not t.exists():
            errs.append(f"tx2gene TSV path does not exist: {t}")
        elif t.is_dir():
            errs.append(
                f"tx2gene TSV must be a FILE (tx2gene.tsv), not a "
                f"directory: {t}. On the Reference tab, click "
                f"'Use these paths in Config' to auto-populate the "
                f"correct file path."
            )
        elif t.stat().st_size == 0:
            errs.append(f"tx2gene TSV is empty: {t}")

    return errs


def reconcile_contrasts(current: list[str], available: list[str]) -> list[str]:
    """Filter ``current`` down to entries still present in ``available``.

    Used to reconcile stale widget session state after the user changes
    the reference group. Preserves the user's original ordering for
    whatever values survive; returns an empty list if nothing does.
    """
    if not current:
        return []
    return [c for c in current if c in available]

# anota2seq filtering thresholds. The p-value and slope bounds mirror
# the package defaults published in ``anota2seqRun.R`` (Oertlin et al.
# 2019, NAR). ``selDeltaPT`` / ``selDeltaTP`` are deliberately looser
# than the package's ``log2(1.2) ≈ 0.263`` — at n=3 per group we accept
# smaller effect sizes for discovery and rely on orthogonal validation
# for confidence.
DEFAULT_DELTA_PT = 0.1
DEFAULT_DELTA_TP = 0.1
DEFAULT_MAX_PADJ = 0.15
DEFAULT_MIN_SLOPE_TRANS = -1.0
DEFAULT_MAX_SLOPE_TRANS = 2.0
DEFAULT_MIN_SLOPE_BUFF = -2.0
DEFAULT_MAX_SLOPE_BUFF = 1.0

# FPKM floor for IP/Input ratios.
DEFAULT_MIN_FPKM = 0.1

# DESeq2 small-n recovery defaults.
#   min_count_filter — drop genes with rowSums(counts) below this before
#     calling DESeq(); standard DESeq2 vignette practice. 0 disables.
#   use_ihw         — IHW p-value weighting (Ignatiadis et al. 2016) on
#     the Wald p-values, weighted by ``baseMean``. Adds a ``padj_ihw``
#     column. Falls back gracefully when the IHW package is missing.
#   use_apeglm      — apeglm LFC shrinkage with s-values (Zhu et al.
#     2018 / Stephens 2017). Adds ``log2FoldChange_apeglm``,
#     ``lfcSE_apeglm``, and ``svalue`` columns. Falls back gracefully
#     when the apeglm package is missing.
DEFAULT_DESEQ2_MIN_COUNT_FILTER = 10
DEFAULT_DESEQ2_USE_IHW = True
DEFAULT_DESEQ2_USE_APEGLM = True

# Targeted analysis filter — restricts the multiple-testing universe
# of every analysis-tab table to a pre-specified gene set declared a
# priori (typically a GO term that matches the experiment's biological
# question; e.g. GO:0007218 "neuropeptide signaling pathway" for a
# phosphoTRAP study targeting neuronal-marker discovery). When
# enabled, each table's BH-FDR padj is recomputed on the subset; the
# original padj is preserved as ``padj_full``. This is reviewer-
# defensible only when the gene set is pre-registered — declare it
# explicitly in the methods.
DEFAULT_TARGET_FILTER_ENABLED = False
DEFAULT_TARGET_FILTER_TEXT = ""
DEFAULT_TARGET_FILTER_GO_ID = "GO:0007218"
DEFAULT_TARGET_FILTER_GO_TAXON = "10090"


@dataclass
class AppConfig:
    # Paths
    fastq_dir: str = ""
    salmon_index: str = ""
    tx2gene_tsv: str = ""
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_dir: str = DEFAULT_REPORT_DIR
    rscript_path: str = DEFAULT_RSCRIPT

    # Runtime
    threads: int = 8
    run_fastp: bool = True
    force_rerun: bool = False
    salmon_libtype: str = "A"

    # Design
    reference_group: str = DEFAULT_REFERENCE_GROUP
    contrasts: list[str] = field(default_factory=lambda: list(DEFAULT_CONTRASTS))

    # anota2seq thresholds
    anota_delta_pt: float = DEFAULT_DELTA_PT
    anota_delta_tp: float = DEFAULT_DELTA_TP
    anota_max_padj: float = DEFAULT_MAX_PADJ
    anota_min_slope_trans: float = DEFAULT_MIN_SLOPE_TRANS
    anota_max_slope_trans: float = DEFAULT_MAX_SLOPE_TRANS
    anota_min_slope_buff: float = DEFAULT_MIN_SLOPE_BUFF
    anota_max_slope_buff: float = DEFAULT_MAX_SLOPE_BUFF

    # Sign-consistency / Mann-Whitney defaults
    min_fpkm: float = DEFAULT_MIN_FPKM
    sign_consistency_min: int = 3

    # DESeq2 small-n recovery toggles
    deseq2_min_count_filter: int = DEFAULT_DESEQ2_MIN_COUNT_FILTER
    deseq2_use_ihw: bool = DEFAULT_DESEQ2_USE_IHW
    deseq2_use_apeglm: bool = DEFAULT_DESEQ2_USE_APEGLM

    # Targeted analysis filter (a-priori gene set + BH on subset)
    target_filter_enabled: bool = DEFAULT_TARGET_FILTER_ENABLED
    target_filter_text: str = DEFAULT_TARGET_FILTER_TEXT
    target_filter_go_id: str = DEFAULT_TARGET_FILTER_GO_ID
    target_filter_go_taxon: str = DEFAULT_TARGET_FILTER_GO_TAXON

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
            if not isinstance(raw, dict):
                return cls()
        except (json.JSONDecodeError, OSError):
            return cls()

        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            value = raw[f.name]
            coerced = _coerce(value, f.type)
            if coerced is _INVALID:
                # Leave the field at its default.
                continue
            kwargs[f.name] = coerced

        # Domain validation for fields that flow into filesystem paths.
        # ``reference_group`` becomes part of the anota2seq scratch dir
        # name via ``contrasts_for_reference``, and each contrast
        # string becomes a directory on disk. Reject anything that
        # isn't a safe token / safe contrast string and fall back to
        # the field's default — do not silently propagate a
        # path-traversal-capable value downstream.
        ref = kwargs.get("reference_group")
        if ref is not None and not is_safe_token(ref):
            kwargs.pop("reference_group")
        contrasts = kwargs.get("contrasts")
        if contrasts is not None:
            clean = [c for c in contrasts if is_safe_contrast(c)]
            if clean:
                kwargs["contrasts"] = clean
            else:
                kwargs.pop("contrasts")

        try:
            return cls(**kwargs)
        except TypeError:
            return cls()

    def diff(self, other: "AppConfig") -> dict[str, tuple[Any, Any]]:
        """Return a dict of fields where self and other differ."""
        a = asdict(self)
        b = asdict(other)
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    # ------------------------------------------------------------------
    # Effective path getters
    #
    # The raw ``output_dir`` / ``report_dir`` fields are whatever the
    # user currently has in the Config tab's text inputs, including an
    # empty string (if they backspaced the field clear) or ``.`` (the
    # degenerate cwd alias). Every code path that flows these into a
    # filesystem path should go through these helpers instead of
    # ``Path(cfg.output_dir)`` directly, so a blank field doesn't
    # silently scatter ``output/salmon/`` or ``logs/`` into whatever
    # directory streamlit happened to be launched from.
    # ------------------------------------------------------------------
    def effective_output_dir(self) -> Path:
        return Path(_coerce_blank_path(self.output_dir, DEFAULT_OUTPUT_DIR))

    def effective_report_dir(self) -> Path:
        return Path(_coerce_blank_path(self.report_dir, DEFAULT_REPORT_DIR))


# Sentinel: the supplied value cannot be coerced to the target type.
_INVALID = object()


def _coerce(value: Any, target: Any) -> Any:
    """Best-effort coerce ``value`` into the dataclass field type."""
    # Dataclass ``fields()`` gives the *annotation*, which in modern
    # Python may still be a string if ``from __future__ import annotations``
    # is in effect. Handle both cases.
    if isinstance(target, str):
        return _coerce_from_str(value, target)

    origin = get_origin(target)
    # AppConfig only declares ``list[str]`` (the ``contrasts`` field),
    # so the list branch is the only one we exercise. The earlier
    # tuple-handling code was dead — kept the list branch tightened.
    if origin is list:
        if not isinstance(value, (list, tuple)):
            return _INVALID
        args = get_args(target)
        if args:
            inner = args[0]
            out = []
            for item in value:
                c = _coerce(item, inner)
                if c is _INVALID:
                    return _INVALID
                out.append(c)
            return out
        return list(value)

    if target is int:
        try:
            if isinstance(value, bool):
                return _INVALID
            return int(value)
        except (TypeError, ValueError):
            return _INVALID
    if target is float:
        try:
            if isinstance(value, bool):
                return _INVALID
            return float(value)
        except (TypeError, ValueError):
            return _INVALID
    if target is bool:
        if isinstance(value, bool):
            return value
        return _INVALID
    if target is str:
        if isinstance(value, str):
            return value
        return _INVALID
    return value


def _coerce_from_str(value: Any, target_name: str) -> Any:
    t = target_name.strip()
    if t == "int":
        return _coerce(value, int)
    if t == "float":
        return _coerce(value, float)
    if t == "bool":
        return _coerce(value, bool)
    if t == "str":
        return _coerce(value, str)
    if t in ("list[str]", "List[str]", "list", "typing.List[str]"):
        if not isinstance(value, list):
            return _INVALID
        out = []
        for item in value:
            if not isinstance(item, str):
                return _INVALID
            out.append(item)
        return out
    # Unknown annotation — accept as-is.
    return value
