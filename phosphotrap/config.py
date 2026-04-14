"""Persistent configuration for the phosphoribotrap Streamlit app.

``AppConfig`` is a dataclass that round-trips to ``config/phosphotrap.json``.
``load()`` coerces each field's type using :func:`dataclasses.fields` so that
a malformed JSON file (wrong types, missing keys, garbage ``contrasts``
list) falls through to defaults instead of crashing downstream Streamlit
widgets.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

DEFAULT_CONFIG_PATH = Path("config/phosphotrap.json")

DEFAULT_REFERENCE_GROUP = "NCD"
DEFAULT_CONTRASTS = ["HSD1_vs_NCD", "HSD3_vs_NCD"]
ALL_CONTRASTS = ["HSD1_vs_NCD", "HSD3_vs_NCD", "HSD3_vs_HSD1"]

# anota2seq thresholds for a chronic stimulus — deliberately loose.
DEFAULT_DELTA_PT = 0.1
DEFAULT_DELTA_TP = 0.1
DEFAULT_MAX_PADJ = 0.1
DEFAULT_MIN_SLOPE_TRANS = 0.0
DEFAULT_MAX_SLOPE_TRANS = 2.0

# FPKM floor for IP/Input ratios.
DEFAULT_MIN_FPKM = 0.1


@dataclass
class AppConfig:
    # Paths
    fastq_dir: str = ""
    salmon_index: str = ""
    tx2gene_tsv: str = ""
    output_dir: str = "output"
    report_dir: str = "output/reports"
    rscript_path: str = "Rscript"

    # Runtime
    threads: int = 8
    run_fastp: bool = True
    force_rerun: bool = False
    salmon_libtype: str = "A"

    # Design
    reference_group: str = DEFAULT_REFERENCE_GROUP
    contrasts: list[str] = field(default_factory=lambda: list(DEFAULT_CONTRASTS))
    chronic_preset: bool = True

    # anota2seq thresholds
    anota_delta_pt: float = DEFAULT_DELTA_PT
    anota_delta_tp: float = DEFAULT_DELTA_TP
    anota_max_padj: float = DEFAULT_MAX_PADJ
    anota_min_slope_trans: float = DEFAULT_MIN_SLOPE_TRANS
    anota_max_slope_trans: float = DEFAULT_MAX_SLOPE_TRANS

    # Sign-consistency / Mann-Whitney defaults
    min_fpkm: float = DEFAULT_MIN_FPKM
    sign_consistency_min: int = 3

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
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()

    def diff(self, other: "AppConfig") -> dict[str, tuple[Any, Any]]:
        """Return a dict of fields where self and other differ."""
        a = asdict(self)
        b = asdict(other)
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}


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
    if origin in (list, tuple):
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
            return out if origin is list else tuple(out)
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
