"""Sample sheet handling for the 3-group phosphoribotrap design.

The default sheet is an 18-row table: 3 groups (NCD, HSD1, HSD3) × 3
biological replicates × 2 fractions (IP, INPUT). Every row carries a
``group`` and an ``replicate`` column so IP and INPUT can be paired by
animal within a group.

Replicate numbers are non-contiguous — IDs 2 and 7 were dropped from the
cohort — so do not assume ``replicate`` is a zero-based index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

GROUPS = ("NCD", "HSD1", "HSD3")
FRACTIONS = ("IP", "INPUT")

# Regex capturing the CCG id out of the vendor fastq filenames.
FASTQ_RE = re.compile(
    r"^A006200122_(?P<ccg>\d+)_S(?P<s>\d+)_L002_R(?P<r>[12])_001\.fastq\.gz$"
)


@dataclass
class SampleRecord:
    ccg_id: str
    sample: str
    comment: str
    replicate: int
    group: str
    fraction: str  # "IP" or "INPUT"
    fastq_r1: str = ""
    fastq_r2: str = ""

    def name(self) -> str:
        """Short identifier used in filenames and logs."""
        return f"{self.group}_{self.fraction}{self.replicate}"


# Default sample sheet baked into the app. Order mirrors the prompt.
DEFAULT_SAMPLE_ROWS: list[dict] = [
    # IP libraries
    {"ccg_id": "138011", "sample": "IP1",  "comment": "IP NCD1",   "replicate": 1,  "group": "NCD",  "fraction": "IP"},
    {"ccg_id": "138013", "sample": "IP3",  "comment": "IP NCD2",   "replicate": 3,  "group": "NCD",  "fraction": "IP"},
    {"ccg_id": "138015", "sample": "IP4",  "comment": "IP NCD3",   "replicate": 4,  "group": "NCD",  "fraction": "IP"},
    {"ccg_id": "138017", "sample": "IP5",  "comment": "IP HSD1-1", "replicate": 5,  "group": "HSD1", "fraction": "IP"},
    {"ccg_id": "138019", "sample": "IP6",  "comment": "IP HSD1-2", "replicate": 6,  "group": "HSD1", "fraction": "IP"},
    {"ccg_id": "138021", "sample": "IP8",  "comment": "IP HSD1-3", "replicate": 8,  "group": "HSD1", "fraction": "IP"},
    {"ccg_id": "138023", "sample": "IP9",  "comment": "IP HSD3-1", "replicate": 9,  "group": "HSD3", "fraction": "IP"},
    {"ccg_id": "138025", "sample": "IP10", "comment": "IP HSD3-2", "replicate": 10, "group": "HSD3", "fraction": "IP"},
    {"ccg_id": "138027", "sample": "IP11", "comment": "IP HSD3-3", "replicate": 11, "group": "HSD3", "fraction": "IP"},
    # INPUT libraries
    {"ccg_id": "138029", "sample": "INPUT1",  "comment": "Input NCD1",   "replicate": 1,  "group": "NCD",  "fraction": "INPUT"},
    {"ccg_id": "138031", "sample": "INPUT3",  "comment": "Input NCD2",   "replicate": 3,  "group": "NCD",  "fraction": "INPUT"},
    {"ccg_id": "138033", "sample": "INPUT4",  "comment": "Input NCD3",   "replicate": 4,  "group": "NCD",  "fraction": "INPUT"},
    {"ccg_id": "138035", "sample": "INPUT5",  "comment": "Input HSD1-1", "replicate": 5,  "group": "HSD1", "fraction": "INPUT"},
    {"ccg_id": "138037", "sample": "INPUT6",  "comment": "Input HSD1-2", "replicate": 6,  "group": "HSD1", "fraction": "INPUT"},
    {"ccg_id": "138039", "sample": "INPUT8",  "comment": "Input HSD1-3", "replicate": 8,  "group": "HSD1", "fraction": "INPUT"},
    {"ccg_id": "138041", "sample": "INPUT9",  "comment": "Input HSD3-1", "replicate": 9,  "group": "HSD3", "fraction": "INPUT"},
    {"ccg_id": "138043", "sample": "INPUT10", "comment": "Input HSD3-2", "replicate": 10, "group": "HSD3", "fraction": "INPUT"},
    {"ccg_id": "138045", "sample": "INPUT11", "comment": "Input HSD3-3", "replicate": 11, "group": "HSD3", "fraction": "INPUT"},
]


def default_sample_df() -> pd.DataFrame:
    df = pd.DataFrame(DEFAULT_SAMPLE_ROWS)
    df["fastq_r1"] = ""
    df["fastq_r2"] = ""
    return df


# ----------------------------------------------------------------------
# Conversion helpers
# ----------------------------------------------------------------------
def _clean_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def to_records(df: pd.DataFrame) -> list[SampleRecord]:
    """Convert a (possibly user-edited) dataframe to typed records.

    NaN-safe — blank rows or half-filled rows from ``st.data_editor`` are
    skipped without raising. Rows missing any of ``ccg_id``, ``group``,
    or ``replicate`` are dropped.
    """
    records: list[SampleRecord] = []
    for _, row in df.iterrows():
        ccg = _clean_cell(row.get("ccg_id"))
        group = _clean_cell(row.get("group"))
        frac = _clean_cell(row.get("fraction")).upper() or "IP"
        rep_raw = row.get("replicate")
        if not ccg or not group:
            continue
        try:
            if rep_raw is None or (isinstance(rep_raw, float) and pd.isna(rep_raw)):
                continue
            replicate = int(rep_raw)
        except (TypeError, ValueError):
            continue
        records.append(
            SampleRecord(
                ccg_id=ccg,
                sample=_clean_cell(row.get("sample")),
                comment=_clean_cell(row.get("comment")),
                replicate=replicate,
                group=group,
                fraction=frac if frac in FRACTIONS else "IP",
                fastq_r1=_clean_cell(row.get("fastq_r1")),
                fastq_r2=_clean_cell(row.get("fastq_r2")),
            )
        )
    return records


def records_to_df(records: Iterable[SampleRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


# ----------------------------------------------------------------------
# Fastq discovery & pairing
# ----------------------------------------------------------------------
def discover_fastqs(directory: Path) -> dict[str, dict[str, Path]]:
    """Return ``{ccg_id: {'R1': path, 'R2': path}}`` for the given directory."""
    directory = Path(directory)
    out: dict[str, dict[str, Path]] = {}
    if not directory.exists():
        return out
    for p in sorted(directory.glob("*.fastq.gz")):
        m = FASTQ_RE.match(p.name)
        if not m:
            continue
        ccg = m.group("ccg")
        read = "R" + m.group("r")
        out.setdefault(ccg, {})[read] = p
    return out


def populate_fastq_paths(
    records: list[SampleRecord], directory: Path
) -> list[SampleRecord]:
    mapping = discover_fastqs(directory)
    for r in records:
        entry = mapping.get(r.ccg_id)
        if not entry:
            continue
        if "R1" in entry:
            r.fastq_r1 = str(entry["R1"])
        if "R2" in entry:
            r.fastq_r2 = str(entry["R2"])
    return records


def ready_records(records: Iterable[SampleRecord]) -> list[SampleRecord]:
    return [
        r
        for r in records
        if r.fastq_r1
        and r.fastq_r2
        and Path(r.fastq_r1).exists()
        and Path(r.fastq_r2).exists()
    ]


# ----------------------------------------------------------------------
# Pair resolution — IP ↔ INPUT matched by replicate within a group
# ----------------------------------------------------------------------
@dataclass
class Pair:
    group: str
    replicate: int
    ip: SampleRecord
    input: SampleRecord


def pairs(records: Iterable[SampleRecord]) -> list[Pair]:
    by_key: dict[tuple[str, int, str], SampleRecord] = {}
    for r in records:
        by_key[(r.group, r.replicate, r.fraction)] = r
    out: list[Pair] = []
    seen: set[tuple[str, int]] = set()
    for (grp, rep, frac), rec in by_key.items():
        if frac != "IP":
            continue
        key = (grp, rep)
        if key in seen:
            continue
        inp = by_key.get((grp, rep, "INPUT"))
        if inp is None:
            continue
        out.append(Pair(group=grp, replicate=rep, ip=rec, input=inp))
        seen.add(key)
    out.sort(key=lambda p: (p.group, p.replicate))
    return out


def pairs_by_group(records: Iterable[SampleRecord]) -> dict[str, list[Pair]]:
    out: dict[str, list[Pair]] = {}
    for p in pairs(records):
        out.setdefault(p.group, []).append(p)
    return out


def records_for_contrast(
    records: Iterable[SampleRecord], alt_group: str, ref_group: str
) -> list[Pair]:
    """Return exactly the 6 matched animal pairs involved in a contrast.

    Each :class:`Pair` bundles the IP and INPUT libraries from one
    animal, so 3 ref + 3 alt pairs = 6 ``Pair`` objects (12 libraries).
    Order is deterministic: reference group first sorted by replicate,
    then alt group.
    """
    by_group = pairs_by_group(records)
    out: list[Pair] = []
    for grp in (ref_group, alt_group):
        grp_pairs = sorted(by_group.get(grp, []), key=lambda p: p.replicate)
        out.extend(grp_pairs)
    return out


def summary(records: Iterable[SampleRecord]) -> dict[str, int]:
    recs = list(records)
    pr = pairs(recs)
    groups_seen = sorted({r.group for r in recs})
    return {
        "n_records": len(recs),
        "n_groups": len(groups_seen),
        "n_pairs": len(pr),
        **{f"n_pairs_{g}": sum(1 for p in pr if p.group == g) for g in groups_seen},
    }
