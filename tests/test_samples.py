"""Smoke tests for the sample sheet / contrast pairing layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phosphotrap.samples import (
    default_sample_df,
    pairs,
    pairs_by_group,
    records_for_contrast,
    summary,
    to_records,
)


def test_default_sample_sheet_has_18_records_3_groups_3_pairs_each():
    df = default_sample_df()
    assert len(df) == 18
    recs = to_records(df)
    assert len(recs) == 18

    s = summary(recs)
    assert s["n_records"] == 18
    assert s["n_groups"] == 3
    assert s["n_pairs"] == 9  # 3 groups × 3 replicates

    by_group = pairs_by_group(recs)
    assert set(by_group.keys()) == {"NCD", "HSD1", "HSD3"}
    for g in ("NCD", "HSD1", "HSD3"):
        assert len(by_group[g]) == 3

    # Replicate ids 2 and 7 are absent by design.
    all_reps = {r.replicate for r in recs}
    assert 2 not in all_reps
    assert 7 not in all_reps

    # Every pair is IP + INPUT of the same (group, replicate).
    for p in pairs(recs):
        assert p.ip.fraction == "IP"
        assert p.input.fraction == "INPUT"
        assert p.ip.group == p.input.group == p.group
        assert p.ip.replicate == p.input.replicate == p.replicate


def test_records_for_contrast_returns_six_pairs_in_canonical_order():
    df = default_sample_df()
    recs = to_records(df)

    subset = records_for_contrast(recs, alt_group="HSD1", ref_group="NCD")
    assert len(subset) == 6  # 6 matched animal pairs (3 ref + 3 alt)

    # Reference group first, then alt, each sorted by replicate.
    groups = [p.group for p in subset]
    assert groups == ["NCD"] * 3 + ["HSD1"] * 3

    ncd_reps = [p.replicate for p in subset[:3]]
    hsd1_reps = [p.replicate for p in subset[3:]]
    assert ncd_reps == sorted(ncd_reps) == [1, 3, 4]
    assert hsd1_reps == sorted(hsd1_reps) == [5, 6, 8]

    # Every pair bundles one IP + one INPUT from the same animal.
    for p in subset:
        assert p.ip.fraction == "IP"
        assert p.input.fraction == "INPUT"
        assert p.ip.replicate == p.input.replicate == p.replicate
        assert p.ip.group == p.input.group == p.group


def test_to_records_is_nan_safe():
    df = default_sample_df().copy()
    # Inject NaN rows of the sort st.data_editor leaves behind.
    blank = {c: np.nan for c in df.columns}
    df.loc[len(df)] = blank
    df.loc[len(df)] = blank
    recs = to_records(df)
    assert len(recs) == 18  # blank rows dropped, original 18 preserved


def test_to_records_rejects_path_traversal_tokens():
    """Dangerous group/ccg_id values must not survive into SampleRecords."""
    import pandas as pd

    df = pd.DataFrame([
        {"ccg_id": "138011", "sample": "IP1", "comment": "",
         "replicate": 1, "group": "NCD", "fraction": "IP"},
        # Path-traversal attempt in group
        {"ccg_id": "999999", "sample": "IPX", "comment": "",
         "replicate": 5, "group": "../evil", "fraction": "IP"},
        # Slash in ccg_id
        {"ccg_id": "1380/11", "sample": "IPY", "comment": "",
         "replicate": 6, "group": "NCD", "fraction": "IP"},
        # Space + control char in group
        {"ccg_id": "138099", "sample": "IPZ", "comment": "",
         "replicate": 7, "group": "N CD", "fraction": "IP"},
    ])
    recs = to_records(df)
    # Only the first (clean) row survives.
    assert len(recs) == 1
    assert recs[0].ccg_id == "138011"
    assert recs[0].group == "NCD"


def test_discover_fastqs_warns_on_no_matches(tmp_path):
    """If files exist but none match the regex, the user should see a warning.

    The phosphotrap logger is non-propagating, so pytest's ``caplog``
    (which attaches at the root logger) never sees its records. Attach
    a capturing handler directly to the named logger instead.
    """
    import logging

    from phosphotrap.samples import discover_fastqs

    # Put files with the wrong naming convention.
    (tmp_path / "wrong_prefix_S1_L001_R1_001.fastq.gz").write_text("x")
    (tmp_path / "wrong_prefix_S1_L001_R2_001.fastq.gz").write_text("x")

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    log = logging.getLogger("phosphotrap")
    log.addHandler(handler)
    try:
        result = discover_fastqs(tmp_path)
    finally:
        log.removeHandler(handler)

    assert result == {}
    assert any("0 matched regex" in r.getMessage() for r in records)
