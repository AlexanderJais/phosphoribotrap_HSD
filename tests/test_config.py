"""Smoke tests for AppConfig load/save/diff behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from phosphotrap.config import (
    DEFAULT_CONTRASTS,
    AppConfig,
    contrasts_for_reference,
    reconcile_contrasts,
)


def test_load_bad_types_falls_through_to_defaults(tmp_path: Path):
    p = tmp_path / "phosphotrap.json"
    bad = {
        "threads": "not-an-int",
        "anota_delta_pt": "nope",
        "contrasts": ["HSD1_vs_NCD", 42],  # malformed — second element not str
        "reference_group": "NCD",
        "run_fastp": "yes",  # non-bool
    }
    p.write_text(json.dumps(bad))

    cfg = AppConfig.load(p)
    # The malformed fields fell through to defaults.
    assert cfg.threads == AppConfig().threads
    assert cfg.anota_delta_pt == AppConfig().anota_delta_pt
    assert cfg.contrasts == list(DEFAULT_CONTRASTS)
    assert cfg.run_fastp == AppConfig().run_fastp
    # The one well-typed field is honoured.
    assert cfg.reference_group == "NCD"


def test_load_malformed_json_falls_through(tmp_path: Path):
    p = tmp_path / "phosphotrap.json"
    p.write_text("not-json{{{")
    cfg = AppConfig.load(p)
    assert cfg.contrasts == list(DEFAULT_CONTRASTS)


def test_save_and_roundtrip(tmp_path: Path):
    p = tmp_path / "phosphotrap.json"
    cfg = AppConfig(threads=12, anota_delta_pt=0.15, reference_group="NCD")
    cfg.save(p)
    loaded = AppConfig.load(p)
    assert loaded.threads == 12
    assert loaded.anota_delta_pt == 0.15


def test_diff_reports_only_changed_fields():
    a = AppConfig()
    b = AppConfig()
    assert a.diff(b) == {}
    b.threads = 99
    d = a.diff(b)
    assert "threads" in d and d["threads"] == (a.threads, 99)


# ----------------------------------------------------------------------
# reconcile_contrasts: guards the multiselect against stale session state
# after a reference-group change. Without this reconciliation Streamlit
# either crashes or silently drops selections when ``options`` changes.
# ----------------------------------------------------------------------
def test_reconcile_contrasts_preserves_valid_selection():
    current = ["HSD1_vs_NCD", "HSD3_vs_NCD"]
    available = ["HSD1_vs_NCD", "HSD3_vs_NCD", "HSD3_vs_HSD1"]
    assert reconcile_contrasts(current, available) == current


def test_reconcile_contrasts_filters_stale_entries():
    # User switched reference group from NCD to HSD1. Old state
    # ``HSD1_vs_NCD`` is no longer valid, but ``HSD3_vs_NCD`` still is.
    current = ["HSD1_vs_NCD", "HSD3_vs_NCD"]
    available = contrasts_for_reference("HSD1", ("NCD", "HSD1", "HSD3"))
    # available is ["NCD_vs_HSD1", "HSD3_vs_HSD1", "HSD3_vs_NCD"]
    reconciled = reconcile_contrasts(current, available)
    assert reconciled == ["HSD3_vs_NCD"]


def test_reconcile_contrasts_returns_empty_when_nothing_survives():
    current = ["HSD1_vs_NCD", "HSD3_vs_NCD"]
    # Simulate a total-mismatch reference group (e.g., user picked a
    # reference group that doesn't appear in any of the old contrasts)
    available = ["ALT_vs_REF"]
    assert reconcile_contrasts(current, available) == []


def test_reconcile_contrasts_empty_current_is_noop():
    assert reconcile_contrasts([], ["HSD1_vs_NCD", "HSD3_vs_NCD"]) == []
