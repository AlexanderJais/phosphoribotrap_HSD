"""Smoke tests for AppConfig load/save/diff behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from phosphotrap.config import DEFAULT_CONTRASTS, AppConfig


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
