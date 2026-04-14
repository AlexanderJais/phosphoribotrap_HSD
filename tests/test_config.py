"""Smoke tests for AppConfig load/save/diff behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from phosphotrap.config import (
    DEFAULT_CONTRASTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCE_GROUP,
    DEFAULT_REPORT_DIR,
    AppConfig,
    contrasts_for_reference,
    is_safe_contrast,
    is_safe_token,
    reconcile_contrasts,
    resolve_rscript,
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


# ----------------------------------------------------------------------
# Safe-token + contrast validation, and config-load path-traversal guard
# ----------------------------------------------------------------------
def test_is_safe_token_allows_alnum_underscore_hyphen():
    assert is_safe_token("NCD")
    assert is_safe_token("HSD1")
    assert is_safe_token("group-1")
    assert is_safe_token("group_1")


def test_is_safe_token_rejects_path_traversal_and_whitespace():
    assert not is_safe_token("")
    assert not is_safe_token("../evil")
    assert not is_safe_token("HSD 1")
    assert not is_safe_token("HSD/1")
    assert not is_safe_token("HSD\x001")
    assert not is_safe_token(".")


def test_is_safe_contrast_requires_vs_separator():
    assert is_safe_contrast("HSD1_vs_NCD")
    assert is_safe_contrast("alt-1_vs_ref-2")
    assert not is_safe_contrast("HSD1vs_NCD")   # missing underscore prefix
    assert not is_safe_contrast("_vs_NCD")      # empty alt
    assert not is_safe_contrast("HSD1_vs_")     # empty ref
    assert not is_safe_contrast("HSD1_vs_../evil")
    assert not is_safe_contrast("HSD1 _vs_ NCD")


def test_load_rejects_path_traversal_reference_group(tmp_path: Path):
    """A JSON-edited config with a malicious reference_group must not
    propagate the dangerous value. It should fall back to the default.
    """
    p = tmp_path / "phosphotrap.json"
    p.write_text(json.dumps({
        "reference_group": "../evil",
        "contrasts": ["HSD1_vs_NCD", "HSD3_vs_NCD"],
    }))
    cfg = AppConfig.load(p)
    assert cfg.reference_group == DEFAULT_REFERENCE_GROUP
    # Valid contrasts still survive.
    assert cfg.contrasts == ["HSD1_vs_NCD", "HSD3_vs_NCD"]


def test_load_drops_malicious_contrasts_keeps_valid_ones(tmp_path: Path):
    p = tmp_path / "phosphotrap.json"
    p.write_text(json.dumps({
        "reference_group": "NCD",
        "contrasts": [
            "HSD1_vs_NCD",      # valid
            "HSD1_vs_../evil",  # path traversal
            "rm -rf /",         # junk
            "HSD3_vs_NCD",      # valid
        ],
    }))
    cfg = AppConfig.load(p)
    assert cfg.contrasts == ["HSD1_vs_NCD", "HSD3_vs_NCD"]


def test_load_all_bad_contrasts_falls_through_to_default(tmp_path: Path):
    p = tmp_path / "phosphotrap.json"
    p.write_text(json.dumps({
        # None of these match ^[safe]_vs_[safe]$: path traversal, no
        # "_vs_" separator at all, whitespace.
        "contrasts": ["../evil", "rm_rf_slash", "HSD1 vs NCD"],
    }))
    cfg = AppConfig.load(p)
    assert cfg.contrasts == list(DEFAULT_CONTRASTS)


def test_resolve_rscript_returns_configured_value():
    cfg = AppConfig(rscript_path="/opt/conda/envs/phosphotrap/bin/Rscript")
    assert resolve_rscript(cfg) == "/opt/conda/envs/phosphotrap/bin/Rscript"


def test_resolve_rscript_empty_falls_back_to_path_lookup():
    cfg = AppConfig(rscript_path="")
    assert resolve_rscript(cfg) == "Rscript"


# ----------------------------------------------------------------------
# effective_output_dir / effective_report_dir:
# guard against an empty / "." path field silently flowing through
# Path(cfg.output_dir) and scattering output/salmon/ into cwd.
# ----------------------------------------------------------------------
def test_effective_output_dir_empty_falls_back_to_default():
    cfg = AppConfig(output_dir="")
    assert cfg.effective_output_dir() == Path(DEFAULT_OUTPUT_DIR)


def test_effective_output_dir_whitespace_falls_back_to_default():
    cfg = AppConfig(output_dir="   ")
    assert cfg.effective_output_dir() == Path(DEFAULT_OUTPUT_DIR)


def test_effective_output_dir_dot_falls_back_to_default():
    """'.' resolves to cwd via Path — treat it as 'blank' so the
    user doesn't accidentally pollute wherever streamlit was launched.
    """
    cfg = AppConfig(output_dir=".")
    assert cfg.effective_output_dir() == Path(DEFAULT_OUTPUT_DIR)


def test_effective_output_dir_honours_explicit_path():
    cfg = AppConfig(output_dir="/tmp/phosphotrap_out")
    assert cfg.effective_output_dir() == Path("/tmp/phosphotrap_out")


def test_effective_output_dir_strips_surrounding_whitespace():
    cfg = AppConfig(output_dir="  /tmp/phosphotrap_out  ")
    assert cfg.effective_output_dir() == Path("/tmp/phosphotrap_out")


def test_effective_report_dir_empty_falls_back_to_default():
    cfg = AppConfig(report_dir="")
    assert cfg.effective_report_dir() == Path(DEFAULT_REPORT_DIR)


def test_effective_report_dir_honours_explicit_path():
    cfg = AppConfig(report_dir="/var/log/phosphotrap")
    assert cfg.effective_report_dir() == Path("/var/log/phosphotrap")
