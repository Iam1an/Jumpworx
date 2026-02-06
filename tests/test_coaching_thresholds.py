# tests/test_coaching_thresholds.py
#
# Tests for jwcore/coaching_thresholds.py: TAU/UNITS/PHASE_TAG consistency,
# material_delta, select_top_metrics.

import pytest


def test_tau_keys_match_units_and_phase_tag():
    from jwcore.coaching_thresholds import TAU, UNITS, PHASE_TAG
    for key in TAU:
        assert key in UNITS, f"TAU key '{key}' missing from UNITS"
        assert key in PHASE_TAG, f"TAU key '{key}' missing from PHASE_TAG"


def test_material_delta_above_threshold():
    from jwcore.coaching_thresholds import material_delta
    assert material_delta("pitch_total_rad", 0.5) is True
    assert material_delta("pitch_total_rad", 0.1) is False


def test_select_top_metrics_filters_below_tau():
    from jwcore.coaching_thresholds import select_top_metrics
    am = {"pitch_total_rad": 0.1}  # below TAU of 0.35
    result = select_top_metrics(am, None, ["pitch_total_rad"])
    assert len(result) == 0


def test_select_top_metrics_includes_above_tau():
    from jwcore.coaching_thresholds import select_top_metrics
    am = {"pitch_total_rad": 1.0}
    pro = {"pitch_total_rad": 0.2}
    result = select_top_metrics(am, pro, ["pitch_total_rad"])
    assert len(result) == 1
    assert result[0]["key"] == "pitch_total_rad"
    assert abs(result[0]["delta"] - 0.8) < 1e-6


def test_select_top_metrics_respects_max_items():
    from jwcore.coaching_thresholds import select_top_metrics, TAU
    am = {}
    pro = {}
    keys = []
    for i, key in enumerate(TAU):
        am[key] = TAU[key] * 3  # well above threshold
        pro[key] = 0.0
        keys.append(key)
    result = select_top_metrics(am, pro, keys, max_items=2)
    assert len(result) <= 2
