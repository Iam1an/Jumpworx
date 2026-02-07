# tests/test_coach.py
#
# Tests for jwcore/coach.py: rule-based coaching path (no LLM).

import pytest


def test_coach_rule_based_returns_structure():
    from jwcore.coach import coach
    features = {
        "airtime_s": 0.8,
        "pitch_total_rad": 3.14,
        "ankle_dev_pct_of_torso_max": 25.0,
        "hand_dev_pct_of_torso_max": 22.0,
    }
    pro_features = {
        "ankle_dev_pct_of_torso_max": 5.0,
        "hand_dev_pct_of_torso_max": 4.0,
    }
    result = coach(features, predicted_label="backflip", pro_features=pro_features)
    assert "tips" in result
    assert "source" in result
    assert "label" in result
    assert result["source"] == "rule"
    assert isinstance(result["tips"], list)


def test_coach_empty_features():
    from jwcore.coach import coach
    result = coach({}, predicted_label="backflip")
    assert "tips" in result
    assert len(result["tips"]) > 0


def test_coach_non_dict_features():
    from jwcore.coach import coach
    result = coach(None)
    assert result["source"] == "none"
    assert result["tips"] == []
