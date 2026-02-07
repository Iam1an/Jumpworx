# tests/test_compare_metrics.py
#
# Tests for jwcore/compare_metrics.py: DTW alignment, scalar metrics,
# shape validation.

import numpy as np
import pytest


def test_compare_metrics_identical_poses():
    from jwcore.compare_metrics import compare_metrics_from_xyz
    T, J = 30, 33
    rng = np.random.RandomState(99)
    pose = rng.randn(T, J, 3).astype(np.float32)
    result = compare_metrics_from_xyz(pose.copy(), pose.copy())
    assert "scalars" in result
    assert "joints" in result
    assert "alignment" in result
    assert "phase_scores" in result
    # Identical poses: pitch delta should be ~0
    pitch_delta = result["scalars"]["pitch_total_rad"]
    assert abs(pitch_delta) < 0.01, f"pitch_total_rad for identical poses: {pitch_delta}"


def test_compare_metrics_returns_all_scalar_keys():
    from jwcore.compare_metrics import compare_metrics_from_xyz
    T, J = 20, 33
    am = np.random.RandomState(1).randn(T, J, 3).astype(np.float32)
    pro = np.random.RandomState(2).randn(T, J, 3).astype(np.float32)
    result = compare_metrics_from_xyz(am, pro)
    expected_keys = [
        "midair_hand_span_pct_of_torso",
        "midair_hand_asym_pct_of_torso",
        "landing_stance_width_pct_of_hip",
        "pitch_total_rad",
        "ankle_dev_pct_of_torso_max",
        "hand_dev_pct_of_torso_max",
        "leg_axis_diff_deg_apex",
        "head_early_pitch_lead_deg",
    ]
    for key in expected_keys:
        assert key in result["scalars"], f"Missing scalar key: {key}"


def test_compare_metrics_shape_validation():
    from jwcore.compare_metrics import compare_metrics_from_xyz
    bad = np.zeros((10, 33), dtype=np.float32)  # 2D, not 3D
    with pytest.raises(ValueError, match="must be.*3.*arrays"):
        compare_metrics_from_xyz(bad, bad)
