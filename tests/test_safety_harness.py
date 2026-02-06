# tests/test_safety_harness.py
#
# Safety harness: locks down core module behaviors with deterministic,
# synthetic-data tests. No external dependencies (no cv2, mediapipe, LLM).
#
# Run: python -m pytest tests/test_safety_harness.py -v

import json
import math
import os

import numpy as np
import pytest


# ===================================================================
# 1. Package import (no cv2 required)
# ===================================================================

def test_jwcore_import_without_cv2():
    """jwcore package must import without cv2/mediapipe installed."""
    import jwcore
    assert hasattr(jwcore, "load_pose")
    assert hasattr(jwcore, "normalize_prerot")
    assert hasattr(jwcore, "height_series_up")


# ===================================================================
# 2. Normalization (jwcore/normalize.py)
# ===================================================================

def test_normalize_prerot_shape():
    from jwcore.normalize import normalize_prerot
    kps = np.zeros((10, 33, 2), dtype=np.float32)
    result = normalize_prerot(kps)
    assert result.shape == (10, 33, 2)


def test_normalize_prerot_finite_output():
    from jwcore.normalize import normalize_prerot
    kps = np.random.RandomState(42).randn(20, 33, 2).astype(np.float32)
    result = normalize_prerot(kps)
    assert np.isfinite(result).all()


def test_height_series_up_shape():
    from jwcore.normalize import normalize_prerot, height_series_up
    kps = np.zeros((15, 33, 2), dtype=np.float32)
    kps_norm = normalize_prerot(kps)
    heights = height_series_up(kps_norm)
    assert heights.shape == (15,)


# ===================================================================
# 3. Phase segmentation (jwcore/phase_segmentation.py)
# ===================================================================

def _make_synthetic_jump(T=120, fps=60, air_start=40, air_end=80):
    """Synthetic jump with known airtime window."""
    kps = np.zeros((T, 33, 3), dtype=np.float32)
    kps[:, :, 1] = 0.5  # baseline ground y
    for t in range(T):
        if air_start <= t <= air_end:
            for j in [27, 28, 31, 32]:  # ankles + toes
                kps[t, j, 1] = 0.2  # "in air" (lower y)
        else:
            for j in [27, 28, 31, 32]:
                kps[t, j, 1] = 0.5
    return kps, fps, air_start, air_end


def test_phase_segmentation_returns_phases_dataclass():
    """Verify segment_phases_with_airtime_v2 returns (Phases, debug) tuple."""
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2, Phases
    kps = np.full((60, 33, 3), 0.5, dtype=np.float32)
    result = segment_phases_with_airtime_v2(kps, fps=30)
    assert isinstance(result, tuple)
    assert len(result) == 2
    phases, debug = result
    assert isinstance(phases, Phases)
    assert isinstance(phases.quality, dict)
    assert "status" in phases.quality


def test_phase_segmentation_quality_status():
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2
    kps = np.full((60, 33, 3), 0.5, dtype=np.float32)
    phases, _ = segment_phases_with_airtime_v2(kps, fps=30)
    assert phases.quality["status"] in ("ok", "warn", "fail")


def test_phase_segmentation_debug_output():
    """When return_debug=True, debug dict should be a dict (even if empty on fail)."""
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2
    kps = np.full((60, 33, 3), 0.5, dtype=np.float32)
    phases, debug = segment_phases_with_airtime_v2(kps, fps=30, return_debug=True)
    assert debug is not None
    assert isinstance(debug, dict)


def test_phase_fine_labels_from_phases():
    """fine_labels_from_phases generates per-frame labels of correct length."""
    from jwcore.phase_segmentation import Phases, fine_labels_from_phases
    T = 100
    ph = Phases(
        approach=(0, 10), set=(10, 20), airtime=(20, 80),
        landing=(80, 100), takeoff_idx=20, landing_idx=80,
        airtime_seconds=1.0, quality={"status": "ok"},
    )
    labels = fine_labels_from_phases(ph, T)
    assert len(labels) == T
    assert "early_air" in labels
    assert "midair" in labels


def test_phase_result_from_pose_attributes():
    """phase_result_from_pose returns PhaseResult with expected attributes."""
    from jwcore.phase_segmentation import PhaseResult
    # Just verify the dataclass shape
    pr = PhaseResult(
        labels=["unknown"] * 10,
        takeoff_idx=None, landing_idx=None,
        air_start=None, air_end=None, apex_idx=None,
        quality={"status": "fail"},
    )
    assert hasattr(pr, "labels")
    assert hasattr(pr, "takeoff_idx")
    assert hasattr(pr, "quality")


def test_phase_no_air_returns_fail():
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2
    # Flat signal — no airtime
    kps = np.full((60, 33, 3), 0.5, dtype=np.float32)
    phases, _ = segment_phases_with_airtime_v2(kps, fps=30)
    assert phases.quality["status"] == "fail"


# ===================================================================
# 4. Pose utils / feature extraction (jwcore/pose_utils.py)
# ===================================================================

def test_feature_keys_contract():
    from jwcore.pose_utils import FEATURE_KEYS
    assert isinstance(FEATURE_KEYS, list)
    assert len(FEATURE_KEYS) == 3
    assert "pitch_total_rad" in FEATURE_KEYS
    assert "pitch_speed_abs_mean_rad_s" in FEATURE_KEYS
    assert "height_clearance_px_p95" in FEATURE_KEYS


def test_features_from_posetrack_returns_all_keys():
    from jwcore.pose_utils import features_from_posetrack, FEATURE_KEYS
    T = 30
    P = np.zeros((T, 33, 3), dtype=np.float32)
    V = np.ones((T, 33), dtype=np.float32)
    meta = {"fps": 30.0, "image_h": 720, "image_w": 1280}
    feats = features_from_posetrack(P, V, meta)
    for key in FEATURE_KEYS:
        assert key in feats, f"Missing feature key: {key}"


def test_features_from_posetrack_deterministic():
    from jwcore.pose_utils import features_from_posetrack
    rng = np.random.RandomState(123)
    P = rng.randn(40, 33, 3).astype(np.float32)
    V = np.ones((40, 33), dtype=np.float32)
    meta = {"fps": 60.0}
    f1 = features_from_posetrack(P, V, meta)
    f2 = features_from_posetrack(P, V, meta)
    for k in f1:
        if isinstance(f1[k], float) and math.isfinite(f1[k]):
            assert f1[k] == f2[k], f"Non-deterministic feature: {k}"


def test_compute_adaptive_pitch_returns_tuple():
    from jwcore.pose_utils import compute_adaptive_pitch
    P = np.zeros((20, 33, 3), dtype=np.float32)
    series, plane = compute_adaptive_pitch(P)
    assert isinstance(series, np.ndarray)
    assert series.shape == (20,)
    assert plane in ("YZ", "XZ", "XY")


def test_estimate_mirror_sign_values():
    from jwcore.pose_utils import estimate_mirror_sign
    P = np.zeros((10, 33, 3), dtype=np.float32)
    s = estimate_mirror_sign(P)
    assert s in (-1.0, 1.0)


def test_sign_convention_positive():
    from jwcore.pose_utils import SIGN_CONVENTION
    assert SIGN_CONVENTION == 1.0


# ===================================================================
# 5. Coaching thresholds (jwcore/coaching_thresholds.py)
# ===================================================================

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


# ===================================================================
# 6. Coach (jwcore/coach.py) — rule-based path only
# ===================================================================

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


# ===================================================================
# 7. Compare metrics (jwcore/compare_metrics.py)
# ===================================================================

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


# ===================================================================
# 8. Posetrack I/O (jwcore/posetrack_io.py)
# ===================================================================

def test_load_posetrack_canonical_layout(tmp_path):
    from jwcore.posetrack_io import load_posetrack
    T, J = 10, 33
    P = np.random.RandomState(7).randn(T, J, 3).astype(np.float32)
    V = np.ones((T, J), dtype=np.float32)
    meta = json.dumps({"fps": 60.0, "image_h": 720, "image_w": 1280})
    path = str(tmp_path / "test.posetrack.npz")
    np.savez_compressed(path, P=P, V=V, meta_json=meta)
    P_out, V_out, fps, meta_out = load_posetrack(path)
    assert P_out.shape == (T, J, 3)
    assert V_out.shape == (T, J)
    assert fps == 60.0
    np.testing.assert_array_almost_equal(P_out, P)


def test_load_posetrack_legacy_layout(tmp_path):
    from jwcore.posetrack_io import load_posetrack
    T, J = 8, 33
    kps = np.zeros((T, J, 3), dtype=np.float32)
    vis = np.ones((T, J), dtype=np.float32)
    path = str(tmp_path / "legacy.npz")
    np.savez_compressed(path, kps_xyz=kps, visibility=vis, fps=np.array([30.0]))
    P, V, fps, meta = load_posetrack(path)
    assert P.shape == (T, J, 3)
    assert fps == 30.0


def test_load_posetrack_missing_file():
    from jwcore.posetrack_io import load_posetrack
    with pytest.raises(FileNotFoundError):
        load_posetrack("/nonexistent/path.npz")


def test_load_posetrack_bad_layout(tmp_path):
    from jwcore.posetrack_io import load_posetrack
    path = str(tmp_path / "bad.npz")
    np.savez_compressed(path, random_key=np.zeros(5))
    with pytest.raises(KeyError):
        load_posetrack(path)


# ===================================================================
# 9. Pose extract / process_file (jwcore/pose_extract.py)
# ===================================================================

def test_process_file_creates_json(tmp_path):
    from jwcore.pose_extract import process_file
    T, J = 10, 33
    P = np.zeros((T, J, 3), dtype=np.float32)
    V = np.ones((T, J), dtype=np.float32)
    meta = json.dumps({"fps": 30.0, "image_h": 720, "image_w": 1280})
    npz_path = str(tmp_path / "test.posetrack.npz")
    np.savez_compressed(npz_path, P=P, V=V, meta_json=meta)
    out_dir = str(tmp_path)
    result_path = process_file(npz_path, out_dir, verbose=False)
    assert os.path.exists(result_path)
    with open(result_path) as f:
        obj = json.load(f)
    assert "features" in obj
    assert isinstance(obj["features"], dict)


# ===================================================================
# 10. Joints constants (jwcore/joints.py)
# ===================================================================

def test_joints_constants():
    from jwcore.joints import (
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
        JOINT_NAME_BY_INDEX,
    )
    assert NOSE == 0
    assert LEFT_SHOULDER == 11
    assert RIGHT_SHOULDER == 12
    assert LEFT_HIP == 23
    assert RIGHT_HIP == 24
    assert LEFT_ANKLE == 27
    assert RIGHT_ANKLE == 28
    assert isinstance(JOINT_NAME_BY_INDEX, dict)
    assert len(JOINT_NAME_BY_INDEX) == 33


# ===================================================================
# 11. Trick classifier (jwcore/trick_classifier.py)
# ===================================================================

def test_trick_classifier_feature_names():
    """TrickClassifier.feature_names must match pose_utils.FEATURE_KEYS."""
    from jwcore.trick_classifier import TrickClassifier
    from jwcore.pose_utils import FEATURE_KEYS
    clf = TrickClassifier()
    assert clf.feature_keys == FEATURE_KEYS


def test_trick_classifier_predict():
    from jwcore.trick_classifier import TrickClassifier
    clf = TrickClassifier()
    feats = {
        "pitch_total_rad": 5.0,
        "pitch_speed_abs_mean_rad_s": 20.0,
        "height_clearance_px_p95": 100.0,
    }
    label = clf.predict(feats)
    assert isinstance(label, str)
    assert len(label) > 0


def test_trick_classifier_predict_with_proba():
    from jwcore.trick_classifier import TrickClassifier
    clf = TrickClassifier()
    feats = {
        "pitch_total_rad": 5.0,
        "pitch_speed_abs_mean_rad_s": 20.0,
        "height_clearance_px_p95": 100.0,
    }
    label, proba = clf.predict_with_proba(feats)
    assert isinstance(label, str)
    # proba can be None or float
    if proba is not None:
        assert 0.0 <= proba <= 1.0


# ===================================================================
# 12. Feature deltas (jwcore/feature_deltas.py)
# ===================================================================

def test_feature_deltas():
    from jwcore.feature_deltas import stance_hand_deltas, STANCE_KEY, HAND_KEY
    am = {STANCE_KEY: 1.2, HAND_KEY: 0.8}
    pro = {STANCE_KEY: 1.0, HAND_KEY: 0.6}
    deltas = stance_hand_deltas(am, pro)
    assert isinstance(deltas, dict)
    assert "stance_width_delta" in deltas
    assert "hand_span_delta" in deltas
    assert abs(deltas["stance_width_delta"] - 0.2) < 1e-6
    assert abs(deltas["hand_span_delta"] - 0.2) < 1e-6
