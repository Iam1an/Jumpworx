# jwcore/pro_utils.py
# Helpers for managing pro reference metrics ("pro_features").
#
# Goals:
# - For each pro .posetrack.npz, compute a stable set of scalar reference metrics.
# - Cache them as JSON next to the NPZ.
# - Auto-rebuild if the NPZ changes (mtime-based).
#
# These pro_features are used by:
#   - nearest-pro selection
#   - coach.py (via select_top_metrics) as the "ideal" comparison target.

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from jwcore.posetrack_io import load_posetrack
from jwcore.joints import (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


# ========= Internal helpers (minimal, local) =========

def _pelvis_center_torso_scale(frame_xy: np.ndarray) -> tuple[np.ndarray, float]:
    hips = frame_xy[[LEFT_HIP, RIGHT_HIP]]
    shs = frame_xy[[LEFT_SHOULDER, RIGHT_SHOULDER]]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return frame_xy.copy(), 1.0
    pelvis = hips.mean(axis=0)
    shmid = shs.mean(axis=0)
    span = float(np.linalg.norm(shmid - pelvis))
    scale = 1.0 if span < 1e-6 else (1.0 / span)
    return (frame_xy - pelvis) * scale, scale


def _normalize_pose_sequence(arr_xyz: np.ndarray) -> np.ndarray:
    """Pelvis-centered, torso-scaled XY sequence."""
    T, J, _ = arr_xyz.shape
    out = np.empty((T, J, 2), dtype=np.float32)
    for t in range(T):
        xy = arr_xyz[t, :, :2]
        xy_n, _ = _pelvis_center_torso_scale(xy)
        out[t] = xy_n
    return out


def _midair_mask(T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    start = int(0.40 * T)
    end = max(start + 1, int(0.60 * T))
    m[start:end] = True
    return m


def _land_idx_heuristic(T: int) -> int:
    return max(0, int(0.85 * (T - 1)))


def _hand_span_pct_of_torso(xy: np.ndarray) -> float:
    if not (np.isfinite(xy[LEFT_WRIST]).all() and np.isfinite(xy[RIGHT_WRIST]).all()):
        return float("nan")
    hips = xy[[LEFT_HIP, RIGHT_HIP]]
    shs = xy[[LEFT_SHOULDER, RIGHT_SHOULDER]]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return float("nan")
    pelvis = hips.mean(axis=0)
    shmid = shs.mean(axis=0)
    torso = float(np.linalg.norm(shmid - pelvis))
    if torso < 1e-6:
        return float("nan")
    span = float(np.linalg.norm(xy[LEFT_WRIST] - xy[RIGHT_WRIST]))
    return 100.0 * (span / torso)


def _hand_asym_pct_of_torso(xy: np.ndarray) -> float:
    hips = xy[[LEFT_HIP, RIGHT_HIP]]
    shs = xy[[LEFT_SHOULDER, RIGHT_SHOULDER]]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return float("nan")
    pelvis = hips.mean(axis=0)
    shmid = shs.mean(axis=0)
    torso = float(np.linalg.norm(shmid - pelvis))
    if torso < 1e-6:
        return float("nan")
    midx = 0.5 * (pelvis[0] + shmid[0])
    if not (np.isfinite(xy[LEFT_WRIST]).all() and np.isfinite(xy[RIGHT_WRIST]).all()):
        return float("nan")
    dxL = float(xy[LEFT_WRIST, 0] - midx)
    dxR = float(xy[RIGHT_WRIST, 0] - midx)
    asym = abs(dxL + dxR)
    return 100.0 * (asym / torso)


def _stance_width_pct_of_hip(xy: np.ndarray) -> float:
    if not (np.isfinite(xy[LEFT_ANKLE]).all() and np.isfinite(xy[RIGHT_ANKLE]).all()):
        return float("nan")
    hips = xy[[LEFT_HIP, RIGHT_HIP]]
    if not np.isfinite(hips).all():
        return float("nan")
    hip_w = float(np.linalg.norm(hips[0] - hips[1]))
    if hip_w < 1e-6:
        return float("nan")
    stance = float(np.linalg.norm(xy[LEFT_ANKLE] - xy[RIGHT_ANKLE]))
    return 100.0 * (stance / hip_w)


# ========= Public: compute & cache pro_features =========

def compute_pro_features_from_xyz(xyz: np.ndarray) -> Dict[str, float]:
    """
    Compute baseline 'ideal' metrics from a single pro pose track.

    For metrics that are inherently defined as deltas between athlete and pro
    (e.g. ankle_dev_pct_of_torso_max), the pro baseline is 0.0 by design.
    """
    if xyz.ndim != 3:
        raise ValueError("xyz must be (T, J, 3)")

    xy = _normalize_pose_sequence(xyz)
    T = xy.shape[0]
    if T < 4:
        # Too short; return neutral defaults.
        return {
            "midair_hand_span_pct_of_torso": 100.0,
            "midair_hand_asym_pct_of_torso": 0.0,
            "landing_stance_width_pct_of_hip": 100.0,
            "pitch_total_rad": 0.0,
            "ankle_dev_pct_of_torso_max": 0.0,
            "hand_dev_pct_of_torso_max": 0.0,
            "leg_axis_diff_deg_apex": 0.0,
            "head_early_pitch_lead_deg": 0.0,
        }

    midair = _midair_mask(T)
    land_idx = _land_idx_heuristic(T)

    # Midair averages
    def _midair_mean(fn) -> float:
        vals = []
        for t in range(T):
            if not midair[t]:
                continue
            v = fn(xy[t])
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    midair_span = _midair_mean(_hand_span_pct_of_torso)
    midair_asym = _midair_mean(_hand_asym_pct_of_torso)

    # Landing stance
    landing_xy = xy[min(max(0, land_idx), T - 1)]
    landing_stance = _stance_width_pct_of_hip(landing_xy)

    # For flip tricks, we treat the pro's net pitch delta and all "delta" metrics
    # as reference 0; athlete values are interpreted relative to these.
    return {
        "midair_hand_span_pct_of_torso": float(midair_span),
        "midair_hand_asym_pct_of_torso": float(midair_asym),
        "landing_stance_width_pct_of_hip": float(landing_stance),
        "pitch_total_rad": 0.0,
        "ankle_dev_pct_of_torso_max": 0.0,
        "hand_dev_pct_of_torso_max": 0.0,
        "leg_axis_diff_deg_apex": 0.0,
        "head_early_pitch_lead_deg": 0.0,
    }


def get_or_build_pro_features(npz_path: str) -> Dict[str, float]:
    """
    Return pro_features for a given pro .posetrack.npz.

    - Looks for '<npz_path>.profeatures.json'.
    - If JSON is missing or older than NPZ, recomputes and overwrites.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    json_path = npz_path + ".profeatures.json"
    npz_mtime = os.path.getmtime(npz_path)

    # Try to load cached if fresh
    if os.path.isfile(json_path):
        try:
            j_mtime = os.path.getmtime(json_path)
            if j_mtime >= npz_mtime:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            # Fall through to recompute
            pass

    # (Re)compute from NPZ
    P, _, _, _ = load_posetrack(npz_path)
    feats = compute_pro_features_from_xyz(P)

    # Best-effort cache write
    try:
        with open(json_path, "w") as f:
            json.dump(feats, f, indent=2)
    except Exception:
        # Non-fatal: still return computed features
        pass

    return feats


__all__ = [
    "compute_pro_features_from_xyz",
    "get_or_build_pro_features",
]
