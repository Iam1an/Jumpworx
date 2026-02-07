#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTRACT (PURE FEATURE ENGINE)
==============================
- Input NPZ schema (canonical): keys ["P","V","meta_json"]
  * P: (T, 33, 3) float32, coordinates in pixels_xy and pixel-scaled z
  * V: (T, 33) float32, per-landmark visibility in [0,1]
  * meta_json: JSON string with at least {"fps": float, "image_h": int, "image_w": int}

- Output features (single source of truth for training/inference):
  FEATURE_KEYS = [
      "pitch_total_rad",              # signed: backflip +, frontflip -
      "pitch_speed_abs_mean_rad_s",   # mean |dθ/dt| over chosen window
      "height_clearance_px_p95",      # 95th percentile feet_y - baseline
  ]

Notes on sign convention:
- SIGN_CONVENTION = +1.0  → positive pitch_total = BACKFLIP; negative = FRONTFLIP
- We compute torso pitch from the vector hip_mid->ankle_mid, choosing the plane
  (YZ / XZ / XY) that yields the largest |Δangle| and normalizing its sign to
  match the YZ convention (backflip positive).
- The old mirror_sign heuristic is applied ONLY when the chosen plane is YZ.
  (For XZ/XY we already normalized sign; applying mirror_sign would double-flip.)
"""

from __future__ import annotations
import argparse
import json
import math
from typing import Dict, Tuple, List

import numpy as np

# ---------------------------------------------------------------------
# MediaPipe Pose landmark name → index (33 landmarks)
# (Kept local to avoid importing mediapipe; indices match MP BlazePose)
# ---------------------------------------------------------------------
MP_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
NAME_TO_IDX = {n: i for i, n in enumerate(MP_NAMES)}

# ---------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------
SIGN_CONVENTION = +1.0  # +1: backflip => positive pitch_total ; -1 would invert
FEATURE_KEYS = [
    "pitch_total_rad",
    "pitch_speed_abs_mean_rad_s",
    "height_clearance_px_p95",
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _nanpercentile(a: np.ndarray, q: float) -> float:
    try:
        return float(np.nanpercentile(a, q))
    except Exception:
        return float("nan")


def estimate_mirror_sign(P: np.ndarray) -> float:
    """
    Heuristic: if right_shoulder.x < left_shoulder.x most of the time,
    return -1, else +1. This indicates the clip is mirrored horizontally.
    (Only used when the chosen plane is YZ; see notes in header.)
    """
    ls, rs = NAME_TO_IDX["left_shoulder"], NAME_TO_IDX["right_shoulder"]
    x_l = P[:, ls, 0]
    x_r = P[:, rs, 0]
    finite = np.isfinite(x_l) & np.isfinite(x_r)
    if not np.any(finite):
        return 1.0
    sign = -1.0 if float(np.nanmedian(x_r[finite] - x_l[finite])) < 0.0 else 1.0
    return sign


def detect_airtime_window(P: np.ndarray, fps: float, pad_frames: int = 2) -> Tuple[int, int]:
    """
    Detect airtime via foot clearance; FALL BACK to full clip when:
      - feet are too NaN to measure reliably, or
      - most of the clip is airborne anyway, or
      - the detected run is implausibly short (<60% of T) for Jumpworx clips.

    For your data (clips ~1–2s, mostly airtime), this typically returns (0, T-1).
    """
    T = P.shape[0]
    if T == 0:
        return 0, 0

    feet_idx = [
        NAME_TO_IDX["left_foot_index"], NAME_TO_IDX["right_foot_index"],
        NAME_TO_IDX["left_heel"], NAME_TO_IDX["right_heel"]
    ]
    feet_y = np.nanmin(P[:, feet_idx, 1], axis=1)  # smaller y = higher
    finite = np.isfinite(feet_y)
    if not np.any(finite):
        return 0, T - 1  # can’t measure clearance → use full clip

    baseline = _nanpercentile(feet_y, 5)
    clearance = feet_y - baseline
    if not np.any(np.isfinite(clearance)):
        return 0, T - 1

    thr = _nanpercentile(clearance, 60)
    airborne = clearance > thr
    airborne_ratio = float(np.mean(airborne[finite]))

    # If most of the clip is airborne, just use the full clip.
    if airborne_ratio >= 0.5:
        return 0, T - 1

    # Otherwise pick longest contiguous airborne run
    flags = np.concatenate(([False], airborne, [False]))
    edges = np.where(np.diff(flags) != 0)[0]
    if edges.size == 0:
        return 0, T - 1
    runs = edges.reshape(-1, 2)
    spans = runs[:, 1] - runs[:, 0]
    idx = int(np.argmax(spans))
    a, b = int(runs[idx, 0]), int(runs[idx, 1])  # [a, b)
    t0 = max(0, a - pad_frames)
    t1 = min(T - 1, b - 1 + pad_frames)

    # If the resulting window is still too short for Jumpworx clips, use full clip.
    if (t1 - t0 + 1) < int(0.6 * T):
        return 0, T - 1

    return t0, t1


# ---------------------------------------------------------------------
# Pitch computation (adaptive plane)
# ---------------------------------------------------------------------
def _unwrap_series(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ang = np.arctan2(b, a)
    fin = np.isfinite(ang)
    out = ang.copy()
    if fin.any():
        out[fin] = np.unwrap(ang[fin])
    return out, fin


def compute_adaptive_pitch(P: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Adaptive pitch of vector hip_mid -> ankle_mid, unwrapped.
    Chooses the viewing plane (YZ / XZ / XY) that yields the largest |Δangle|
    and normalizes its sign to match the YZ convention (backflip positive).
    Returns: (series, plane_name)
    """
    lh, rh = NAME_TO_IDX["left_hip"], NAME_TO_IDX["right_hip"]
    la, ra = NAME_TO_IDX["left_ankle"], NAME_TO_IDX["right_ankle"]
    ls, rs = NAME_TO_IDX["left_shoulder"], NAME_TO_IDX["right_shoulder"]

    hip = 0.5 * (P[:, lh, :] + P[:, rh, :])
    ank = 0.5 * (P[:, la, :] + P[:, ra, :])
    vec = ank - hip

    yz, f_yz = _unwrap_series(vec[:, 1], vec[:, 2])  # atan2(z, y)  (reference)
    xz, f_xz = _unwrap_series(vec[:, 0], vec[:, 2])  # atan2(z, x)
    xy, f_xy = _unwrap_series(vec[:, 0], vec[:, 1])  # atan2(y, x)

    def _span(series: np.ndarray, finite: np.ndarray) -> float:
        if not finite.any():
            return -np.inf
        idx = np.where(finite)[0]
        return float(series[idx[-1]] - series[idx[0]])

    d_yz = _span(yz, f_yz)
    d_xz = _span(xz, f_xz)
    d_xy = _span(xy, f_xy)

    # choose plane with largest |Δ|
    planes = [("YZ", d_yz, yz, f_yz), ("XZ", d_xz, xz, f_xz), ("XY", d_xy, xy, f_xy)]
    name, _, series, finite = max(planes, key=lambda t: abs(t[1]))

    # normalize sign to match YZ (backflip positive)
    series = series if name == "YZ" else -series

    if not finite.any():
        # fallback: hip->shoulder in YZ
        sho = 0.5 * (P[:, ls, :] + P[:, rs, :])
        torso2 = sho - hip
        y2, z2 = torso2[:, 1], torso2[:, 2]
        ang2, f2 = _unwrap_series(y2, z2)
        return ang2, "YZ"

    return series, name


def compute_torso_pitch_series(P: np.ndarray) -> np.ndarray:
    """
    Compatibility wrapper: return the adaptive-plane series only.
    (Kept for external imports/tests that expect this function.)
    """
    series, _ = compute_adaptive_pitch(P)
    return series


# ---------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------
def features_from_posetrack(P: np.ndarray, V: np.ndarray, meta: Dict) -> Dict[str, float]:
    """
    Compute the model's feature vector from (P, V, meta).
    """
    T = P.shape[0]
    fps = float(meta.get("fps", 60.0))

    # 1) airtime window (with robust full-clip fallback)
    t0, t1 = detect_airtime_window(P, fps=fps, pad_frames=2)
    t0 = int(max(0, min(t0, T - 1)))
    t1 = int(max(0, min(t1, T - 1)))
    if t1 < t0:
        t0, t1 = 0, T - 1

    # 2) torso pitch (adaptive plane) + sign logic
    pitch, plane = compute_adaptive_pitch(P)

    # robust delta: use first/last finite inside [t0,t1]; fallback to whole clip
    seg_idx = np.arange(t0, t1 + 1, dtype=int)
    seg_fin = np.isfinite(pitch[seg_idx])
    if seg_fin.sum() >= 2:
        idxs = seg_idx[seg_fin]
        dtheta_total = float(pitch[idxs[-1]] - pitch[idxs[0]])
    else:
        all_fin = np.isfinite(pitch)
        if all_fin.sum() >= 2:
            ai = np.where(all_fin)[0]
            dtheta_total = float(pitch[ai[-1]] - pitch[ai[0]])
        else:
            dtheta_total = float("nan")

    mir = estimate_mirror_sign(P)
    # Apply mirror_sign ONLY for YZ (reference plane)
    mirror_factor = mir if plane == "YZ" else 1.0

    signed_total = SIGN_CONVENTION * mirror_factor * dtheta_total

    pitch_total_turns = signed_total / (2.0 * math.pi)

    # 3) mean |angular speed| over window
    # central difference on the unwrapped series (already in radians)
    seg = slice(t0, t1 + 1)
    ps = pitch[seg]
    finite = np.isfinite(ps)
    if finite.sum() >= 3:
        ang = ps[finite]
        dt = 1.0 / max(fps, 1e-6)
        # simple finite difference
        d = np.diff(ang) / dt
        pitch_speed_abs_mean = float(np.nanmean(np.abs(d)))
    else:
        pitch_speed_abs_mean = 0.0

    # 4) foot clearance (p95 above baseline)
    feet_idx = [
        NAME_TO_IDX["left_foot_index"], NAME_TO_IDX["right_foot_index"],
        NAME_TO_IDX["left_heel"], NAME_TO_IDX["right_heel"]
    ]
    feet_y = np.nanmin(P[:, feet_idx, 1], axis=1)  # smaller y = higher; use baseline diff
    baseline = _nanpercentile(feet_y, 5)
    clearance = feet_y - baseline
    height_clearance_px_p95 = _nanpercentile(clearance, 95)

    feats = {
        "pitch_total_rad": float(signed_total),
        "pitch_total_turns": float(pitch_total_turns),  # helpful for debugging/inspection
        "pitch_speed_abs_mean_rad_s": pitch_speed_abs_mean,
        "height_clearance_px_p95": height_clearance_px_p95,
        # expose diagnostics (not part of FEATURE_KEYS, but useful via CLI)
        "air_t0": float(t0),
        "air_t1": float(t1),
        "airtime_s": float((t1 - t0 + 1) / max(fps, 1e-6)),
        "mirror_sign": float(mir),
        "plane": plane,
        "fps": float(fps),
        "n_frames": float(T),
    }
    # Return only the model features to callers that expect the vector:
    return feats


# ---------------------------------------------------------------------
# CLI (for quick inspection)
# ---------------------------------------------------------------------
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Compute features from canonical .posetrack.npz")
    ap.add_argument("--npz", required=True, help="Path to canonical NPZ (with P,V,meta_json)")
    args = ap.parse_args()

    from jwcore.posetrack_io import load_posetrack_npz
    P, V, _fps, meta = load_posetrack_npz(args.npz)
    feats = features_from_posetrack(P, V, meta)

    # Pretty print a compact summary honoring FEATURE_KEYS
    out = {
        "air_t0": int(feats["air_t0"]),
        "air_t1": int(feats["air_t1"]),
        "airtime_s": feats["airtime_s"],
        "height_clearance_px_p95": feats["height_clearance_px_p95"],
        "mirror_sign": feats["mirror_sign"],
        "plane": feats["plane"],
        "pitch_speed_abs_mean_rad_s": feats["pitch_speed_abs_mean_rad_s"],
        "pitch_total_rad": feats["pitch_total_rad"],
        "pitch_total_turns": feats["pitch_total_turns"],
    }
    print(f"=== {args.npz} ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
