#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_frame_table.py

Generate a dense frame-by-frame CSV comparing an amateur and pro pose-track.

Goals:
  - Reuse jwcore.compare_metrics helpers wherever possible.
  - Align sequences (DTW on 1D proxy via compare_metrics).
  - Normalize to pelvis-centered, torso-length ~ 1.
  - For each aligned frame, output:
      * per-frame similarity score (0–100)
      * core kinematic metrics (pitch, head, knee, spans, etc.)
      * per-joint normalized coords + deltas (am_, pr_, d_)
      * optional phase labels if jwcore.phase_detect is available
  - Print raw, post-align, and CSV-level missing-data stats.

Usage:
  python -m scripts.build_frame_table \
      cache/AM.posetrack.npz \
      cache/PRO.posetrack.npz \
      --align_feature ankle_y \
      --out_csv viz/frame_table_AM_vs_PRO.csv
"""

import os
import sys
import csv
import argparse
from typing import Optional, Dict, Any, List

import numpy as np

# Make jwcore importable when script is in ./scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.joints import JOINT_NAME_BY_INDEX

# We intentionally reach into compare_metrics "private" helpers
# to keep math consistent across code paths.
from jwcore.compare_metrics import (
    _align_sequences_dtw as cm_align_sequences_dtw,
    _normalize_pose_sequence as cm_normalize_pose_sequence,
    _torso_pitch_deg_series as cm_torso_pitch_deg_series,
    _head_pitch_deg_series as cm_head_pitch_deg_series,
    _knee_mean_deg_series as cm_knee_mean_deg_series,
    _series_ankle_y as cm_series_ankle_y,
    _series_hip_y as cm_series_hip_y,
    _stance_width_pct_of_hip as cm_stance_width_pct_of_hip,
    _hand_span_pct_of_torso as cm_hand_span_pct_of_torso,
    _hand_asym_pct_of_torso as cm_hand_asym_pct_of_torso,
    _leg_axis_angle_deg as cm_leg_axis_angle_deg,
)

# Optional phase detection
try:
    from jwcore.phase_detect import detect_phases_from_pose, PhaseResult  # type: ignore
    PHASE_DETECT_AVAILABLE = True
except Exception:
    detect_phases_from_pose = None  # type: ignore
    PhaseResult = None  # type: ignore
    PHASE_DETECT_AVAILABLE = False


# ==================== Score hyperparams (match viz) ====================

D0_JOINT = 0.15   # torso units
A0_PITCH = 20.0   # deg
A0_KNEE  = 25.0   # deg
A0_HEAD  = 15.0   # deg

W_JOINT = 1.00
W_PITCH = 0.60
W_KNEE  = 0.50
W_HEAD  = 0.60


# ==================== Utilities ====================

def _missing_pct(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    m = ~np.isfinite(arr)
    return 100.0 * float(m.sum()) / float(arr.size)


def _compute_frame_similarity_scores(
    am_xy: np.ndarray,
    pr_xy: np.ndarray,
    include_head: bool = True,
) -> np.ndarray:
    """
    Rebuilds the same style of similarity score as viz_compare_side_by_side:
      - per-joint RBF over torso-normalized distance
      - pitch diff
      - knee mean angle diff
      - optional head pitch diff
    Returns:
      (T,) float32 scores in [0,100].
    """
    T, J, _ = am_xy.shape
    assert pr_xy.shape == am_xy.shape

    # Per-joint distances (torso units)
    dists = np.linalg.norm(am_xy[:, :, :2] - pr_xy[:, :, :2], axis=2)  # (T, J)

    # Joint similarity: average RBF over all finite joints per frame
    with np.errstate(invalid="ignore"):
        Sj = np.exp(- (dists / D0_JOINT) ** 2)  # (T, J)
    S_joint = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = np.isfinite(Sj[t])
        S_joint[t] = float(np.nanmean(Sj[t, m])) if m.any() else 0.0

    # Pitch, head, knee series from compare_metrics helpers
    pitch_am = cm_torso_pitch_deg_series(am_xy)
    pitch_pr = cm_torso_pitch_deg_series(pr_xy)
    head_am = cm_head_pitch_deg_series(am_xy)
    head_pr = cm_head_pitch_deg_series(pr_xy)
    knee_am = cm_knee_mean_deg_series(am_xy)
    knee_pr = cm_knee_mean_deg_series(pr_xy)

    d_pitch = np.abs(pitch_am - pitch_pr)
    d_knee = np.abs(knee_am - knee_pr)
    d_head = np.abs(head_am - head_pr) if include_head else None

    S_pitch = np.exp(- (d_pitch / A0_PITCH) ** 2)
    S_knee  = np.exp(- (d_knee  / A0_KNEE ) ** 2)
    if d_head is not None:
        S_head = np.exp(- (d_head / A0_HEAD) ** 2)
    else:
        S_head = None

    w_sum = W_JOINT + W_PITCH + W_KNEE + (W_HEAD if S_head is not None else 0.0)
    sim = (
        W_JOINT * S_joint
        + W_PITCH * S_pitch
        + W_KNEE  * S_knee
        + (W_HEAD * S_head if S_head is not None else 0.0)
    ) / max(1e-6, w_sum)

    scores = 100.0 * np.clip(sim, 0.0, 1.0)
    return scores.astype(np.float32)


def _safe_phase_labels_from_result(
    result: Any,
    T: int,
) -> Optional[List[str]]:
    """
    Best-effort extraction of per-frame phase labels from phase_detect result.
    Returns:
      list[str] of length T or None if incompatible.
    """
    if result is None:
        return None

    # If PhaseResult-like with .labels
    if hasattr(result, "labels"):
        labels = getattr(result, "labels")
    else:
        labels = result

    if isinstance(labels, (list, tuple)):
        if len(labels) == T:
            return [str(x) for x in labels]
        return None

    if isinstance(labels, np.ndarray):
        if labels.shape[0] == T:
            return [str(x) for x in labels.tolist()]
        return None

    return None


# ==================== Core build logic ====================

def build_frame_table(
    amateur_npz: str,
    pro_npz: str,
    out_csv: str,
    align_feature: str = "ankle_y",
    include_head: bool = True,
) -> None:
    # ---- Load pose-tracks ----
    am_P, am_V, am_fps, _ = load_posetrack_npz(amateur_npz)
    pr_P, pr_V, pr_fps, _ = load_posetrack_npz(pro_npz)

    print(
        f"[INFO] Loaded amateur: {amateur_npz} shape={am_P.shape} fps={am_fps:.3f}\n"
        f"[INFO] Loaded pro:     {pro_npz} shape={pr_P.shape} fps={pr_fps:.3f}"
    )

    # Raw missing
    raw_miss_am = _missing_pct(am_P)
    raw_miss_pr = _missing_pct(pr_P)
    print(
        f"[INFO] Raw missing data: Amateur {raw_miss_am:.2f}% | "
        f"Pro {raw_miss_pr:.2f}%"
    )

    # ---- Align using compare_metrics DTW helper ----
    # This returns aligned subsequences in XYZ torso/world units.
    am_win_xyz, pr_win_xyz, ali = cm_align_sequences_dtw(
        am_P, pr_P, feature=align_feature
    )
    T, J, _ = am_win_xyz.shape
    assert pr_win_xyz.shape[0] == T and pr_win_xyz.shape[1] == J

    print(
        f"[INFO] Alignment mode={ali.mode} feature={ali.feature} "
        f"T={T}"
    )

    # Post-align missing
    post_miss_am = _missing_pct(am_win_xyz)
    post_miss_pr = _missing_pct(pr_win_xyz)
    print(
        f"[INFO] Post-align missing data: Amateur {post_miss_am:.2f}% | "
        f"Pro {post_miss_pr:.2f}%"
    )

    # ---- Normalize to pelvis-centered, torso length ~1 (compare_metrics) ----
    am_xy, am_scales = cm_normalize_pose_sequence(am_win_xyz)
    pr_xy, pr_scales = cm_normalize_pose_sequence(pr_win_xyz)

    # ---- Phase detection (optional, on aligned windows) ----
    phase_am = None
    phase_pr = None

    if PHASE_DETECT_AVAILABLE and detect_phases_from_pose is not None:
        try:
            res_am = detect_phases_from_pose(am_win_xyz, am_fps)
            phase_am = _safe_phase_labels_from_result(res_am, T)
        except Exception as e:
            print(f"[WARN] Amateur phase detection failed: {e}")
            phase_am = None

        try:
            res_pr = detect_phases_from_pose(pr_win_xyz, pr_fps)
            phase_pr = _safe_phase_labels_from_result(res_pr, T)
        except Exception as e:
            print(f"[WARN] Pro phase detection failed: {e}")
            phase_pr = None
    else:
        print("[INFO] Phase detection not available; phase columns will be empty.")

    # ---- Frame-level core series (reuse compare_metrics) ----
    # Work on normalized XY.
    pitch_am = cm_torso_pitch_deg_series(am_xy)
    pitch_pr = cm_torso_pitch_deg_series(pr_xy)
    head_am = cm_head_pitch_deg_series(am_xy)
    head_pr = cm_head_pitch_deg_series(pr_xy)
    knee_am = cm_knee_mean_deg_series(am_xy)
    knee_pr = cm_knee_mean_deg_series(pr_xy)

    # Use raw XYZ for vertical proxies
    ankle_y_am = cm_series_ankle_y(am_win_xyz)
    ankle_y_pr = cm_series_ankle_y(pr_win_xyz)
    hip_y_am = cm_series_hip_y(am_win_xyz)
    hip_y_pr = cm_series_hip_y(pr_win_xyz)

    # Instantaneous stance / span / asym / leg-axis on normalized XY
    stance_am = np.array(
        [cm_stance_width_pct_of_hip(am_xy[t]) for t in range(T)],
        dtype=float,
    )
    stance_pr = np.array(
        [cm_stance_width_pct_of_hip(pr_xy[t]) for t in range(T)],
        dtype=float,
    )

    hand_span_am = np.array(
        [cm_hand_span_pct_of_torso(am_xy[t]) for t in range(T)],
        dtype=float,
    )
    hand_span_pr = np.array(
        [cm_hand_span_pct_of_torso(pr_xy[t]) for t in range(T)],
        dtype=float,
    )

    hand_asym_am = np.array(
        [cm_hand_asym_pct_of_torso(am_xy[t]) for t in range(T)],
        dtype=float,
    )
    hand_asym_pr = np.array(
        [cm_hand_asym_pct_of_torso(pr_xy[t]) for t in range(T)],
        dtype=float,
    )

    leg_axis_am = np.array(
        [cm_leg_axis_angle_deg(am_xy[t]) for t in range(T)],
        dtype=float,
    )
    leg_axis_pr = np.array(
        [cm_leg_axis_angle_deg(pr_xy[t]) for t in range(T)],
        dtype=float,
    )

    # ---- Per-frame similarity score (0–100) ----
    frame_scores = _compute_frame_similarity_scores(
        am_xy,
        pr_xy,
        include_head=include_head,
    )

    # ---- CSV columns: header definition ----

    # Core per-frame summary metrics
    base_cols = [
        "frame",
        "score",                    # similarity score 0..100
        "am_phase",
        "pr_phase",
        "am_ankle_y",
        "pr_ankle_y",
        "am_hip_y",
        "pr_hip_y",
        "am_pitch_deg",
        "pr_pitch_deg",
        "am_head_pitch_deg",
        "pr_head_pitch_deg",
        "am_knee_mean_deg",
        "pr_knee_mean_deg",
        "am_stance_width_pct_hip",
        "pr_stance_width_pct_hip",
        "am_hand_span_pct_torso",
        "pr_hand_span_pct_torso",
        "am_hand_asym_pct_torso",
        "pr_hand_asym_pct_torso",
        "am_leg_axis_deg",
        "pr_leg_axis_deg",
    ]

    # Per-joint normalized coords + deltas
    joint_cols: List[str] = []
    for j in range(J):
        jname = JOINT_NAME_BY_INDEX.get(j, f"joint_{j}")
        joint_cols.extend([
            f"am_{jname}_x",
            f"am_{jname}_y",
            f"pr_{jname}_x",
            f"pr_{jname}_y",
            f"d_{jname}_x",
            f"d_{jname}_y",
            f"d_{jname}_norm",
        ])

    header = base_cols + joint_cols

    # ---- Write CSV + track numeric missing ----
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    n_rows = 0
    numeric_total = 0
    numeric_missing = 0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for t in range(T):
            am_phase_t = phase_am[t] if phase_am is not None else ""
            pr_phase_t = phase_pr[t] if phase_pr is not None else ""

            row: List[Any] = [
                t,
                float(frame_scores[t]),
                am_phase_t,
                pr_phase_t,
                float(ankle_y_am[t]) if np.isfinite(ankle_y_am[t]) else np.nan,
                float(ankle_y_pr[t]) if np.isfinite(ankle_y_pr[t]) else np.nan,
                float(hip_y_am[t]) if np.isfinite(hip_y_am[t]) else np.nan,
                float(hip_y_pr[t]) if np.isfinite(hip_y_pr[t]) else np.nan,
                float(pitch_am[t]) if np.isfinite(pitch_am[t]) else np.nan,
                float(pitch_pr[t]) if np.isfinite(pitch_pr[t]) else np.nan,
                float(head_am[t]) if np.isfinite(head_am[t]) else np.nan,
                float(head_pr[t]) if np.isfinite(head_pr[t]) else np.nan,
                float(knee_am[t]) if np.isfinite(knee_am[t]) else np.nan,
                float(knee_pr[t]) if np.isfinite(knee_pr[t]) else np.nan,
                float(stance_am[t]) if np.isfinite(stance_am[t]) else np.nan,
                float(stance_pr[t]) if np.isfinite(stance_pr[t]) else np.nan,
                float(hand_span_am[t]) if np.isfinite(hand_span_am[t]) else np.nan,
                float(hand_span_pr[t]) if np.isfinite(hand_span_pr[t]) else np.nan,
                float(hand_asym_am[t]) if np.isfinite(hand_asym_am[t]) else np.nan,
                float(hand_asym_pr[t]) if np.isfinite(hand_asym_pr[t]) else np.nan,
                float(leg_axis_am[t]) if np.isfinite(leg_axis_am[t]) else np.nan,
                float(leg_axis_pr[t]) if np.isfinite(leg_axis_pr[t]) else np.nan,
            ]

            # Per-joint XY + deltas (normalized coords)
            for j in range(J):
                ax, ay = am_xy[t, j, 0], am_xy[t, j, 1]
                px, py = pr_xy[t, j, 0], pr_xy[t, j, 1]
                if np.isfinite(ax) and np.isfinite(ay) and np.isfinite(px) and np.isfinite(py):
                    dx = ax - px
                    dy = ay - py
                    dn = float(np.sqrt(dx * dx + dy * dy))
                    row.extend([float(ax), float(ay), float(px), float(py),
                                float(dx), float(dy), dn])
                else:
                    row.extend([np.nan, np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan])

            w.writerow(row)
            n_rows += 1

            # Track numeric missing
            for val in row:
                if isinstance(val, (int, float, np.generic)):
                    numeric_total += 1
                    if not np.isfinite(float(val)):
                        numeric_missing += 1

    print(f"[INFO] Wrote {n_rows} rows to {out_csv}")

    if numeric_total > 0:
        csv_missing_pct = 100.0 * float(numeric_missing) / float(numeric_total)
        print(
            f"[INFO] CSV numeric missing data: {csv_missing_pct:.2f}% "
            f"(counts all per-frame numeric cells, NaN as missing)"
        )
    else:
        print("[WARN] No numeric cells accounted when computing CSV missing %.")


# ==================== CLI ====================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("amateur_npz", help="Amateur pose-track NPZ")
    ap.add_argument("pro_npz", help="Pro (reference) pose-track NPZ")
    ap.add_argument(
        "--align_feature",
        choices=["ankle_y", "hip_y", "pitch"],
        default="ankle_y",
        help="Proxy feature for DTW alignment inside compare_metrics.",
    )
    ap.add_argument(
        "--out_csv",
        default="viz/frame_table.csv",
        help="Output CSV path",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    build_frame_table(
        amateur_npz=args.amateur_npz,
        pro_npz=args.pro_npz,
        out_csv=args.out_csv,
        align_feature=args.align_feature,
    )


if __name__ == "__main__":
    main()
