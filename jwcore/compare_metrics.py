#!/usr/bin/env python3
# compare_metrics.py
#
# Shared pose-track comparison utilities for:
#   - visualization (side-by-side)
#   - coaching (LLM + rule-based)
#
# Responsibilities:
#   - align amateur vs pro pose sequences in time
#   - normalize coordinates (pelvis-centered, torso-scale)
#   - compute joint-wise deltas over time
#   - summarize into a compact set of scalar metrics suitable for coaching_thresholds.py
#
# This module is intentionally IO-free (no video, no CLI, no printing).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

from jwcore.joints import (
    NOSE,
    LEFT_EYE, RIGHT_EYE,
    LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    JOINT_NAME_BY_INDEX,
)

from jwcore.coaching_thresholds import TAU  # for τ-normalized phase scores


# ========= Small dataclasses =========

@dataclass
class AlignmentInfo:
    mode: str
    feature: str
    T: int
    note: str = ""


# ========= Core helpers =========

def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if not np.isnan(y).any():
        return y
    idx = np.arange(len(y))
    m = np.isfinite(y)
    if not m.any():
        return np.zeros_like(y)
    y[~m] = np.interp(idx[~m], idx[m], y[m])
    return y


def _series_ankle_y(arr: np.ndarray) -> np.ndarray:
    pair = arr[:, [LEFT_ANKLE, RIGHT_ANKLE], 1]
    finite_any = np.isfinite(pair).any(axis=1)
    out = np.full(arr.shape[0], np.nan, float)
    with np.errstate(invalid="ignore"):
        out[finite_any] = np.nanmean(pair[finite_any], axis=1)
    return _interp_nan_1d(out)


def _series_hip_y(arr: np.ndarray) -> np.ndarray:
    pair = arr[:, [LEFT_HIP, RIGHT_HIP], 1]
    finite_any = np.isfinite(pair).any(axis=1)
    out = np.full(arr.shape[0], np.nan, float)
    with np.errstate(invalid="ignore"):
        out[finite_any] = np.nanmean(pair[finite_any], axis=1)
    return _interp_nan_1d(out)


def _torso_pitch_deg_frame(frame_xy: np.ndarray) -> float:
    need = (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER)
    if not all(np.isfinite(frame_xy[i]).all() for i in need):
        return np.nan
    hip_mid = 0.5 * (frame_xy[LEFT_HIP] + frame_xy[RIGHT_HIP])
    sh_mid = 0.5 * (frame_xy[LEFT_SHOULDER] + frame_xy[RIGHT_SHOULDER])
    v = sh_mid - hip_mid
    up = np.array([0.0, -1.0], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.nan
    cosang = float(np.clip(np.dot(v / n, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))  # 0° = upright


def _torso_pitch_deg_series(xy_seq: np.ndarray) -> np.ndarray:
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        out[i] = _torso_pitch_deg_frame(xy_seq[i])
    return _interp_nan_1d(out)


def _knee_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _knee_mean_deg_series(xy_seq: np.ndarray) -> np.ndarray:
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        L = _knee_angle_deg(
            xy_seq[i, LEFT_HIP],
            xy_seq[i, LEFT_KNEE],
            xy_seq[i, LEFT_ANKLE],
        )
        R = _knee_angle_deg(
            xy_seq[i, RIGHT_HIP],
            xy_seq[i, RIGHT_KNEE],
            xy_seq[i, RIGHT_ANKLE],
        )
        vals = [v for v in (L, R) if np.isfinite(v)]
        out[i] = float(np.mean(vals)) if vals else np.nan
    return _interp_nan_1d(out)


def _head_midpoint(xy: np.ndarray) -> np.ndarray:
    if np.isfinite(xy[LEFT_EAR]).all() and np.isfinite(xy[RIGHT_EAR]).all():
        return 0.5 * (xy[LEFT_EAR] + xy[RIGHT_EAR])
    if np.isfinite(xy[LEFT_EYE]).all() and np.isfinite(xy[RIGHT_EYE]).all():
        return 0.5 * (xy[LEFT_EYE] + xy[RIGHT_EYE])
    if np.isfinite(xy[NOSE]).all():
        return xy[NOSE]
    return np.array([np.nan, np.nan], dtype=np.float32)


def _head_pitch_deg_series(xy_seq: np.ndarray) -> np.ndarray:
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    up = np.array([0.0, -1.0], dtype=np.float32)
    for i in range(T):
        xy = xy_seq[i]
        hm = _head_midpoint(xy)
        nose = xy[NOSE] if np.isfinite(xy[NOSE]).all() else None
        if nose is None or not np.isfinite(hm).all():
            continue
        v = nose - hm
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        cosang = float(np.clip(np.dot(v / n, up), -1.0, 1.0))
        out[i] = float(np.degrees(np.arccos(cosang)))
    return _interp_nan_1d(out)


def _pelvis_center_torso_scale(frame_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center at pelvis; scale so torso length ~= 1. Returns (xy_norm, scale)."""
    hips = frame_xy[[LEFT_HIP, RIGHT_HIP], :]
    shs = frame_xy[[LEFT_SHOULDER, RIGHT_SHOULDER], :]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return frame_xy.copy(), 1.0
    pelvis = hips.mean(axis=0)
    shmid = shs.mean(axis=0)
    span = float(np.linalg.norm(shmid - pelvis))
    scale = 1.0 if span < 1e-6 else (1.0 / span)
    return (frame_xy - pelvis) * scale, scale


def _normalize_pose_sequence(arr_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize sequence to pelvis-centered, torso-length ~ 1.
    Returns:
      xy_norm: (T,J,2)
      torso_scales: (T,) scale factors used (1/torso_len_pixels)
    """
    T = arr_xyz.shape[0]
    out = np.empty((T, arr_xyz.shape[1], 2), dtype=np.float32)
    scales = np.full(T, np.nan, float)
    for t in range(T):
        xy = arr_xyz[t, :, :2]
        xy_n, s = _pelvis_center_torso_scale(xy)
        out[t] = xy_n
        scales[t] = s
    return out, scales


# ========= DTW alignment (1D) =========

def _dtw_path_1d(a: np.ndarray, b: np.ndarray) -> List[Tuple[int, int]]:
    a = _interp_nan_1d(a)
    b = _interp_nan_1d(b)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf, float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = n, m
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = min(((i - 1, j), (i, j - 1), (i - 1, j - 1)), key=lambda ij: D[ij])
    path.reverse()
    return path


def _build_index_map_dtw(arrA: np.ndarray, arrB: np.ndarray, feature: str = "ankle_y") -> Dict[int, int]:
    if feature == "ankle_y":
        sA, sB = _series_ankle_y(arrA), _series_ankle_y(arrB)
    elif feature == "hip_y":
        sA, sB = _series_hip_y(arrA), _series_hip_y(arrB)
    elif feature == "pitch":
        sA = _torso_pitch_deg_series(arrA[:, :, :2])
        sB = _torso_pitch_deg_series(arrB[:, :, :2])
    else:
        raise ValueError(f"Unknown align feature: {feature}")
    path = _dtw_path_1d(sA, sB)
    mapA2B: Dict[int, int] = {}
    for i, j in path:
        if i not in mapA2B:
            mapA2B[i] = j
        else:
            # keep closer match
            if abs(sB[j] - sA[i]) < abs(sB[mapA2B[i]] - sA[i]):
                mapA2B[i] = j
    return mapA2B


def _align_sequences_dtw(
    am_xyz: np.ndarray,
    pro_xyz: np.ndarray,
    feature: str = "ankle_y",
) -> Tuple[np.ndarray, np.ndarray, AlignmentInfo]:
    """
    Returns aligned subsequences (am_win, pro_win) using DTW on a 1D proxy.
    """
    mapA2B = _build_index_map_dtw(am_xyz, pro_xyz, feature=feature)
    if not mapA2B:
        T = min(am_xyz.shape[0], pro_xyz.shape[0])
        return (
            am_xyz[:T],
            pro_xyz[:T],
            AlignmentInfo(
                mode="naive",
                feature=feature,
                T=T,
                note="DTW mapping empty; used naive min-length align.",
            ),
        )

    idxA = np.array(sorted(mapA2B.keys()), dtype=np.int32)
    idxB = np.array([mapA2B[int(i)] for i in idxA], dtype=np.int32)
    idxB = np.maximum.accumulate(idxB)
    idxB = np.clip(idxB, 0, pro_xyz.shape[0] - 1)

    am_win = am_xyz[idxA]
    pro_win = pro_xyz[idxB]
    T = am_win.shape[0]

    return (
        am_win,
        pro_win,
        AlignmentInfo(
            mode="dtw",
            feature=feature,
            T=int(T),
            note="DTW on 1D proxy with monotonic B indices.",
        ),
    )


# ========= Phase helpers =========

def _summarize_series_delta(delta: np.ndarray) -> Dict[str, float]:
    d = np.asarray(delta, float)
    with np.errstate(invalid="ignore"):
        return {
            "max": float(np.nanmax(np.abs(d))) if d.size else float("nan"),
            "mean": float(np.nanmean(np.abs(d))) if d.size else float("nan"),
        }


def _land_idx_heuristic(T: int) -> int:
    return max(0, int(0.85 * (T - 1)))  # last ~15%


def _midair_mask(T: int) -> np.ndarray:
    start = int(0.40 * T)
    end = max(start + 1, int(0.60 * T))
    m = np.zeros(T, bool)
    m[start:end] = True
    return m


def _stance_width_pct_of_hip(xy: np.ndarray) -> float:
    if not (np.isfinite(xy[LEFT_ANKLE]).all() and np.isfinite(xy[RIGHT_ANKLE]).all()):
        return float("nan")
    if not (np.isfinite(xy[LEFT_HIP]).all() and np.isfinite(xy[RIGHT_HIP]).all()):
        return float("nan")
    hip_w = float(np.linalg.norm(xy[LEFT_HIP] - xy[RIGHT_HIP]))
    if hip_w < 1e-6:
        return float("nan")
    stance = float(np.linalg.norm(xy[LEFT_ANKLE] - xy[RIGHT_ANKLE]))
    return 100.0 * (stance / hip_w)


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
    if not (
        np.isfinite(xy[LEFT_WRIST]).all()
        and np.isfinite(xy[RIGHT_WRIST]).all()
    ):
        return float("nan")
    dxL = float(xy[LEFT_WRIST, 0] - midx)
    dxR = float(xy[RIGHT_WRIST, 0] - midx)
    asym = abs(dxL + dxR)
    return 100.0 * (asym / torso)


def _apex_index_from_ankle_y(xy_seq: np.ndarray) -> int:
    if xy_seq.shape[1] <= max(LEFT_ANKLE, RIGHT_ANKLE):
        return 0
    y = 0.5 * (xy_seq[:, LEFT_ANKLE, 1] + xy_seq[:, RIGHT_ANKLE, 1])
    if not np.isfinite(y).any():
        return 0
    return int(np.nanargmin(y))


def _max_dev_pct_torso(am_xy_win: np.ndarray, pro_xy_win: np.ndarray, joint_ids: List[int]) -> float:
    if not joint_ids:
        return float("nan")
    Jmax = am_xy_win.shape[1]
    if any(j >= Jmax for j in joint_ids):
        return float("nan")
    diff = am_xy_win[:, joint_ids, :2] - pro_xy_win[:, joint_ids, :2]
    d = np.linalg.norm(diff, axis=2)  # (T, len(joint_ids))
    if not np.isfinite(d).any():
        return float("nan")
    val = float(np.nanmax(d))
    return 100.0 * val if np.isfinite(val) else float("nan")


def _leg_axis_angle_deg(frame_xy: np.ndarray) -> float:
    """Angle between hip→ankle line and hip→shoulder (torso) line."""
    need = (
        LEFT_HIP, RIGHT_HIP,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ANKLE, RIGHT_ANKLE,
    )
    if not all(np.isfinite(frame_xy[i]).all() for i in need):
        return float("nan")
    hip_mid = 0.5 * (frame_xy[LEFT_HIP] + frame_xy[RIGHT_HIP])
    sh_mid = 0.5 * (frame_xy[LEFT_SHOULDER] + frame_xy[RIGHT_SHOULDER])
    ank_mid = 0.5 * (frame_xy[LEFT_ANKLE] + frame_xy[RIGHT_ANKLE])
    v_torso = sh_mid - hip_mid
    v_leg = ank_mid - hip_mid
    nt = float(np.linalg.norm(v_torso))
    nl = float(np.linalg.norm(v_leg))
    if nt < 1e-6 or nl < 1e-6:
        return float("nan")
    cosang = float(np.clip(np.dot(v_torso / nt, v_leg / nl), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


# ========= Phase-level τ-normalized helper =========

def _phase_delta_ratio(
    series_am: np.ndarray,
    series_pr: np.ndarray,
    mask: np.ndarray,
    tau_key: str,
) -> float:
    """
    Signed how-many-τ difference between amateur and pro in a phase.
    >0 means amateur > pro; magnitude is |Δ| / τ.
    """
    if series_am is None or series_pr is None:
        return float("nan")
    if mask is None or mask.shape[0] != series_am.shape[0]:
        return float("nan")
    m = mask & np.isfinite(series_am) & np.isfinite(series_pr)
    if not m.any():
        return float("nan")
    d = float(np.nanmean(series_am[m] - series_pr[m]))
    tau = float(TAU.get(tau_key, 0.0))
    if not np.isfinite(tau) or tau <= 1e-6:
        return float("nan")
    return d / tau


# ========= Public API =========

def compare_metrics_from_xyz(
    amateur_xyz: np.ndarray,
    pro_xyz: np.ndarray,
    *,
    align_feature: str = "ankle_y",
) -> Dict[str, Any]:
    """
    Core entry: compute rich but compact metrics from raw 3D pose tracks.
    Expected shapes: (T, J, 3) with MediaPipe indexing.

    Returns dict with keys:
      - alignment
      - joints: per-joint delta summaries in torso units
      - scalars: scalar metrics (global) for coach
      - series: short time-series for key angles
      - joint_diffs: (T, J, 3) amateur - pro in torso units
      - torso_lengths: (T,)
      - phase_scores: τ-normalized per-phase deltas (compact, no frame firehose)
    """
    if amateur_xyz.ndim != 3 or pro_xyz.ndim != 3:
        raise ValueError("amateur_xyz and pro_xyz must be (T, J, 3) arrays")

    # 1) Align sequences in time
    am_win, pro_win, ali = _align_sequences_dtw(
        amateur_xyz, pro_xyz, feature=align_feature
    )

    # 2) Normalize to pelvis-centered, torso=1
    am_xy, am_scales = _normalize_pose_sequence(am_win)
    pro_xy, pro_scales = _normalize_pose_sequence(pro_win)
    T, J = am_xy.shape[0], am_xy.shape[1]

    # 3) Joint-wise summaries (torso units)
    joint_summaries: Dict[str, Dict[str, float]] = {}
    for j in range(J):
        name = JOINT_NAME_BY_INDEX.get(j, f"joint_{j}")
        d = np.linalg.norm(am_xy[:, j, :2] - pro_xy[:, j, :2], axis=1)
        joint_summaries[name] = _summarize_series_delta(d)

    # 4) Core series
    pitch_deg_am = _torso_pitch_deg_series(am_xy)
    pitch_deg_pro = _torso_pitch_deg_series(pro_xy)
    head_pitch_deg_am = _head_pitch_deg_series(am_xy)
    head_pitch_deg_pro = _head_pitch_deg_series(pro_xy)
    knee_mean_deg_am = _knee_mean_deg_series(am_xy)
    knee_mean_deg_pro = _knee_mean_deg_series(pro_xy)

    series: Dict[str, np.ndarray] = {
        "pitch_deg_am": pitch_deg_am,
        "pitch_deg_pro": pitch_deg_pro,
        "head_pitch_deg_am": head_pitch_deg_am,
        "head_pitch_deg_pro": head_pitch_deg_pro,
        "knee_mean_deg_am": knee_mean_deg_am,
        "knee_mean_deg_pro": knee_mean_deg_pro,
    }

    # 5) Global scalar metrics

    # Pitch total (deg → rad)
    pitch_total_deg_am = float(pitch_deg_am[-1] - pitch_deg_am[0])
    pitch_total_deg_pro = float(pitch_deg_pro[-1] - pitch_deg_pro[0])
    pitch_total_rad_delta = float(
        np.radians(pitch_total_deg_am - pitch_total_deg_pro)
    )

    # Midair / landing windows
    midair_mask = _midair_mask(T)
    land_idx = _land_idx_heuristic(T)

    def _midair_mean(fn) -> float:
        vals: List[float] = []
        for t in range(T):
            if not midair_mask[t]:
                continue
            v = fn(am_xy[t])
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    midair_span_pct = _midair_mean(_hand_span_pct_of_torso)
    midair_asym_pct = _midair_mean(_hand_asym_pct_of_torso)

    landing_xy = am_xy[min(max(0, land_idx), T - 1)]
    landing_stance_pct = float(_stance_width_pct_of_hip(landing_xy))

    # Apex-centered max deviations (ankles/hands) in torso units → % torso
    apex_am = _apex_index_from_ankle_y(am_xy)
    apex_pr = _apex_index_from_ankle_y(pro_xy)
    win_radius = (
        min(
            30,
            apex_am,
            apex_pr,
            T - 1 - apex_am,
            T - 1 - apex_pr,
        )
        if T > 2
        else 0
    )

    if win_radius > 0:
        length = 2 * win_radius + 1
        A_win = am_xy[apex_am - win_radius : apex_am - win_radius + length]
        B_win = pro_xy[apex_pr - win_radius : apex_pr - win_radius + length]
    else:
        A_win = am_xy
        B_win = pro_xy

    ankle_joint_ids: List[int] = []
    if LEFT_ANKLE < J and RIGHT_ANKLE < J:
        ankle_joint_ids += [LEFT_ANKLE, RIGHT_ANKLE]
    if LEFT_FOOT_INDEX < J and RIGHT_FOOT_INDEX < J:
        ankle_joint_ids += [LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]

    hand_joint_ids: List[int] = []
    if LEFT_WRIST < J and RIGHT_WRIST < J:
        hand_joint_ids += [LEFT_WRIST, RIGHT_WRIST]

    ankle_dev_pct = _max_dev_pct_torso(A_win, B_win, ankle_joint_ids)
    hand_dev_pct = _max_dev_pct_torso(A_win, B_win, hand_joint_ids)

    # Leg axis vs torso at apex
    leg_axis_am = (
        _leg_axis_angle_deg(am_xy[apex_am]) if 0 <= apex_am < T else float("nan")
    )
    leg_axis_pr = (
        _leg_axis_angle_deg(pro_xy[apex_pr]) if 0 <= apex_pr < T else float("nan")
    )
    leg_axis_diff_deg_apex = (
        float(leg_axis_am - leg_axis_pr)
        if np.isfinite(leg_axis_am) and np.isfinite(leg_axis_pr)
        else float("nan")
    )

    # Head early extension vs pro (global early-window summary; kept for compatibility)
    early_end = max(1, int(0.25 * T))
    if early_end > 1:
        am_head_early = float(np.nanmean(head_pitch_deg_am[:early_end]))
        pr_head_early = float(np.nanmean(head_pitch_deg_pro[:early_end]))
        head_early_pitch_lead_deg = (
            float(am_head_early - pr_head_early)
            if np.isfinite(am_head_early) and np.isfinite(pr_head_early)
            else float("nan")
        )
    else:
        head_early_pitch_lead_deg = float("nan")

    scalars: Dict[str, float] = {
        "midair_hand_span_pct_of_torso": midair_span_pct,
        "midair_hand_asym_pct_of_torso": midair_asym_pct,
        "landing_stance_width_pct_of_hip": landing_stance_pct,
        "pitch_total_rad": pitch_total_rad_delta,
        "ankle_dev_pct_of_torso_max": ankle_dev_pct,
        "hand_dev_pct_of_torso_max": hand_dev_pct,
        "leg_axis_diff_deg_apex": leg_axis_diff_deg_apex,
        "head_early_pitch_lead_deg": head_early_pitch_lead_deg,
    }

    # 6) Phase-level τ-normalized scores (compact, phase-aware)

    # Simple, aligned phase masks
    set_mask = np.zeros(T, bool)
    set_mask[: max(1, int(0.25 * T))] = True

    midair_mask = _midair_mask(T)

    landing_mask = np.zeros(T, bool)
    li = _land_idx_heuristic(T)
    landing_mask[max(0, li - int(0.1 * T)) : T] = True

    phase_scores: Dict[str, Dict[str, float]] = {
        # Early head throw in the set phase: positive = athlete leads pro
        "head_early_pitch_lead_deg": {
            "set": _phase_delta_ratio(
                head_pitch_deg_am,
                head_pitch_deg_pro,
                set_mask,
                "head_early_pitch_lead_deg",
            )
        },
        # Rotation profile midair vs pro, normalized by pitch_total_rad τ
        "pitch_profile": {
            "midair": _phase_delta_ratio(
                pitch_deg_am,
                pitch_deg_pro,
                midair_mask,
                "pitch_total_rad",
            )
        },
    }

    # 7) joint_diffs + torso_lengths
    joint_diffs = np.zeros((T, J, 3), dtype=np.float32)
    joint_diffs[:, :, :2] = am_xy - pro_xy
    torso_lengths = np.where(
        np.isfinite(am_scales) & (am_scales > 1e-6),
        1.0 / am_scales,
        np.nan,
    ).astype(np.float32)

    return {
        "alignment": {
            "mode": ali.mode,
            "feature": ali.feature,
            "frames_aligned": ali.T,
            "note": ali.note,
        },
        "joints": joint_summaries,
        "scalars": scalars,
        "series": series,
        "joint_diffs": joint_diffs,
        "torso_lengths": torso_lengths,
        "phase_scores": phase_scores,
    }
