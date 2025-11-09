#!/usr/bin/env python3
"""
inspect_compare_metrics.py

Debug / inspection tool for Jumpworx pose comparisons.

Shows:
  - Alignment info
  - Core scalar metrics (pre-τ)
  - Joint delta summaries (pre-τ, torso units)
  - Phase scores (post-τ, from compare_metrics)
  - Frame-level similarity scores (mirroring viz_compare_side_by_side)
  - Overall score (same formula as viz_compare_side_by_side)

Usage examples:

  # Single pair with JSON dump
  python -m scripts.inspect_compare_metrics \
      cache/TRICK26_BACKFLIP.posetrack.npz \
      cache/TRICK12_BACKFLIP.posetrack.npz \
      --align_feature ankle_y \
      --json_out viz/TRICK26_vs_TRICK12_inspect.json

  # Skip verbose joint list
  python -m scripts.inspect_compare_metrics \
      cache/TRICK26_BACKFLIP.posetrack.npz \
      cache/TRICK12_BACKFLIP.posetrack.npz \
      --no_joints
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, Tuple, List

import numpy as np

# ---------------------------------------------------------------------
# Make jwcore importable when this script is in ./scripts/
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.compare_metrics import compare_metrics_from_xyz
from jwcore.coaching_thresholds import TAU

# ========= Constants (mirror viz_compare_side_by_side) =========

# MediaPipe indices
NOSE = 0
L_EYE = 2
R_EYE = 5
L_EAR = 7
R_EAR = 8
L_SHO = 11
R_SHO = 12
L_ELB = 13
R_ELB = 14
L_WRI = 15
R_WRI = 16
L_HIP = 23
R_HIP = 24
L_KNE = 25
R_KNE = 26
L_ANK = 27
R_ANK = 28

# For scoring and labeling
KEY_NAMES = {
    L_SHO: "L_SHO",
    R_SHO: "R_SHO",
    L_ELB: "L_ELB",
    R_ELB: "R_ELB",
    L_WRI: "L_HAND",
    R_WRI: "R_HAND",
    L_HIP: "L_HIP",
    R_HIP: "R_HIP",
    L_KNE: "L_KNE",
    R_KNE: "R_KNE",
    L_ANK: "L_ANK",
    R_ANK: "R_ANK",
}

KEYS_BASE8 = [L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]
KEYS_EXPAND = [L_ELB, R_ELB, L_WRI, R_WRI]

# Scoring hyperparams (same as viz)
D0_JOINT = 0.15   # torso units
A0_PITCH = 20.0   # deg
A0_KNEE  = 25.0   # deg
A0_HEAD  = 15.0   # deg

W_JOINT = 1.00
W_PITCH = 0.60
W_KNEE  = 0.50
W_HEAD  = 0.60

# ========= Small helpers =========

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


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
    pair = arr[:, [L_ANK, R_ANK], 1]
    finite_any = np.isfinite(pair).any(axis=1)
    out = np.full(arr.shape[0], np.nan, float)
    with np.errstate(invalid="ignore"):
        out[finite_any] = np.nanmean(pair[finite_any], axis=1)
    return _interp_nan_1d(out)


def _torso_pitch_deg_frame(frame_xy: np.ndarray) -> float:
    need = (L_HIP, R_HIP, L_SHO, R_SHO)
    if not all(np.isfinite(frame_xy[i]).all() for i in need):
        return np.nan
    hip_mid = 0.5 * (frame_xy[L_HIP] + frame_xy[R_HIP])
    sh_mid = 0.5 * (frame_xy[L_SHO] + frame_xy[R_SHO])
    v = sh_mid - hip_mid
    up = np.array([0.0, -1.0], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.nan
    cosang = float(np.clip(np.dot(v / n, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _torso_pitch_deg_series(xy_seq: np.ndarray) -> np.ndarray:
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        out[i] = _torso_pitch_deg_frame(xy_seq[i])
    return _interp_nan_1d(out)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
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


def _knee_flex_deg_series(xy_seq: np.ndarray, side: str) -> np.ndarray:
    T = xy_seq.shape[0]
    out = np.full(T, np.nan, float)
    for i in range(T):
        if side == "L":
            out[i] = _angle_deg(xy_seq[i][L_HIP], xy_seq[i][L_KNE], xy_seq[i][L_ANK])
        else:
            out[i] = _angle_deg(xy_seq[i][R_HIP], xy_seq[i][R_KNE], xy_seq[i][R_ANK])
    return _interp_nan_1d(out)


def _head_midpoint(xy: np.ndarray) -> np.ndarray:
    if np.isfinite(xy[L_EAR]).all() and np.isfinite(xy[R_EAR]).all():
        return 0.5 * (xy[L_EAR] + xy[R_EAR])
    if np.isfinite(xy[L_EYE]).all() and np.isfinite(xy[R_EYE]).all():
        return 0.5 * (xy[L_EYE] + xy[R_EYE])
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
    hips = frame_xy[[L_HIP, R_HIP], :]
    shs = frame_xy[[L_SHO, R_SHO], :]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return frame_xy.copy(), 1.0
    pelvis = hips.mean(axis=0)
    shmid = shs.mean(axis=0)
    span = float(np.linalg.norm(shmid - pelvis))
    scale = 1.0 if span < 1e-6 else (1.0 / span)
    return (frame_xy - pelvis) * scale, scale


def _normalize_pose_sequence(arr_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = arr_xyz.shape[0]
    J = arr_xyz.shape[1]
    out = np.empty((T, J, 2), dtype=np.float32)
    scales = np.full(T, np.nan, float)
    for t in range(T):
        xy = arr_xyz[t, :, :2]
        xy_n, s = _pelvis_center_torso_scale(xy)
        out[t] = xy_n
        scales[t] = s
    return out, scales


# ========= DTW alignment (same logic style as compare_metrics) =========

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
        i, j = min(
            ((i - 1, j), (i, j - 1), (i - 1, j - 1)),
            key=lambda ij: D[ij],
        )
    path.reverse()
    return path


def _build_index_map_dtw(arrA: np.ndarray, arrB: np.ndarray, feature: str) -> Dict[int, int]:
    if feature == "ankle_y":
        sA, sB = _series_ankle_y(arrA), _series_ankle_y(arrB)
    elif feature == "hip_y":
        # simple hip-y series for completeness if needed
        pairA = arrA[:, [L_HIP, R_HIP], 1]
        pairB = arrB[:, [L_HIP, R_HIP], 1]
        sA = _interp_nan_1d(np.nanmean(pairA, axis=1))
        sB = _interp_nan_1d(np.nanmean(pairB, axis=1))
    elif feature == "pitch":
        sA = _torso_pitch_deg_series(arrA[:, :, :2])
        sB = _torso_pitch_deg_series(arrB[:, :, :2])
    else:
        raise ValueError(f"Unknown align feature: {feature}")
    path = _dtw_path_1d(sA, sB)
    m: Dict[int, int] = {}
    for i, j in path:
        if i not in m:
            m[i] = j
        else:
            if abs(sB[j] - sA[i]) < abs(sB[m[i]] - sA[i]):
                m[i] = j
    return m


def _align_sequences_dtw(
    am_xyz: np.ndarray,
    pro_xyz: np.ndarray,
    feature: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns aligned subsequences (am_win, pro_win) using DTW on a 1D proxy.
    Mirrors the logic used conceptually in compare_metrics/viz.
    """
    m = _build_index_map_dtw(am_xyz, pro_xyz, feature=feature)
    if not m:
        T = min(am_xyz.shape[0], pro_xyz.shape[0])
        return am_xyz[:T], pro_xyz[:T]

    idxA = np.array(sorted(m.keys()), dtype=np.int32)
    idxB = np.array([m[int(i)] for i in idxA], dtype=np.int32)
    idxB = np.maximum.accumulate(idxB)
    idxB = np.clip(idxB, 0, pro_xyz.shape[0] - 1)

    return am_xyz[idxA], pro_xyz[idxB]


# ========= Score computation (mirror viz) =========

def compute_frame_scores(
    am_xyz: np.ndarray,
    pro_xyz: np.ndarray,
    align_feature: str = "ankle_y",
    include_hands: bool = True,
    include_head: bool = True,
    ema_alpha: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Rebuild per-frame scores and overall_score using the same recipe
    as viz_compare_side_by_side.py.

    Returns:
      score_disp: (T,) per-frame scores after EMA (if any)
      overall_score: phase-aware blended scalar (0..100)
    """
    # 1) Align
    am_win, pro_win = _align_sequences_dtw(am_xyz, pro_xyz, feature=align_feature)

    # 2) Normalize
    Am, _ = _normalize_pose_sequence(am_win)
    Bm, _ = _normalize_pose_sequence(pro_win)
    T = Am.shape[0]

    # 3) Joint deltas (subset like viz)
    joints = list(KEYS_BASE8) + (KEYS_EXPAND if include_hands else [])
    delta = {}
    for j in joints:
        name = KEY_NAMES.get(j, f"J{j}")
        d = np.linalg.norm(Am[:, j, :2] - Bm[:, j, :2], axis=1)
        delta[name] = d

    # 4) Angles
    pitchA = _torso_pitch_deg_series(Am)
    pitchB = _torso_pitch_deg_series(Bm)
    d_pitch = np.abs(pitchA - pitchB)

    kneeLA = _knee_flex_deg_series(Am, "L")
    kneeLB = _knee_flex_deg_series(Bm, "L")
    kneeRA = _knee_flex_deg_series(Am, "R")
    kneeRB = _knee_flex_deg_series(Bm, "R")
    d_knee = 0.5 * (np.abs(kneeLA - kneeLB) + np.abs(kneeRA - kneeRB))

    if include_head:
        headA = _head_pitch_deg_series(Am)
        headB = _head_pitch_deg_series(Bm)
        d_head = np.abs(headA - headB)
    else:
        d_head = None

    # 5) Similarities
    joint_keys = list(delta.keys())
    S_joint = np.ones(T, dtype=np.float32)
    if joint_keys:
        mat = np.stack([delta[k] for k in joint_keys], axis=1)  # (T, J)
        with np.errstate(invalid="ignore"):
            Sj = np.exp(- (mat / D0_JOINT) ** 2)
            m = np.isfinite(Sj)
            valid = m.any(axis=1)
            S_joint = np.zeros(T, dtype=np.float32)
            S_joint[valid] = np.nanmean(Sj[valid], axis=1)

    S_pitch = np.exp(- (d_pitch / A0_PITCH) ** 2)
    S_knee = np.exp(- (d_knee / A0_KNEE) ** 2)
    if d_head is not None:
        S_head = np.exp(- (d_head / A0_HEAD) ** 2)
    else:
        S_head = None

    w_sum = W_JOINT + W_PITCH + W_KNEE + (W_HEAD if S_head is not None else 0.0)
    sim = (
        W_JOINT * S_joint
        + W_PITCH * S_pitch
        + W_KNEE * S_knee
        + ((W_HEAD * S_head) if S_head is not None else 0.0)
    ) / max(1e-6, w_sum)

    score_raw = 100.0 * np.clip(sim, 0.0, 1.0)

    # 6) EMA smoothing (like viz)
    alpha = float(max(0.0, min(1.0, ema_alpha)))
    if alpha > 0.0:
        score_disp = np.copy(score_raw)
        ema = score_disp[0] if (len(score_disp) and np.isfinite(score_disp[0])) else 0.0
        for i in range(len(score_disp)):
            x = score_disp[i] if np.isfinite(score_disp[i]) else ema
            ema = alpha * x + (1.0 - alpha) * ema
            score_disp[i] = ema
    else:
        score_disp = score_raw

    # 7) Overall score (same recipe you used: frame mean + τ phase scores)
    mean_disp = float(np.nanmean(score_disp)) if np.isfinite(np.nanmean(score_disp)) else 0.0
    S_frame = mean_disp / 100.0

    # We'll plug in phase_scores outside (from compare_metrics_from_xyz),
    # so here we just return score_disp; overall_score is computed in main().
    return score_disp, S_frame


def _phase_score_to_weight(ps_val: Any) -> float:
    if ps_val is None:
        return 1.0
    try:
        x = float(ps_val)
    except Exception:
        return 1.0
    if not np.isfinite(x):
        return 1.0
    # exp(-|Δ/τ|), same as viz logic
    return float(np.exp(-abs(x)))


# ========= Main CLI =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amateur_npz", help="Path to amateur .posetrack.npz")
    ap.add_argument("pro_npz", help="Path to pro .posetrack.npz")
    ap.add_argument(
        "--align_feature",
        choices=["ankle_y", "hip_y", "pitch"],
        default="ankle_y",
        help="Feature used for DTW alignment in this inspector + compare_metrics.",
    )
    ap.add_argument(
        "--include_hands",
        action="store_true",
        help="Include elbows/wrists in similarity score (matches viz behavior when enabled).",
    )
    ap.add_argument(
        "--include_head",
        action="store_true",
        help="Include head pitch in similarity score (matches viz behavior when enabled).",
    )
    ap.add_argument(
        "--ema_alpha",
        type=float,
        default=0.3,
        help="EMA alpha for frame scores (0 = off, e.g. side-by-side often uses 0.3).",
    )
    ap.add_argument(
        "--no_joints",
        action="store_true",
        help="Skip printing detailed per-joint summaries.",
    )
    ap.add_argument(
        "--json_out",
        default=None,
        help="Optional path to dump full compare_metrics + scores as JSON.",
    )
    args = ap.parse_args()

    # ---- Load pose tracks ----
    am_P, _, am_fps, _ = load_posetrack_npz(args.amateur_npz)
    pr_P, _, pr_fps, _ = load_posetrack_npz(args.pro_npz)

    print("== INPUT ==")
    print(f"Amateur: {args.amateur_npz}  shape={am_P.shape}  fps={am_fps:.3f}")
    print(f"Pro:     {args.pro_npz}  shape={pr_P.shape}  fps={pr_fps:.3f}")
    print()

    # ---- compare_metrics_from_xyz (gives scalars + phase_scores etc.) ----
    cm = compare_metrics_from_xyz(
        am_P,
        pr_P,
        align_feature=args.align_feature,
    )

    alignment = cm.get("alignment", {})
    scalars = cm.get("scalars", {})
    joints = cm.get("joints", {})
    phase_scores = cm.get("phase_scores", {})

    # ============================================================
    # 1) Alignment summary
    # ============================================================
    print("== ALIGNMENT (from compare_metrics_from_xyz) ==")
    print(json.dumps(alignment, indent=2))
    print()

    # ============================================================
    # 2) Core scalar metrics (pre-τ)
    # ============================================================
    print("== SCALARS (raw, before τ) ==")
    for k in sorted(scalars.keys()):
        v = scalars[k]
        if isinstance(v, float) and not np.isfinite(v):
            v_str = "nan"
        else:
            v_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"{k:40s}: {v_str}")
    print()

    # ============================================================
    # 3) Per-joint summaries (pre-τ, torso units)
    # ============================================================
    if not args.no_joints and joints:
        print("== JOINT DELTAS (|Δ| in torso units; sorted by max) ==")
        items = []
        for name, s in joints.items():
            mx = s.get("max", np.nan)
            mean = s.get("mean", np.nan)
            items.append((name, mx, mean))
        items.sort(key=lambda x: (-(x[1] if np.isfinite(x[1]) else -1e9)))
        for name, mx, mean in items:
            mx_s = f"{mx:.3f}" if np.isfinite(mx) else "nan"
            mn_s = f"{mean:.3f}" if np.isfinite(mean) else "nan"
            print(f"{name:24s}  max={mx_s:8s}  mean={mn_s:8s}")
        print()

    # ============================================================
    # 4) Phase scores (post-τ)
    # ============================================================
    print("== PHASE SCORES (after τ-normalization) ==")
    print("Each value is mean(am - pro) over phase, divided by TAU[metric].")
    print("Interpretation: ~0 = matches pro; +1 = 1τ worse; -1 = 1τ better.\n")

    for metric_key, phases in phase_scores.items():
        print(f"[{metric_key}]")
        for phase_name, val in phases.items():
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                v_str = "nan"
            else:
                v_str = f"{val:+.3f}"
            tau_val = TAU.get(metric_key, None)
            tau_str = f"{tau_val:.4f}" if isinstance(tau_val, (int, float)) else "None"
            print(f"  {phase_name:10s}: {v_str}   (τ={tau_str})")
        print()
    print()

    # ============================================================
    # 5) Frame scores + Overall score (mirror viz)
    # ============================================================
    score_disp, S_frame = compute_frame_scores(
        am_P,
        pr_P,
        align_feature=args.align_feature,
        include_hands=args.include_hands,
        include_head=args.include_head,
        ema_alpha=args.ema_alpha,
    )

    # Use phase_scores (τ-normalized) to blend like your viz
    pitch_midair_ps = phase_scores.get("pitch_profile", {}).get("midair", None)
    head_set_ps = phase_scores.get("head_early_pitch_lead_deg", {}).get("set", None)

    S = (
        0.6 * S_frame
        + 0.25 * _phase_score_to_weight(pitch_midair_ps)
        + 0.15 * _phase_score_to_weight(head_set_ps)
    )
    overall_score = 100.0 * max(0.0, min(1.0, S))

    finite = np.isfinite(score_disp)
    frame_mean = float(np.nanmean(score_disp[finite])) if finite.any() else float("nan")
    frame_min = float(np.nanmin(score_disp[finite])) if finite.any() else float("nan")
    frame_max = float(np.nanmax(score_disp[finite])) if finite.any() else float("nan")

    print("== FRAME SCORES (viz-style similarity) ==")
    print(f"Frames used:         {score_disp.shape[0]}")
    print(f"Frame score mean:    {frame_mean:.2f} / 100")
    print(f"Frame score min/max: {frame_min:.2f} – {frame_max:.2f}")
    print(f"Overall score:       {overall_score:.2f} / 100")
    print()

    # ============================================================
    # 6) Optional JSON dump
    # ============================================================
    if args.json_out:
        out = {
            "compare_metrics": _to_jsonable(cm),
            "frame_scores": _to_jsonable(score_disp),
            "frame_scores_summary": {
                "mean": frame_mean,
                "min": frame_min,
                "max": frame_max,
            },
            "overall_score": overall_score,
            "config": {
                "align_feature": args.align_feature,
                "include_hands": bool(args.include_hands),
                "include_head": bool(args.include_head),
                "ema_alpha": float(args.ema_alpha),
            },
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Full metrics + scores JSON written to: {args.json_out}")


if __name__ == "__main__":
    main()

