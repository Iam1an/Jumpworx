#!/usr/bin/env python3
"""
JWCore — feedback_generator.py (lean, dependency-light)

Purpose
-------
Turn pose-tracks (NPZ) and/or feature JSONs into a compact feedback report.
Designed to compose with the MVP pipeline you've built:

  extract_keypoints.py  → cache/<clip>.posetrack.npz
  jwcore.pose_extract   → features/<clip>.json
  jwcore.trick_classifier → label/confidence
  scripts/viz_2d.py     → viz overlay

What this file keeps from your older version
--------------------------------------------
✓ Banded DTW fallback (no SciPy required)
✓ Analytics normalization = pelvis translate + torso scale (no XY rotation)
✓ Posture angles & height(+up) signals
✓ Simple rubric → posture/timing/airtime/overall scores

What it removes (by design for MVP)
-----------------------------------
✗ MediaPipe video inference (we use cached NPZ instead)
✗ Hard dependency on OpenCV/MediaPipe/OpenAI
✗ LLM call (still supported via a tiny hook, but off by default)

CLI
---
Single-file mode:
  python -m jwcore.feedback_generator \
    --student_npz cache/stu.posetrack.npz \
    --expert_npz  cache/pro.posetrack.npz \
    --student_features features/stu.json \
    --expert_features  features/pro.json \
    --label backflip \
    --out_dir feedback/

You can pass only NPZs (we'll compute features we need), or only features
(airtime delta then won't be available), or both (best).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# JWCore deps
from jwcore.pose_utils import FEATURE_KEYS, extract_trick_features
from jwcore.normalize import normalize_prerot
try:
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2
    _HAS_PHASE = True
except Exception:
    segment_phases_with_airtime_v2 = None
    _HAS_PHASE = False

# --------------------------------------------------------------------------------------
# Indices (BlazePose style) — keep consistent with the rest of JWCore
# --------------------------------------------------------------------------------------
L_SHO, R_SHO = 11, 12
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
L_FTO, R_FTO = 31, 32

# --------------------------------------------------------------------------------------
# Small helpers (kept close; NaN‑safe)
# --------------------------------------------------------------------------------------

def _forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for t in range(1, out.shape[0]):
        bad = ~np.isfinite(out[t])
        if bad.any():
            out[t][bad] = out[t-1][bad]
    return out


def _normalize_for_analytics(kps_pix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pelvis translate + torso scale; NO XY rotation (as in your original)."""
    T = kps_pix.shape[0]
    kps = np.dstack([kps_pix, np.zeros((T, kps_pix.shape[1]), dtype=np.float32)])  # (T,33,3)
    kps = _forward_fill_nan(kps)
    pelvis = 0.5 * (kps[:, L_HIP, :] + kps[:, R_HIP, :])
    shctr = 0.5 * (kps[:, L_SHO, :] + kps[:, R_SHO, :])
    kps -= pelvis[:, None, :]
    torso = np.linalg.norm(shctr, axis=1) + eps
    vals = torso[np.isfinite(torso)]
    scale = np.median(vals) if vals.size else 1.0
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    kps /= scale
    return kps[:, :, :2]


def _angle2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        return float("nan")
    ba, bc = a - b, c - b
    nba = np.linalg.norm(ba) + 1e-6
    nbc = np.linalg.norm(bc) + 1e-6
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosv)))))


def _frame_angles(pts: np.ndarray) -> np.ndarray:
    return np.array([
        _angle2(pts[L_SHO, :2], pts[L_ELB, :2], pts[L_WRI, :2]),
        _angle2(pts[R_SHO, :2], pts[R_ELB, :2], pts[R_WRI, :2]),
        _angle2(pts[L_HIP, :2], pts[L_KNE, :2], pts[L_ANK, :2]),
        _angle2(pts[R_HIP, :2], pts[R_KNE, :2], pts[R_ANK, :2]),
        _angle2(pts[L_ELB, :2], pts[L_SHO, :2], pts[L_HIP, :2]),
        _angle2(pts[R_ELB, :2], pts[R_SHO, :2], pts[R_HIP, :2]),
    ], dtype=np.float32)


def _height_series_up(kps_norm: np.ndarray) -> np.ndarray:
    toes_y = np.nanmean(kps_norm[:, [L_FTO, R_FTO], 1], axis=1)
    if not np.isfinite(toes_y).any():
        toes_y = np.nanmean(kps_norm[:, [L_ANK, R_ANK], 1], axis=1)
    baseline = np.nanmedian(toes_y)
    return baseline - toes_y

# --------------------------------------------------------------------------------------
# DTW (banded fallback)
# --------------------------------------------------------------------------------------

def _dtw_path_band(a: np.ndarray, b: np.ndarray, band_frac: float = 0.1) -> np.ndarray:
    A, B = len(a), len(b)
    w = int(max(1, band_frac * max(A, B)))
    inf = np.float32(np.inf)
    D = np.full((A + 1, B + 1), inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, A + 1):
        j_lo = max(1, i - w)
        j_hi = min(B, i + w)
        ai = a[i - 1]
        for j in range(j_lo, j_hi + 1):
            c = (ai - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j = A, B
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
        i, j = min(candidates, key=lambda s: D[s[0], s[1]])
    path.reverse()
    return np.asarray(path, dtype=int)

# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

@dataclass
class RubricWeights:
    posture: float = 0.60
    timing: float = 0.25
    airtime: float = 0.15

@dataclass
class ScoreBreakdown:
    posture_mae_deg: float
    timing_cost: float
    airtime_delta_s: float | None
    posture_score: float
    timing_score: float
    airtime_score: float
    overall: float
    path_len: int

# --------------------------------------------------------------------------------------
# Core API
# --------------------------------------------------------------------------------------

def compute_breakdown_from_npz(stu_npz_path: str, pro_npz_path: str, band_frac: float = 0.1) -> Tuple[ScoreBreakdown, np.ndarray, Dict]:
    """Compute scores + DTW path using pose-tracks (pixel-space)."""
    ds = np.load(stu_npz_path, allow_pickle=True)
    dp = np.load(pro_npz_path, allow_pickle=True)
    kpsS = ds["kps_xy"]; fpsS = float(ds["fps"]) if "fps" in ds else 30.0
    kpsP = dp["kps_xy"]; fpsP = float(dp["fps"]) if "fps" in dp else 30.0

    An = _normalize_for_analytics(kpsS)
    Bn = _normalize_for_analytics(kpsP)

    # DTW on height(+up)
    hA = _height_series_up(An)
    hB = _height_series_up(Bn)
    path = _dtw_path_band(hA, hB, band_frac=band_frac)
    idxA, idxB = path[:, 0], path[:, 1]

    # Posture MAE (angles)
    angA = np.array([_frame_angles(An[i]) for i in idxA])
    angB = np.array([_frame_angles(Bn[j]) for j in idxB])
    posture_mae = float(np.nanmean(np.abs(angA - angB)))

    # Timing cost (MSE of height(+up))
    timing_cost = float(np.nanmean((hA[idxA] - hB[idxB]) ** 2))

    # Airtime delta via phases (optional, using normalized-then-xyz augmentation)
    airtime_delta = None
    if _HAS_PHASE:
        def to3(k2):
            T = k2.shape[0]
            return np.dstack([k2, np.zeros((T, k2.shape[1]), dtype=np.float32)])
        pA = segment_phases_with_airtime_v2(to3(An), fpsS, require_precontact=False)
        pB = segment_phases_with_airtime_v2(to3(Bn), fpsP, require_precontact=False)
        if pA.airtime_seconds is not None and pB.airtime_seconds is not None:
            airtime_delta = abs(float(pA.airtime_seconds) - float(pB.airtime_seconds))

    # Score map
    posture_score = 100.0 * (1.0 - min(60.0, posture_mae) / 60.0)
    timing_score = 100.0 * (1.0 - min(0.0625, timing_cost) / 0.0625)
    airtime_score = 100.0 if airtime_delta is None else 100.0 * (1.0 - min(0.5, airtime_delta) / 0.5)
    overall = RubricWeights().posture * posture_score + RubricWeights().timing * timing_score + RubricWeights().airtime * airtime_score

    breakdown = ScoreBreakdown(
        posture_mae_deg=float(posture_mae),
        timing_cost=float(timing_cost),
        airtime_delta_s=None if airtime_delta is None else float(airtime_delta),
        posture_score=float(posture_score),
        timing_score=float(timing_score),
        airtime_score=float(airtime_score),
        overall=float(overall),
        path_len=int(len(path)),
    )

    diagnostics = dict(
        fps_student=float(fpsS), fps_expert=float(fpsP),
        n_frames_student=int(kpsS.shape[0]), n_frames_expert=int(kpsP.shape[0]),
    )
    return breakdown, path, diagnostics


def summarize_deltas_from_path(stu_npz_path: str, pro_npz_path: str, path: np.ndarray) -> Dict:
    ds = np.load(stu_npz_path, allow_pickle=True)
    dp = np.load(pro_npz_path, allow_pickle=True)
    A = _normalize_for_analytics(ds["kps_xy"])
    B = _normalize_for_analytics(dp["kps_xy"])
    idxA, idxB = path[:, 0], path[:, 1]

    def knee_to_chest(kps):
        chest = 0.5 * (kps[:, L_SHO, 1] + kps[:, R_SHO, 1])
        knee = 0.5 * (kps[:, L_KNE, 1] + kps[:, R_KNE, 1])
        return knee - chest

    n = len(idxA)
    early = slice(0, max(1, int(0.2 * n)))
    def hand_height_offset(kps):
        lw = kps[:, L_WRI, 1] - kps[:, L_SHO, 1]
        rw = kps[:, R_WRI, 1] - kps[:, R_SHO, 1]
        return float(np.nanmedian(0.5 * (lw + rw)))

    A_aligned = A[idxA]
    B_aligned = B[idxB]
    hands_A = hand_height_offset(A_aligned[early])
    hands_B = hand_height_offset(B_aligned[early])

    a_dist = knee_to_chest(A_aligned)
    b_dist = knee_to_chest(B_aligned)
    a_tuck = int(np.nanargmin(a_dist)) if np.isfinite(a_dist).any() else -1
    b_tuck = int(np.nanargmin(b_dist)) if np.isfinite(b_dist).any() else -1
    tuck_delta = None if a_tuck < 0 or b_tuck < 0 else int(a_tuck - b_tuck)

    def knee_angle_seq(pts):
        return np.array([_angle2(p[L_HIP, :2], p[L_KNE, :2], p[L_ANK, :2]) for p in pts])

    A_end = knee_angle_seq(A_aligned[int(0.9 * n):])
    B_end = knee_angle_seq(B_aligned[int(0.9 * n):])
    land_knee_A = float(np.nanmedian(A_end)) if A_end.size else float("nan")
    land_knee_B = float(np.nanmedian(B_end)) if B_end.size else float("nan")

    return dict(
        hands_start_offset=f"{hands_A:+.3f} vs expert {hands_B:+.3f} (neg = higher hands)",
        tuck_timing_delta_frames=None if tuck_delta is None else int(tuck_delta),
        landing_knee_deg=f"{land_knee_A:.1f} vs expert {land_knee_B:.1f}",
    )

# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def _write_reports(out_dir: str, breakdown: ScoreBreakdown, diagnostics: Dict, deltas: Dict, label: Optional[str]) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "feedback_report.json")
    md_path = os.path.join(out_dir, "feedback_report.md")

    data = dict(
        label=label,
        scores=asdict(breakdown),
        diagnostics=diagnostics,
        deltas=deltas,
    )
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    md = []
    md.append(f"# Trick Feedback\n")
    md.append(f"**Label:** {label or 'unknown'}\n")
    md.append(f"**Overall:** {breakdown.overall:.1f}\n")
    md.append(f"- Posture: {breakdown.posture_score:.1f} (MAE {breakdown.posture_mae_deg:.1f}°)")
    md.append(f"- Timing:  {breakdown.timing_score:.1f} (DTW {breakdown.timing_cost:.4f})")
    if breakdown.airtime_delta_s is not None:
        md.append(f"- Airtime: {breakdown.airtime_score:.1f} (Δ {breakdown.airtime_delta_s:.3f}s)")
    else:
        md.append(f"- Airtime: n/a")
    md.append("\n**Key deltas:**")
    for k, v in deltas.items():
        md.append(f"- {k}: {v}")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    return json_path, md_path

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute posture/timing/airtime scores and produce a feedback report.")
    p.add_argument("--student_npz", type=str, help="Student pose-track .npz from extract_keypoints.py")
    p.add_argument("--expert_npz", type=str, help="Expert pose-track .npz from extract_keypoints.py")
    p.add_argument("--student_features", type=str, help="Student feature JSON from jwcore.pose_extract")
    p.add_argument("--expert_features", type=str, help="Expert feature JSON from jwcore.pose_extract")
    p.add_argument("--label", type=str, help="Classifier label to include in the report")
    p.add_argument("--out_dir", type=str, default="feedback", help="Output folder for feedback_report.*")
    p.add_argument("--band_frac", type=float, default=0.1, help="DTW warping band fraction")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    # Prefer NPZ for rich metrics; fall back to features-only delta
    if args.student_npz and args.expert_npz:
        breakdown, path, diagnostics = compute_breakdown_from_npz(args.student_npz, args.expert_npz, band_frac=float(args.band_frac))
        deltas = summarize_deltas_from_path(args.student_npz, args.expert_npz, path)
    else:
        # Features-only path (no DTW/angles); compute simple airtime/height deltas if both provided
        def _load_feats(p):
            with open(p) as f:
                return json.load(f)["features"]
        if not (args.student_features and args.expert_features):
            raise SystemExit("Need either both NPZs or both feature JSONs.")
        sf, pf = _load_feats(args.student_features), _load_feats(args.expert_features)
        airtime_delta = abs(float(sf.get("airtime_s", 0.0)) - float(pf.get("airtime_s", 0.0)))
        breakdown = ScoreBreakdown(
            posture_mae_deg=float("nan"),
            timing_cost=float("nan"),
            airtime_delta_s=float(airtime_delta),
            posture_score=50.0,  # unknown in features-only mode
            timing_score=50.0,
            airtime_score=100.0 * (1.0 - min(0.5, airtime_delta) / 0.5),
            overall=50.0,  # neutral
            path_len=0,
        )
        diagnostics = dict(fps_student=None, fps_expert=None, n_frames_student=None, n_frames_expert=None)
        deltas = {
            "airtime_delta_s": round(airtime_delta, 3),
            "height_max_delta": round(float(sf.get("height_max", 0.0)) - float(pf.get("height_max", 0.0)), 3),
            "rotation_sign": f"student {sf.get('rotation_sign', 'n/a')} vs expert {pf.get('rotation_sign', 'n/a')}",
        }

    json_path, md_path = _write_reports(args.out_dir, breakdown, diagnostics, deltas, args.label)
    print(f"[save] {json_path}")
    print(f"[save] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
