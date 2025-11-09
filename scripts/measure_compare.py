#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_compare.py

Compute time-standardized joint/angle differences between Amateur and Pro.

Now includes:
- Core lower/upper body joints (shoulders, hips, knees, ankles)
- Elbows, wrists (hands)
- Virtual HEAD point = midpoint(L_EAR, R_EAR)
- Head pitch angle delta (ear-mid → nose vs image vertical)

Outputs:
- CSV with per-sample (0..100%) deltas for selected joints + angles + Overall
- JSON summary with means/medians/maxes, time of max, optional phases
- Optional NPZ with aligned normalized sequences

Usage:
  python scripts/measure_compare.py \
    --npz_a cache/TRICK15_BACKFLIP.posetrack.npz \
    --npz_b cache/TRICK13_BACKFLIP.posetrack.npz \
    --align apex --seconds_before 1.0 --seconds_after 1.0 \
    --samples 101 --use_segmenter \
    --out_csv viz/measure_TRICK15_vs_TRICK13.csv \
    --out_json viz/measure_TRICK15_vs_TRICK13.json \
    --save_npz viz/measure_TRICK15_vs_TRICK13.npz
"""

import os, json, math, argparse, csv
import numpy as np

# ---- MediaPipe indices used elsewhere in your codebase ----
L_SHO, R_SHO = 11, 12
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16   # “hands” (wrists) for Pose
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28

NOSE = 0
L_EYE, R_EYE = 5, 6
L_EAR, R_EAR = 7, 8

# Base eight; expanded set adds elbows, wrists, head (virtual)
KEYS_BASE8 = [L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]
KEYS_EXPAND = [L_ELB, R_ELB, L_WRI, R_WRI]  # elbows + wrists

KEY_NAMES = {
    L_SHO:"L_SHO", R_SHO:"R_SHO", L_ELB:"L_ELB", R_ELB:"R_ELB",
    L_WRI:"L_HAND", R_WRI:"R_HAND",  # label as HAND for readability
    L_HIP:"L_HIP", R_HIP:"R_HIP", L_KNE:"L_KNE", R_KNE:"R_KNE", L_ANK:"L_ANK", R_ANK:"R_ANK",
    NOSE:"NOSE", L_EYE:"L_EYE", R_EYE:"R_EYE", L_EAR:"L_EAR", R_EAR:"R_EAR"
}

# Optional segmenter
SEGMENTER_AVAILABLE = False
try:
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2, Phases
    SEGMENTER_AVAILABLE = True
except Exception:
    segment_phases_with_airtime_v2 = None
    Phases = None

def load_pose_npz(path):
    d = np.load(path, allow_pickle=True)
    arr = d["kps_xyz"] if "kps_xyz" in d else None
    if arr is None:
        for k in ("pose","keypoints","landmarks","arr_0"):
            if k in d:
                arr = d[k]; break
    fps = float(np.asarray(d["fps"]).item()) if "fps" in d else None
    if arr is None or arr.ndim != 3 or arr.shape[1] < 33:
        raise ValueError(f"Bad pose in {path}: shape={getattr(arr,'shape',None)}")
    return arr.astype(np.float32), fps

def _interp_nan_1d(y):
    y = np.asarray(y, float)
    if not np.isnan(y).any(): return y
    idx = np.arange(len(y)); m = np.isfinite(y)
    y[~m] = np.interp(idx[~m], idx[m], y[m]) if m.any() else 0.0
    return y

def apex_index(arr):
    # use ankle y min (image coords: smaller y = higher)
    y = np.nanmean(arr[:, [L_ANK, R_ANK], 1], axis=1)
    y = _interp_nan_1d(y)
    return int(np.argmin(y))

def dtw_path_1d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    for s in (a, b):
        if np.isnan(s).any():
            idx = np.arange(len(s)); m = np.isfinite(s)
            s[~m] = np.interp(idx[~m], idx[m], s[m]) if m.any() else 0.0
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf); D[0,0] = 0.0
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = (ai - b[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    i, j = n, m; path = []
    while i>0 and j>0:
        path.append((i-1, j-1))
        candidates = [(i-1,j), (i,j-1), (i-1,j-1)]
        i, j = min(candidates, key=lambda ij: D[ij])
    path.reverse()
    return path

def series_ankle_y(arr):
    y = np.nanmean(arr[:, [L_ANK, R_ANK], 1], axis=1)
    return _interp_nan_1d(y)

def build_indices(arrA, arrB, fpsA, fpsB, align, seconds_before, seconds_after):
    if align == "apex":
        aA, aB = apex_index(arrA), apex_index(arrB)
        def win(center, fps, T):
            L = max(0, center - int(seconds_before*fps))
            R = min(T-1, center + int(seconds_after*fps))
            return L,R
        wA = win(aA, fpsA, arrA.shape[0]); wB = win(aB, fpsB, arrB.shape[0])
        lenA = wA[1]-wA[0]+1; lenB = wB[1]-wB[0]+1
        T = max(lenA, lenB)
        idxA = np.clip(np.arange(T), 0, lenA-1) + wA[0]
        idxB = np.clip(np.arange(T), 0, lenB-1) + wB[0]
        return idxA.astype(int), idxB.astype(int)
    elif align == "dtw":
        sA, sB = series_ankle_y(arrA), series_ankle_y(arrB)
        path = dtw_path_1d(sA, sB)
        mapA2B = {}
        for i,j in path:
            if i not in mapA2B: mapA2B[i]=j
        idxA = np.arange(len(sA)); idxB = np.array([mapA2B.get(int(i), 0) for i in idxA])
        return idxA.astype(int), np.clip(idxB, 0, len(sB)-1).astype(int)
    else:
        raise ValueError("For takeoff/landing alignment, run segmentation upstream and slice before calling this.")

def pelvis_center_torso_scale(frame_xy):
    # pelvis = mid-hip; shoulder mid
    hips = frame_xy[[L_HIP, R_HIP], :]
    shs  = frame_xy[[L_SHO, R_SHO], :]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()):
        return frame_xy.copy()
    pelvis = hips.mean(axis=0)
    shmid  = shs.mean(axis=0)
    span = np.linalg.norm(shmid - pelvis)
    scale = 1.0 if span < 1e-6 else (1.0/span)
    return (frame_xy - pelvis) * scale

def normalize_pose_sequence(arr):
    out = np.empty_like(arr[:, :, :2], dtype=np.float32)  # 2D normalized
    for t in range(arr.shape[0]):
        xy = arr[t, :, :2]
        with np.errstate(invalid="ignore"):
            out[t] = pelvis_center_torso_scale(xy)
    return out  # (T,33,2)

def resample_to_N(seq, N):
    # seq: (T, J, D)
    T = seq.shape[0]
    if T == N: return seq.copy()
    xi = np.linspace(0, T-1, num=N)
    x = np.arange(T)
    out = np.zeros((N, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for j in range(seq.shape[1]):
        for d in range(seq.shape[2]):
            y = seq[:, j, d]
            m = np.isfinite(y)
            if not m.any():
                out[:, j, d] = 0.0
            else:
                yf = y.copy()
                yf[~m] = np.interp(x[~m], x[m], y[m])
                out[:, j, d] = np.interp(xi, x, yf)
    return out

# ---- Virtual head point and angles ----
def head_midpoint(xy):
    # prefer ear midpoint; fallback to eye midpoint; fallback to nose
    if np.isfinite(xy[L_EAR]).all() and np.isfinite(xy[R_EAR]).all():
        return (xy[L_EAR] + xy[R_EAR]) / 2.0
    if np.isfinite(xy[L_EYE]).all() and np.isfinite(xy[R_EYE]).all():
        return (xy[L_EYE] + xy[R_EYE]) / 2.0
    if np.isfinite(xy[NOSE]).all():
        return xy[NOSE]
    return np.array([np.nan, np.nan], dtype=np.float32)

def head_pitch_deg(xy):
    """
    Angle between (ear-mid -> nose) vector and image up (0,-1).
    Smaller => head more upright; larger => tucked/tilted.
    """
    hm = head_midpoint(xy)
    nose = xy[NOSE] if np.isfinite(xy[NOSE]).all() else None
    if nose is None or not np.isfinite(hm).all():
        return np.nan
    v = nose - hm
    up = np.array([0.0, -1.0])
    n = np.linalg.norm(v)
    if n < 1e-6: return np.nan
    cosang = np.clip(np.dot(v/n, up), -1.0, 1.0)
    return math.degrees(math.acos(cosang))  # 0° = nose straight up from ear-midpoint

def angle_deg(a, b, c):
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return np.nan
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return np.nan
    cosang = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def torso_pitch_deg(xy):
    hips = xy[[L_HIP, R_HIP], :]; shs = xy[[L_SHO, R_SHO], :]
    if not (np.isfinite(hips).all() and np.isfinite(shs).all()): return np.nan
    pelvis = hips.mean(axis=0); shmid = shs.mean(axis=0)
    v = shmid - pelvis
    up = np.array([0.0, -1.0])  # image up
    n = np.linalg.norm(v);  cosang = np.dot(v, up) / (n + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))  # 0° upright

def knee_flex_deg(xy, side="L"):
    if side=="L":
        return angle_deg(xy[L_HIP], xy[L_KNE], xy[L_ANK])
    else:
        return angle_deg(xy[R_HIP], xy[R_KNE], xy[R_ANK])

def angle_series(xy_seq):
    N = xy_seq.shape[0]
    pitch = np.full(N, np.nan, float)
    headp = np.full(N, np.nan, float)
    lknee = np.full(N, np.nan, float)
    rknee = np.full(N, np.nan, float)
    for i in range(N):
        xy = xy_seq[i]
        pitch[i] = torso_pitch_deg(xy)
        headp[i] = head_pitch_deg(xy)
        lknee[i] = knee_flex_deg(xy, "L")
        rknee[i] = knee_flex_deg(xy, "R")
    return {
        "torso_pitch": _interp_nan_1d(pitch),
        "head_pitch":  _interp_nan_1d(headp),
        "knee_L":      _interp_nan_1d(lknee),
        "knee_R":      _interp_nan_1d(rknee),
    }

# ---- Per-joint L2 deltas (with virtual HEAD) ----
def joint_l2_deltas_with_head(normA, normB, joints, include_head=True):
    N = normA.shape[0]
    diffs = []
    names = []
    # real joints first
    for j in joints:
        d = np.linalg.norm(normA[:, j, :2] - normB[:, j, :2], axis=1)
        diffs.append(d); names.append(KEY_NAMES.get(j, str(j)))
    # virtual head point
    if include_head:
        d_head = np.zeros(N, dtype=np.float32)
        for i in range(N):
            ha = head_midpoint(normA[i])
            hb = head_midpoint(normB[i])
            if np.isfinite(ha).all() and np.isfinite(hb).all():
                d_head[i] = float(np.linalg.norm(ha - hb))
            else:
                d_head[i] = np.nan
        diffs.append(_interp_nan_1d(d_head)); names.append("HEAD")
    return np.stack(diffs, axis=1), names  # (N, len(names)), names[]

def summarize_deltas(deltas, names, phases=None):
    # deltas: dict name -> (N,) array
    N = len(next(iter(deltas.values())))
    overall = {}
    for name, s in deltas.items():
        s = np.asarray(s)
        with np.errstate(invalid="ignore"):
            mean = float(np.nanmean(s))
            med  = float(np.nanmedian(s))
            mx   = float(np.nanmax(s))
            argm = int(np.nanargmax(s))
        overall[name] = {
            "mean": mean, "median": med, "max": mx, "argmax_sample": argm,
            "argmax_time_pct": float(100*argm/(N-1 if N>1 else 1)),
            "phase_at_max": (phases[argm] if (phases is not None and argm < len(phases)) else None)
        }
    top_overall = sorted([(n, overall[n]["max"]) for n in names], key=lambda x: -x[1])[:5]
    return overall, top_overall

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_a", required=True)
    p.add_argument("--npz_b", required=True)
    p.add_argument("--align", choices=["apex","dtw"], default="apex",
                  help="For takeoff/landing, align upstream and slice before calling this.")
    p.add_argument("--seconds_before", type=float, default=1.0)
    p.add_argument("--seconds_after",  type=float, default=1.0)
    p.add_argument("--samples", type=int, default=101)
    p.add_argument("--use_segmenter", action="store_true")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--save_npz", default=None)
    p.add_argument("--include_hands", action="store_true", help="Include wrists (hands) and elbows in joint list.")
    p.add_argument("--include_head",  action="store_true", help="Include virtual HEAD point & head pitch delta.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    if args.save_npz:
        os.makedirs(os.path.dirname(args.save_npz) or ".", exist_ok=True)

    A, fpsA = load_pose_npz(args.npz_a)
    B, fpsB = load_pose_npz(args.npz_b)
    fpsA = fpsA or 30.0; fpsB = fpsB or 30.0

    # Alignment
    idxA, idxB = build_indices(A, B, fpsA, fpsB, args.align, args.seconds_before, args.seconds_after)
    Awin = A[idxA]
    Bwin = B[idxB]

    # Normalize (pelvis-center + torso-scale), 2D
    An = normalize_pose_sequence(Awin)
    Bn = normalize_pose_sequence(Bwin)

    # Time-standardize to N samples
    N = int(args.samples)
    AnN = resample_to_N(An, N)
    BnN = resample_to_N(Bn, N)

    # Joint set
    joints = list(KEYS_BASE8)
    if args.include_hands:
        joints += KEYS_EXPAND

    # Joint deltas (with optional HEAD)
    D_joint, joint_names = joint_l2_deltas_with_head(AnN, BnN, joints, include_head=args.include_head)

    # Angles: torso pitch, head pitch (opt), knee flex (L,R)
    angA = angle_series(AnN)
    angB = angle_series(BnN)

    D_pitch  = np.abs(angA["torso_pitch"] - angB["torso_pitch"])
    D_kneeL  = np.abs(angA["knee_L"]      - angB["knee_L"])
    D_kneeR  = np.abs(angA["knee_R"]      - angB["knee_R"])
    if args.include_head:
        D_headP = np.abs(angA["head_pitch"] - angB["head_pitch"])
    else:
        D_headP = None

    # Overall (weighted) per-sample difference
    # Joint L2 mean + small weight for angles (tuned lightly)
    w_pitch, w_knee, w_head = 0.015, 0.01, 0.015
    overall_delta = np.nanmean(D_joint, axis=1) \
        + w_pitch*D_pitch \
        + w_knee*(D_kneeL + D_kneeR)/2.0 \
        + (w_head*D_headP if (D_headP is not None) else 0.0)

    # Optional phases (from Amateur timeline)
    phases = None
    if args.use_segmenter and SEGMENTER_AVAILABLE:
        seg = segment_phases_with_airtime_v2(A, fpsA)
        # Build coarse labels by nearest mapping of idxA to N samples
        if seg is not None:
            # simple label list using indices (we mirror what your viz does)
            # We'll just tag coarse names via takeoff/air/landing using indices
            T = A.shape[0]
            labs = ["pre"] * T
            def mark(span, name):
                if not span: return
                a,b = span; a = max(0,int(a)); b = min(T-1,int(b))
                for i in range(a, b+1): labs[i] = name
            if getattr(seg, "approach", None): mark(seg.approach, "pre")
            if getattr(seg, "set", None):      mark(seg.set, "pre")
            if getattr(seg, "airtime", None):  mark(seg.airtime, "air")
            if getattr(seg, "landing", None):  mark(seg.landing, "landing")
            if seg.takeoff_idx is not None: labs[int(seg.takeoff_idx)] = "takeoff"
            if seg.landing_idx is not None: labs[int(seg.landing_idx)] = "landing"

            win_labels = [labs[i] for i in idxA]
            t = np.linspace(0, len(win_labels)-1, num=N)
            phases = [win_labels[int(round(x))] for x in t]

    # Build delta dict for summaries
    delta_dict = {n: D_joint[:,k] for k,n in enumerate(joint_names)}
    delta_dict.update({
        "TorsoPitch_deg": D_pitch,
        "KneeL_deg": D_kneeL,
        "KneeR_deg": D_kneeR,
        **({"HeadPitch_deg": D_headP} if D_headP is not None else {})
    })

    ordered = joint_names + ["TorsoPitch_deg","KneeL_deg","KneeR_deg"] + (["HeadPitch_deg"] if D_headP is not None else [])
    overall_stats, top_overall = summarize_deltas(delta_dict, ordered, phases=phases)

    # CSV per-sample
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sample_pct"] + ordered + (["phase"] if phases is not None else [])
        w.writerow(header)
        for i in range(N):
            pct = 100.0*i/(N-1 if N>1 else 1)
            row = [f"{pct:.1f}"] + [float(delta_dict[n][i]) for n in ordered]
            if phases is not None: row.append(phases[i])
            w.writerow(row)

    # JSON summary
    out = {
        "samples": N,
        "joints": joint_names,
        "angles": ["TorsoPitch_deg","KneeL_deg","KneeR_deg"] + (["HeadPitch_deg"] if D_headP is not None else []),
        "summary": overall_stats,
        "top_overall": [{"name": n, "max": float(v)} for (n,v) in top_overall],
        "notes": {
            "normalization": "pelvis-centered, torso-scaled, 2D",
            "alignment": args.align,
            "time_standardization": "resampled to N samples (0..100%)",
            "weights": {"pitch": w_pitch, "knee_each": w_knee, **({"head_pitch": w_head} if D_headP is not None else {})},
            "head_point": "virtual midpoint of ears; fallback to eyes; fallback to nose"
        }
    }
    if phases is not None: out["phases"] = phases
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", args.out_csv)
    print("Wrote:", args.out_json)

    if args.save_npz:
        np.savez_compressed(args.save_npz, A_norm=AnN, B_norm=BnN, joints=np.array(joints, dtype=np.int32))

if __name__ == "__main__":
    main()
