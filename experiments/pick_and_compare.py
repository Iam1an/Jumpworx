#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, os, sys, subprocess, shlex, json, glob, numpy as np

# Shared thresholds/helpers (single source of truth)
from jwcore.coaching_thresholds import (
    TAU,
    UNITS,
    PHASE_TAG,
    select_top_metrics,   # consistent Top-metrics selector
)

from jwcore.pro_index import pick_pro_for_label


# ---------- small helpers ----------

def run(cmd_list):
    print("[pick_and_compare] Running:", " ".join(shlex.quote(s) for s in cmd_list))
    subprocess.check_call(cmd_list)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def default_out_paths(amateur_video: str, out_dir: str, suffix: str = ""):
    base = stem(amateur_video)
    base = f"{base}{suffix}" if suffix else base
    mp4  = os.path.join(out_dir, f"compare_{base}.mp4")
    jsn  = os.path.join(out_dir, f"compare_{base}.metrics.json")
    csv  = os.path.join(out_dir, f"compare_{base}.metrics.csv")
    return mp4, jsn, csv

def list_pro_videos(pro_dir: str) -> list[str]:
    vids = []
    for ext in ("*.mp4","*.mov","*.m4v","*.avi","*.MP4","*.MOV"):
        vids.extend(glob.glob(os.path.join(pro_dir, "**", ext), recursive=True))
    return sorted(vids)

# ---------- pose utilities (lightweight, local; no extra deps) ----------

# MediaPipe indices (commonly used)
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_ANK, R_ANK = 27, 28
L_FI, R_FI   = 31, 32  # foot_index; often more stable than ankles for contact

def _npz_for(video_path: str) -> str:
    return os.path.join("cache", f"{stem(video_path)}.posetrack.npz")

def _load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    P = d["P"]  # (T, 33, 3)
    meta = {}
    if "meta_json" in d.files:
        try:
            m = d["meta_json"]
            meta = m.item() if hasattr(m, "item") else dict(m)
        except Exception:
            meta = {}
    fps = float(meta.get("fps", 30.0)) if isinstance(meta, dict) else 30.0
    return P, fps

def _torso_len(P):
    v = P[:, L_SH, :] - P[:, L_HIP, :]
    return np.linalg.norm(v, axis=1)

def _hip_width(P):
    v = P[:, R_HIP, :] - P[:, L_HIP, :]
    return np.linalg.norm(v, axis=1)

def _hand_span(P):
    # wrists: 9,10 in MediaPipe; fallback to elbows (13,14) if wrists too sparse
    L_WR, R_WR = 9, 10
    L_EL, R_EL = 13, 14
    s = np.linalg.norm(P[:, R_WR, :] - P[:, L_WR, :], axis=1)
    m = np.isfinite(s)
    if m.sum() < 8:
        s = np.linalg.norm(P[:, R_EL, :] - P[:, L_EL, :], axis=1)
    return s

def _hand_asym_about_midline(P):
    mid = 0.5 * (P[:, L_SH, :] + P[:, R_SH, :])
    lw = P[:, 9, :] - mid
    rw = P[:, 10, :] - mid
    return np.abs(np.linalg.norm(lw, axis=1) - np.linalg.norm(rw, axis=1))

def _stance_width(P):
    return np.linalg.norm(P[:, R_ANK, :] - P[:, L_ANK, :], axis=1)

def _mean_in_window(x, i: int | None, halfw: int) -> float:
    if x is None or i is None or not np.isfinite(i):
        return float("nan")
    i = int(i)
    lo, hi = max(0, i - halfw), min(len(x), i + halfw + 1)
    seg = x[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmean(seg)) if seg.size else float("nan")

def _apex_index(P):
    # average of foot indices (lower y = higher in image coords)
    y = 0.5*(P[:, L_FI, 1] + P[:, R_FI, 1])
    if (~np.isfinite(y)).sum() > (0.9*len(y)):
        # fallback to ankles
        y = 0.5*(P[:, L_ANK, 1] + P[:, R_ANK, 1])
    if not np.isfinite(y).any():
        return None
    return int(np.nanargmin(y))

def _landing_index(P, apex_i: int | None):
    if apex_i is None:
        return None
    # crude: max foot y after apex
    y = 0.5*(P[:, L_FI, 1] + P[:, R_FI, 1])
    if (~np.isfinite(y)).sum() > (0.9*len(y)):
        y = 0.5*(P[:, L_ANK, 1] + P[:, R_ANK, 1])
    if not np.isfinite(y).any():
        return None
    post = y[apex_i:]
    if post.size < 3:
        return None
    j = int(np.nanargmax(post))
    return apex_i + j

def _phase_metrics_for_npz(npz_path: str) -> dict:
    P, fps = _load_npz(npz_path)
    tl = _torso_len(P); hw = _hip_width(P)
    hspan = _hand_span(P); hasym = _hand_asym_about_midline(P)
    stance = _stance_width(P)

    # normalize
    hspan_n = hspan / np.maximum(1e-6, tl)
    hasym_n = hasym / np.maximum(1e-6, tl)
    stance_n = stance / np.maximum(1e-6, hw)

    apex_i = _apex_index(P)
    land_i = _landing_index(P, apex_i)

    midair_hspan_pct = 100.0 * _mean_in_window(hspan_n, apex_i, 6)
    midair_hasym_pct = 100.0 * _mean_in_window(hasym_n, apex_i, 6)
    landing_stance_pct = 100.0 * _mean_in_window(stance_n, land_i, 4)

    return {
        "midair_hand_span_pct_of_torso": float(midair_hspan_pct),
        "midair_hand_asym_pct_of_torso": float(midair_hasym_pct),
        "landing_stance_width_pct_of_hip": float(landing_stance_pct),
        "idx_midair": int(apex_i) if apex_i is not None else -1,
        "idx_landing": int(land_i) if land_i is not None else -1,
        "fps": float(fps),
    }

# ---------- pro selection ----------

def pick_pro_for_label_simple(label: str, pro_dir: str, strategy: str = "closest") -> str | None:
    all_vids = list_pro_videos(pro_dir)
    if not all_vids:
        return None

    lab = label.lower()
    candidates = [v for v in all_vids if lab in os.path.basename(v).lower()]
    if not candidates:
        candidates = all_vids

    if strategy == "first":
        return candidates[0]
    elif strategy == "random":
        import random
        return random.choice(candidates)
    elif strategy == "closest":
        # For "closest", time-standardize series and compare RMSE.
        # Implement hip_y, ankle_y, and pitch (torso angle) series.
        ANK_L, ANK_R = 31, 32  # foot index as proxy for ankle height stability
        HIP_L, HIP_R = 23, 24
        L_SH, R_SH = 11, 12
        L_HIP, R_HIP = 23, 24

        def npz_for(v):
            return os.path.join("cache", f"{stem(v)}.posetrack.npz")

        def series(npz_path, feature):
            try:
                d = np.load(npz_path, allow_pickle=True)
                P = d["P"]
            except Exception:
                return None
            if P.ndim != 3 or P.shape[1] < 33:
                return None
            if feature == "hip_y":
                s = 0.5 * (P[:, HIP_L, 1] + P[:, HIP_R, 1])
            elif feature == "pitch":
                # torso angle: hips -> shoulders; angle vs vertical (y-axis)
                torso = 0.5*(P[:, L_SH, :]) + 0.5*(P[:, R_SH, :]) - (0.5*P[:, L_HIP, :] + 0.5*P[:, R_HIP, :])
                horiz = np.sqrt(np.square(torso[:,0]) + np.square(torso[:,2]))
                s = np.arctan2(horiz, np.abs(torso[:,1]) + 1e-6)  # radians
            else:
                s = 0.5 * (P[:, ANK_L, 1] + P[:, ANK_R, 1])
            s = s.astype(np.float32)
            m = np.isfinite(s)
            if m.sum() < 8:
                return None
            s = (s[m] - s[m].mean()) / (s[m].std() + 1e-6)
            N = 180
            x_src = np.linspace(0.0, 1.0, num=s.size)
            x_dst = np.linspace(0.0, 1.0, num=N)
            return np.interp(x_dst, x_src, s)

        def dist(a, b):
            if a is None or b is None or a.size != b.size:
                return float("inf")
            return float(np.sqrt(np.mean((a - b) ** 2)))

        amat = globals().get("LAST_PARSED_ARGS")
        if not amat:
            return candidates[0]
        amat_npz = npz_for(amat.amateur_video)
        amat_s = series(amat_npz, getattr(amat, "align_feature", "ankle_y"))

        best = (float("inf"), candidates[0])
        for v in candidates:
            npz = npz_for(v)
            if not os.path.exists(npz):
                continue
            pro_s = series(npz, getattr(amat, "align_feature", "ankle_y"))
            d = dist(amat_s, pro_s)
            if d < best[0]:
                best = (d, v)
        print(f"[pick_and_compare] Closest match: {os.path.basename(best[1])} (dist={best[0]:.3f})")
        return best[1]
    else:
        return candidates[0]

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--amateur_video", required=True)
    ap.add_argument("--pro_videos_dir", required=True)
    ap.add_argument("--label", default="backflip")

    # Strategy
    ap.add_argument("--strategy", choices=["first","random","closest"], default="closest")
    ap.add_argument("--seed", type=int, default=None)

    # Alignment/window
    ap.add_argument("--align", choices=["apex","dtw","takeoff","landing"], default="apex")
    ap.add_argument("--align_feature", choices=["ankle_y","hip_y","pitch"], default="ankle_y")
    ap.add_argument("--seconds_before", type=float, default=1.0)
    ap.add_argument("--seconds_after", type=float, default=1.0)

    # Viz/output
    ap.add_argument("--output_fps", type=float, default=12.0)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--coords", choices=["auto","pixels","normalized"], default="auto")
    ap.add_argument("--fit", choices=["none","window","frame"], default="none")
    ap.add_argument("--margin_px", type=int, default=40)
    ap.add_argument("--color_a", default="red")
    ap.add_argument("--color_b", default="blue")
    ap.add_argument("--hud_corner", choices=["left","right"], default="left")
    ap.add_argument("--use_segmenter", action="store_true")
    ap.add_argument("--metrics_panel", action="store_true")
    ap.add_argument("--include_head", action="store_true")
    ap.add_argument("--include_hands", action="store_true")
    ap.add_argument("--invert_y", action="store_true")
    ap.add_argument("--blank_canvas", action="store_true")
    ap.add_argument("--canvas_width", type=int, default=1280)
    ap.add_argument("--canvas_height", type=int, default=720)
    ap.add_argument("--score_ema_alpha", type=float, default=None)
    ap.add_argument("--top_bg_height", type=int, default=None)
    ap.add_argument("--extra_viz_args", default="")
    ap.add_argument("--out_dir", default="viz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics_out", default=None)
    ap.add_argument("--metrics_csv", default=None)
    ap.add_argument("--summary_out", default=None, help="Optional summary JSON with phase metrics and top_metrics")

    args = ap.parse_args()
    global LAST_PARSED_ARGS; LAST_PARSED_ARGS = args

    if args.seed is not None:
        np.random.seed(args.seed)

    pro_video = pick_pro_for_label(
        label=args.label,
        student_features_path=student_feat_path,
        index_path=args.pro_index,
        strategy=args.strategy,
    )
    if pro_video is None:
        raise FileNotFoundError(f"No pro videos found in: {args.pro_videos_dir}")
    print(f"[pick_and_compare] Selected pro: {pro_video}")

    amat_stem = stem(args.amateur_video)
    pro_stem  = stem(pro_video)
    npz_a = os.path.join("cache", f"{amat_stem}.posetrack.npz")
    npz_b = os.path.join("cache", f"{pro_stem}.posetrack.npz")

    if not os.path.exists(npz_a):
        run([sys.executable, "scripts/extract_keypoints.py", "--video", args.amateur_video])
    if not os.path.exists(npz_b):
        run([sys.executable, "scripts/extract_keypoints.py", "--video", pro_video])

    suffix = f"_fps{int(args.output_fps)}" if args.output_fps else ""
    mp4_path, json_path, csv_path = default_out_paths(args.amateur_video, args.out_dir, suffix=suffix)
    out_mp4  = args.out or mp4_path
    out_json = args.metrics_out or json_path
    out_csv  = args.metrics_csv or csv_path
    summary_out = args.summary_out or os.path.splitext(out_json)[0] + ".summary.json"

    for p in (out_mp4, out_json, out_csv, summary_out):
        ensure_dir(p)

    # --------- phase-aware metrics & top_metrics (no LLM) ---------
    am_phase = _phase_metrics_for_npz(npz_a)
    pr_phase = _phase_metrics_for_npz(npz_b)
    self_compare = (stem(args.amateur_video) == stem(pro_video))

    # Use shared selector so thresholds and formatting stay consistent
    top = [] if self_compare else select_top_metrics(
        am_features=am_phase,
        pro_features=pr_phase,
        # keys=None -> defaults from jwcore.coaching_thresholds.DEFAULT_COMPARE_KEYS
        max_items=3,
    )

    summary = {
        "amateur": stem(args.amateur_video),
        "pro": stem(pro_video),
        "align": args.align,
        "align_feature": args.align_feature,
        "self_compare": bool(self_compare),
        "phase_metrics": {
            "amateur": am_phase,
            "pro": pr_phase,
        },
        "top_metrics": top if top else "No material differences above thresholds",
        "tau": TAU,
        "units": UNITS,
        "phase_tag": PHASE_TAG,
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f("[pick_and_compare] summary -> {summary_out}"))

    # --------- visualizer (unchanged UX) ---------
    cmd = [sys.executable, "viz_compare_side_by_side.py",
           "--video_a", args.amateur_video,
           "--video_b", pro_video,
           "--align", args.align,
           "--align_feature", args.align_feature,
           "--seconds_before", str(args.seconds_before),
           "--seconds_after", str(args.seconds_after),
           "--output_fps", str(args.output_fps),
           "--height", str(args.height),
           "--coords", args.coords,
           "--fit", args.fit,
           "--margin_px", str(args.margin_px),
           "--color_a", args.color_a,
           "--color_b", args.color_b,
           "--hud_corner", args.hud_corner,
           "--json_out", out_json,
           "--csv_out", out_csv]
    if args.use_segmenter: cmd.append("--use_segmenter")
    if args.metrics_panel: cmd.append("--metrics_panel")
    if args.include_head: cmd.append("--include_head")
    if args.include_hands: cmd.append("--include_hands")
    if args.invert_y: cmd.append("--invert_y")
    if args.blank_canvas:
        cmd += ["--blank_canvas", "--canvas_width", str(args.canvas_width), "--canvas_height", str(args.canvas_height)]
    if args.score_ema_alpha is not None:
        cmd += ["--score_ema_alpha", str(args.score_ema_alpha)]
    if args.top_bg_height is not None:
        cmd += ["--top_bg_height", str(args.top_bg_height)]
    extra = args.extra_viz_args.strip()
    if extra: cmd += shlex.split(extra)

    run(cmd)

if __name__ == "__main__":
    main()
