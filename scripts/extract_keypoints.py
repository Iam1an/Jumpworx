#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTRACT (I/O SOURCE OF TRUTH)
=============================
This script reads videos, runs MediaPipe Pose, and writes a *canonical* pose-track:
  - Output file: cache/<stem>.posetrack.npz
  - Keys:
      P: (T,33,3) float32 -> x,y in *pixels*; z scaled by *image width* (pixel-ish units)
      V: (T,33)   float32 -> visibility/presence in [0,1]
      meta_json: JSON with at least:
          {"fps": <float>, "image_size": [H,W], "landmark_names": [...33 names...],
           "units": "pixels_xy_and_pixel_scaled_z",
           "z_note": "z scaled by image width; positive toward camera (MediaPipe sign)."}

Conventions locked here:
  - x increases right, y increases downward (image coords).
  - z sign is MediaPipe's (positive toward camera). We DO NOT flip z here.
  - Low-visibility landmarks can be masked to NaN in P (configurable).
  - Short NaN gaps (<=4 frames by default) are interpolated per landmark axis.

Downstream expectations:
  - jwcore.pose_utils.load_posetrack_npz reads exactly this schema.
  - jwcore.pose_utils.features_from_posetrack consumes (P,V,meta) directly.
  - jwcore.pose_extract loads NPZs and saves feature JSONs, without changing any math.

CLI:
  # single video → NPZ
  python scripts/extract_keypoints.py --video videos/training/TRICK06_BACKFLIP.mov

  # batch (glob)
  python scripts/extract_keypoints.py --glob "videos/training/*.mov"

  # inspect existing NPZs
  python scripts/extract_keypoints.py --npz "cache/*.posetrack.npz" --summary
  python scripts/extract_keypoints.py --npz "cache/TRICK06_BACKFLIP.posetrack.npz" --head 10
"""

from __future__ import annotations
import argparse, os, sys, json, glob
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception as e:
    print("OpenCV (cv2) is required. pip install opencv-python", file=sys.stderr)
    raise

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp, mp_pose = None, None

LANDMARK_NAMES: List[str] = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow",
    "right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index",
    "left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index",
]
NAME_TO_IDX = {n:i for i,n in enumerate(LANDMARK_NAMES)}
N_LM = 33

def _stem(path: str) -> str:
    import os
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem

def _to_pixels(xn: float, yn: float, zn: float, H: int, W: int):
    # normalize→pixels, and scale z with width so x,y,z are comparable
    return xn * W, yn * H, zn * W

def _interp_nans_along_time(arr: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Interpolate NaN runs up to max_gap along time for each column independently."""
    T, D = arr.shape
    out = arr.copy()
    for d in range(D):
        v = out[:, d]
        isn = np.isnan(v)
        if not isn.any():
            continue
        ok = np.where(~isn)[0]
        if ok.size == 0:
            continue
        bad = np.where(isn)[0]
        # find contiguous NaN runs
        starts, ends = [], []
        prev = -10**9
        for i in bad:
            if i != prev + 1:
                starts.append(i)
            prev = i
        prev = -10**9
        for i in bad:
            if i != prev + 1 and prev != -10**9:
                ends.append(prev)
            prev = i
        if prev != -10**9:
            ends.append(prev)
        for a, b in zip(starts, ends):
            gap = b - a + 1
            if gap > max_gap:
                continue
            L, R = a - 1, b + 1
            if L < 0 or R >= T or np.isnan(v[L]) or np.isnan(v[R]):
                continue
            xs = np.linspace(0, 1, gap + 2)
            seg = np.interp(xs, [0, 1], [v[L], v[R]])
            out[a:b+1, d] = seg[1:-1]
    return out

def extract_video(
    video_path: str,
    out_dir: str = "cache",
    every: int = 1,
    model_complexity: int = 1,
    min_visibility: float = 0.0,
    interp_max_gap: int = 4,
) -> int:
    if mp_pose is None:
        print("MediaPipe not available. pip install mediapipe", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_stem(video_path)}.posetrack.npz")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return 3

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 60.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0:
        print("Invalid video frame size.", file=sys.stderr)
        return 4

    P_list, V_list = [], []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx % max(1, int(every))) != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            if res is None or res.pose_landmarks is None or res.pose_landmarks.landmark is None:
                P_list.append(np.full((N_LM, 3), np.nan, dtype=np.float32))
                V_list.append(np.zeros((N_LM,), dtype=np.float32))
                frame_idx += 1
                continue

            lms = res.pose_landmarks.landmark
            P_frame = np.full((N_LM, 3), np.nan, dtype=np.float32)
            V_frame = np.zeros((N_LM,), dtype=np.float32)
            for i in range(min(N_LM, len(lms))):
                lm = lms[i]
                xn, yn, zn = float(lm.x), float(lm.y), float(lm.z)
                x, y, z = _to_pixels(xn, yn, zn, H, W)
                if not (0.0 <= xn <= 1.0) or not (0.0 <= yn <= 1.0):
                    x = y = z = float("nan")
                P_frame[i, :] = (x, y, z)
                vis = float(getattr(lm, "visibility", 0.0))
                pres = float(getattr(lm, "presence", 0.0)) if hasattr(lm, "presence") else 0.0
                V_frame[i] = max(0.0, min(1.0, max(vis, pres)))

            if min_visibility > 0.0:
                low = V_frame < min_visibility
                P_frame[low, :] = np.nan

            P_list.append(P_frame)
            V_list.append(V_frame)
            frame_idx += 1

    cap.release()

    P = np.stack(P_list, axis=0) if P_list else np.zeros((0, N_LM, 3), dtype=np.float32)
    V = np.stack(V_list, axis=0) if V_list else np.zeros((0, N_LM), dtype=np.float32)

    if interp_max_gap > 0 and P.shape[0] > 0:
        for i in range(N_LM):
            P[:, i, :] = _interp_nans_along_time(P[:, i, :], max_gap=interp_max_gap)

    meta = {
        "fps": float(fps),
        "image_size": (int(H), int(W)),
        "landmark_names": LANDMARK_NAMES,
        "units": "pixels_xy_and_pixel_scaled_z",
        "z_note": "z scaled by image width; positive toward camera (MediaPipe sign).",
    }

    np.savez_compressed(
        out_path,
        P=P.astype(np.float32),
        V=V.astype(np.float32),
        meta_json=json.dumps(meta),
    )

    print(f"✅ {video_path} → {out_path}  (T={P.shape[0]}, fps={fps:.2f}, size={P.shape})")
    return 0

# -------- inspection helpers --------

def load_posetrack_npz(path: str):
    dat = np.load(path, allow_pickle=False)
    P = dat["P"].astype(np.float32)
    V = dat["V"].astype(np.float32)
    meta = json.loads(str(dat["meta_json"]))
    assert P.ndim == 3 and P.shape[1] == N_LM and P.shape[2] == 3, f"bad P shape {P.shape}"
    assert V.ndim == 2 and V.shape[0] == P.shape[0] and V.shape[1] == N_LM, f"bad V shape {V.shape}"
    return P, V, meta

def summarize_npz(paths: List[str], head: int = 8, summary_only: bool = False):
    for pth in paths:
        try:
            P, V, meta = load_posetrack_npz(pth)
        except Exception as e:
            print(f"ERROR reading {pth}: {e}")
            continue

        T = P.shape[0]
        fps = float(meta.get("fps", float("nan")))
        H, W = meta.get("image_size", (float("nan"), float("nan")))
        vis_rate = float(np.mean(~np.isnan(P[..., 0]))) if T > 0 else 0.0

        feet_idx = [NAME_TO_IDX["left_foot_index"], NAME_TO_IDX["right_foot_index"],
                    NAME_TO_IDX["left_heel"], NAME_TO_IDX["right_heel"]]
        toes_y = np.nanmin(P[:, feet_idx, 1], axis=1) if T > 0 else np.array([])
        baseline = np.nanpercentile(toes_y, 5) if toes_y.size else float("nan")
        clearance = toes_y - baseline if toes_y.size else np.array([])

        if summary_only:
            print(f"{os.path.basename(pth):35s}  T={T:4d}  fps={fps:5.1f}  "
                  f"nan_rate={1.0 - vis_rate:.2f}  clearance_p95={np.nanpercentile(clearance,95) if clearance.size else float('nan'):.1f}px")
            continue

        print(f"=== {pth} ===")
        print(json.dumps({
            "T_frames": T, "fps": fps, "image_h": H, "image_w": W,
            "coord_nan_rate": float(1.0 - vis_rate),
            "foot_clearance_px_p95": float(np.nanpercentile(clearance, 95)) if clearance.size else float("nan"),
        }, indent=2))

        if T > 0:
            nos = NAME_TO_IDX["nose"]
            lhip, rhip = NAME_TO_IDX["left_hip"], NAME_TO_IDX["right_hip"]
            lheel, rheel = NAME_TO_IDX["left_heel"], NAME_TO_IDX["right_heel"]
            for t in range(min(head, T)):
                row = dict(
                    t=t,
                    nose_x=float(P[t, nos, 0]), nose_y=float(P[t, nos, 1]), nose_z=float(P[t, nos, 2]),
                    hip_y=float(np.nanmean([P[t, lhip, 1], P[t, rhip, 1]])),
                    feet_y=float(np.nanmin([P[t, lheel, 1], P[t, rheel, 1]])),
                )
                print(row)

# -------- CLI --------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract MediaPipe Pose keypoints to canonical .posetrack.npz")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--video", type=str, help="Path to a single input video")
    g.add_argument("--glob", type=str, help="Glob for input videos (e.g. videos/training/*.mov)")
    ap.add_argument("--every", type=int, default=1, help="Process every Nth frame (default 1)")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0,1,2], help="MediaPipe Pose model complexity")
    ap.add_argument("--min_visibility", type=float, default=0.0, help="Mask coords to NaN below this visibility (0..1)")
    ap.add_argument("--interp_max_gap", type=int, default=4, help="Interpolate NaN runs up to this length (frames)")
    ap.add_argument("--out_dir", type=str, default="cache", help="Output directory for .posetrack.npz")
    ap.add_argument("--npz", type=str, help="Inspect existing NPZs (glob)")
    ap.add_argument("--summary", action="store_true", help="Print one-line summary per NPZ in inspect mode")
    ap.add_argument("--head", type=int, default=8, help="Rows to print for quick table when inspecting")
    args = ap.parse_args(argv)

    if args.npz:
        paths = sorted(glob.glob(args.npz))
        if not paths:
            print(f"No files matched {args.npz}")
            return 1
        summarize_npz(paths, head=args.head, summary_only=args.summary)
        return 0

    if not args.video and not args.glob:
        print("Provide --video or --glob (or use --npz for inspection).", file=sys.stderr)
        return 2

    videos: List[str] = []
    if args.video:
        videos = [args.video]
    elif args.glob:
        videos = sorted(glob.glob(args.glob))
    if not videos:
        print("No videos found.", file=sys.stderr)
        return 2

    rc = 0
    for vid in videos:
        rc |= extract_video(
            video_path=vid,
            out_dir=args.out_dir,
            every=max(1, int(args.every)),
            model_complexity=int(args.model_complexity),
            min_visibility=float(args.min_visibility),
            interp_max_gap=int(args.interp_max_gap),
        )
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
