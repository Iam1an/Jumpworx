#!/usr/bin/env python3
"""
viz_2d.py — overlay 2D skeleton + phases on a video

Input:
  --video       input mp4 (optional; if omitted, renders on blank canvas using size from .npz)
  --posetrack   .npz produced by scripts/extract_keypoints.py
  --out         output mp4 (default: viz/<stem>_viz2d.mp4)
  --every       stride to render (default: 1)
  --label       optional label text to overlay (e.g., classifier result)

What it draws:
  • BlazePose 33‑kp skeleton per rendered frame (NaN‑safe)
  • (Optional) takeoff/landing vertical lines if we can compute phases via jwcore.phase_segmentation

Deps: pip install opencv-python numpy

Notes:
  - If your .npz has 'frame_indices', we seek to those exact frames.
  - If --video is missing, we render on a blank canvas using npz['size'].
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import cv2

try:
    from jwcore.phase_segmentation import segment_phases_with_airtime_v2
except Exception:
    segment_phases_with_airtime_v2 = None  # phases optional

# BlazePose 33 connections (subset for clarity)
EDGES: List[Tuple[int,int]] = [
    (11,13),(13,15),  # left arm
    (12,14),(14,16),  # right arm
    (23,25),(25,27),  # left leg
    (24,26),(26,28),  # right leg
    (11,12),          # shoulders
    (23,24),          # hips
    (11,23),(12,24),  # torso lines
    (27,31),(28,32),  # ankles to foot index
]


def _default_out(video_path: str|None, npz_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path or npz_path))[0]
    os.makedirs("viz", exist_ok=True)
    return os.path.join("viz", f"{stem}_viz2d.mp4")


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    if "kps_xy" not in data:
        raise KeyError("posetrack npz missing 'kps_xy'")
    return data


def _draw_skeleton(img: np.ndarray, pts: np.ndarray) -> None:
    # pts: (33,2) pixel xy, may contain NaNs
    h, w = img.shape[:2]
    # draw joints
    for i in range(min(33, pts.shape[0])):
        x, y = pts[i,0], pts[i,1]
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, (int(x), int(y)), 3, (50, 220, 50), -1, lineType=cv2.LINE_AA)
    # draw bones
    for a,b in EDGES:
        if a < pts.shape[0] and b < pts.shape[0]:
            xa, ya = pts[a,0], pts[a,1]
            xb, yb = pts[b,0], pts[b,1]
            if np.isfinite([xa,ya,xb,yb]).all():
                cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (0, 200, 255), 2, lineType=cv2.LINE_AA)


def _compute_phase_lines(npz: dict):
    if segment_phases_with_airtime_v2 is None:
        return None
    if "kps_xyz" not in npz:
        return None
    fps = float(npz.get("fps", 0.0)) if not isinstance(npz.get("fps", 0.0), np.ndarray) else float(npz.get("fps")[()])
    if not np.isfinite(fps) or fps <= 0.0:
        return None
    phases = segment_phases_with_airtime_v2(npz["kps_xyz"], fps=fps, require_precontact=False)
    # best-effort: return frame indices (ints) for takeoff/landing if present
    try:
        return {
            "takeoff": int(phases.takeoff_frame) if phases.takeoff_frame is not None else None,
            "landing": int(phases.landing_frame) if phases.landing_frame is not None else None,
        }
    except Exception:
        return None


def render(video: str|None, npz_path: str, out_path: str|None, every: int=1, label: str|None=None) -> str:
    data = _load_npz(npz_path)
    kps_xy = data["kps_xy"]  # (T,33,2)
    frames = data.get("frame_indices")  # (T,), used for seeking
    H, W = (int(data["size"][0]), int(data["size"][1])) if "size" in data else (720, 1280)

    # video i/o
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video}")
        outW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        outH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    else:
        cap = None
        outW, outH = W, H
        fps = float(data.get("fps", 30.0)) if not isinstance(data.get("fps", 30.0), np.ndarray) else float(data["fps"][()])
        fps = fps if np.isfinite(fps) and fps>0 else 30.0

    out_path = out_path or _default_out(video, npz_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps/max(1,every), (outW, outH))

    # optional phases
    phase_lines = _compute_phase_lines(data)

    T = kps_xy.shape[0]
    for i in range(0, T, max(1,every)):
        # fetch base frame
        if cap is not None and frames is not None and i < len(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[i]))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((outH,outW,3), np.uint8)
        elif cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = np.zeros((outH,outW,3), np.uint8)

        # draw skeleton
        pts = kps_xy[i]
        _draw_skeleton(frame, pts)

        # overlay phase markers
        if phase_lines:
            x = int(outW * i / max(1, T-1))
            if phase_lines.get("takeoff") is not None and i == phase_lines["takeoff"]:
                cv2.line(frame, (x,0), (x,outH), (0,0,255), 2)
                cv2.putText(frame, "TO", (x+6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            if phase_lines.get("landing") is not None and i == phase_lines["landing"]:
                cv2.line(frame, (x,0), (x,outH), (0,255,0), 2)
                cv2.putText(frame, "LD", (x+6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        # HUD text
        hud = label or ""
        cv2.putText(frame, f"frame {i}/{T-1} {hud}", (16, outH-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    if cap is not None:
        cap.release()
    return out_path


# ------------------------------- CLI ------------------------------------

def _build_argparser():
    p = argparse.ArgumentParser(description="Overlay 2D BlazePose skeleton on video or blank canvas")
    p.add_argument("--posetrack", required=True, help="Input .npz from extract_keypoints.py")
    p.add_argument("--video", help="Optional source video to overlay onto")
    p.add_argument("--out", help="Output mp4 path (default viz/<stem>_viz2d.mp4)")
    p.add_argument("--every", type=int, default=1, help="Render every Nth frame")
    p.add_argument("--label", help="Optional text label to overlay")
    return p


def main(argv: List[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)
    out = render(args.video, args.posetrack, args.out, every=args.every, label=args.label)
    print(f"✅ wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
