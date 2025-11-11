#!/usr/bin/env python3
"""
Overlay 3–5 coaching tips onto a video.

- Tries OpenCV first; if decoding fails (green/empty), falls back to imageio/FFmpeg.
- Slowmo: --slowmo >1.0 lowers fps (e.g., 2.0 halves fps).

Examples:
  python scripts/overlay_tips.py \
    --video viz/compare/preview_TRICK15_vs_TRICK32_airtime.mp4 \
    --coach-json feedback/coach_TRICK15.json \
    --out viz/compare/preview_TRICK15_with_tips.mp4 \
    --top-n 3 --slowmo 1.5
"""

import os
import sys
import json
import argparse

def _load_tips(path: str, top_n: int):
    import json
    obj = json.loads(open(path, "r").read())
    tips = []
    if isinstance(obj, dict):
        if "coaching" in obj and isinstance(obj["coaching"], dict):
            tips = obj["coaching"].get("tips", [])
        elif "tips" in obj:
            tips = obj.get("tips", [])
    tips = [str(t).strip() for t in tips if str(t).strip()]
    return tips[:max(1, top_n)]

def _draw_box_with_text(frame, lines, margin=24, alpha=0.6, font_scale=0.8, thickness=2):
    import cv2
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # measure text block
    line_h = 0
    max_w = 0
    sizes = []
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        sizes.append((tw, th))
        max_w = max(max_w, tw)
        line_h = max(line_h, th + 8)
    box_w = max_w + 2*margin
    box_h = line_h*len(lines) + 2*margin

    # position: top-left
    x0, y0 = margin, margin
    x1, y1 = min(w-1, x0 + box_w), min(h-1, y0 + box_h)

    # draw translucent rectangle
    overlay = frame.copy()
    import numpy as np
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    # draw text lines
    y = y0 + margin + sizes[0][1]
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (x0 + margin, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
    return frame

def _process_with_opencv(video_path, out_path, tips, slowmo, margin, alpha, font_scale, thickness):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    out_fps = max(1.0, src_fps / max(1e-6, float(slowmo)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

    wrote = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = _draw_box_with_text(
            frame,
            lines=[f"• {t}" for t in tips],
            margin=margin,
            alpha=float(alpha),
            font_scale=float(font_scale),
            thickness=int(thickness),
        )
        wr.write(frame)
        wrote += 1

    cap.release()
    wr.release()
    return wrote, src_fps

def _process_with_imageio(video_path, out_path, tips, slowmo, margin, alpha, font_scale, thickness):
    import imageio.v3 as iio
    import numpy as np
    import cv2

    meta = iio.immeta(video_path, plugin="FFMPEG")
    src_fps = float(meta.get("fps", 24.0))
    out_fps = max(1.0, src_fps / max(1e-6, float(slowmo)))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wrote = 0

    # Streamed writer to avoid loading all frames into RAM
    with iio.imopen(out_path, "w", plugin="FFMPEG", fps=out_fps, codec="libx264") as wr:
        for frame in iio.imiter(video_path, plugin="FFMPEG"):
            # imageio gives RGB; convert to BGR for cv2 drawing, then back to RGB
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr = _draw_box_with_text(
                bgr,
                lines=[f"• {t}" for t in tips],
                margin=margin,
                alpha=float(alpha),
                font_scale=float(font_scale),
                thickness=int(thickness),
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            wr.write(rgb)
            wrote += 1

    return wrote, src_fps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--coach-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--font-scale", type=float, default=0.8)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--slowmo", type=float, default=1.0, help=">1.0 slows output FPS (e.g., 2.0 halves fps)")
    args = ap.parse_args()

    tips = _load_tips(args.coach_json, args.top_n)
    if not tips:
        print("[overlay] no tips found; aborting.")
        sys.exit(1)

    # Try OpenCV first
    wrote, src_fps = _process_with_opencv(
        args.video, args.out, tips, args.slowmo, args.margin, args.alpha, args.font_scale, args.thickness
    )

    # If OpenCV wrote no frames (common macOS H.264 issue), fall back to imageio/FFmpeg
    if wrote == 0:
        try:
            import imageio.v3 as _  # check availability
        except Exception as e:
            print("[overlay] OpenCV failed to decode and imageio is not available. Try `pip install imageio[ffmpeg]`.")
            sys.exit(2)

        wrote, src_fps = _process_with_imageio(
            args.video, args.out, tips, args.slowmo, args.margin, args.alpha, args.font_scale, args.thickness
        )

    out_fps = max(1.0, (src_fps or 24.0) / max(1e-6, float(args.slowmo)))
    print(f"[overlay] wrote {args.out} (frames={wrote}, fps_in={src_fps:.2f} → fps_out={out_fps:.2f})")

if __name__ == "__main__":
    main()
