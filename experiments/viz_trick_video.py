"""
Chunk 3a – Trick Visualization (updated: +up height, toes proxy)
----------------------------------------------------------------
Visualizes pose keypoints and airtime phases for a given video.

Displays or saves:
  - Pose skeleton overlay
  - Takeoff / landing frames
  - (Optional) Height(+up) vs time plot using toes as proxy

Usage:
  python viz_trick_video.py --video ./dataset/Backflip/BACKFLIP_01.mov
  python viz_trick_video.py --video ./dataset/Frontflip/FRONTFLIP_01.mov --save_overlay yes --show_plot yes

Requires:
  - phase_segmentation.py with segment_phases_with_airtime_v2
  - pip install mediapipe opencv-python numpy matplotlib
"""

import os
import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from phase_segmentation import segment_phases_with_airtime_v2

mp_pose = mp.solutions.pose

# Landmark indices
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_ANK, R_ANK = 27, 28
L_FTO, R_FTO = 31, 32  # toes (FOOT_INDEX) – better “height” proxy on trampoline

# -------------------------
# Pose extraction / normalize
# -------------------------
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    kps, conf, frames = [], [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            pts = np.array([[p.x * w, p.y * h, p.z] for p in lm], dtype=np.float32)
            vis = np.array([p.visibility for p in lm], dtype=np.float32)
        else:
            pts = np.full((33, 3), np.nan, dtype=np.float32)
            vis = np.zeros((33,), dtype=np.float32)

        kps.append(pts)
        conf.append(vis)
        frames.append(frame)

    pose.close()
    cap.release()
    if not kps:
        raise RuntimeError("Zero frames extracted.")
    return np.asarray(kps), np.asarray(conf), np.asarray(frames), fps

def forward_fill_nan(arr):
    arr = arr.copy()
    for t in range(1, arr.shape[0]):
        bad = ~np.isfinite(arr[t])
        if bad.any():
            arr[t][bad] = arr[t-1][bad]
    return arr

def normalize(kps, conf=None, min_conf=0.4, eps=1e-6):
    """
    Pelvis translate + torso scale. NO XY rotation (preserve “true” up/down).
    """
    kps = kps.copy().astype(np.float32)
    if conf is not None:
        kps[conf < float(min_conf)] = np.nan
    kps = forward_fill_nan(kps)

    pelvis = 0.5 * (kps[:, L_HIP, :] + kps[:, R_HIP, :])     # (T,3)
    sh_ctr = 0.5 * (kps[:, L_SHO, :] + kps[:, R_SHO, :])     # (T,3)

    kps -= pelvis[:, None, :]

    torso_len = np.linalg.norm(sh_ctr, axis=1) + eps
    vals = torso_len[np.isfinite(torso_len)]
    scale = np.median(vals) if vals.size else 1.0
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    kps /= scale
    return kps

# -------------------------
# Drawing helpers
# -------------------------
def draw_pose(frame, kps_xy):
    """
    kps_xy: (33, 2) pixel coords (already in image space).
    """
    edges = mp_pose.POSE_CONNECTIONS
    for i, j in edges:
        if np.all(np.isfinite(kps_xy[[i, j], :])):
            cv2.line(frame,
                     tuple(kps_xy[i].astype(int)),
                     tuple(kps_xy[j].astype(int)),
                     (255, 255, 255), 1)
    for p in kps_xy:
        if np.all(np.isfinite(p)):
            cv2.circle(frame, tuple(p.astype(int)), 2, (0, 255, 0), -1)
    return frame

# -------------------------
# Main visualization logic
# -------------------------
def visualize_trick(video_path, save_overlay=False, show_plot=False):
    print(f"[viz] Processing {video_path} ...")
    kps_raw, conf, frames, fps = extract_keypoints(video_path)
    kps_norm = normalize(kps_raw, conf)

    # Phase segmentation (robust v2)
    phases = segment_phases_with_airtime_v2(
        kps_norm, fps,
        smooth_win_ms=50, hysteresis_frac=0.20, max_gap_ms=50,
        require_precontact=True, min_precontact_ms=250
    )
    takeoff, landing = phases.takeoff_idx, phases.landing_idx
    airtime_s = phases.airtime_seconds if phases.airtime_seconds is not None else 0.0
    print(f"Takeoff={takeoff}, Landing={landing}, Airtime={airtime_s:.3f}s, Quality={phases.quality}")

    # ----- Height profile (+up), using toes as primary proxy -----
    # Use normalized coords: smaller image-y = higher; we convert to “height-up” via (baseline - y)
    toes_y = np.nanmean(kps_norm[:, [L_FTO, R_FTO], 1], axis=1)
    if not np.isfinite(toes_y).any():
        toes_y = np.nanmean(kps_norm[:, [L_ANK, R_ANK], 1], axis=1)

    T = len(toes_y)
    win = int(round(0.30 * fps))  # ~300 ms
    pre_slice  = slice(max(0, (takeoff or 0) - win), (takeoff or 0))
    post_slice = slice((landing or T-1) + 1, min(T, (landing or T-1) + 1 + win))

    candidates = np.concatenate([
        toes_y[pre_slice]  if pre_slice.stop  > pre_slice.start  else np.array([]),
        toes_y[post_slice] if post_slice.stop > post_slice.start else np.array([]),
    ])
    baseline = (np.nanmedian(candidates) if candidates.size else np.nanmedian(toes_y))

    # Height-up: higher in air -> larger positive value
    height_up = baseline - toes_y

    # ----- Build annotated frames -----
    out_frames = []
    for t, f in enumerate(frames):
        kps_xy = kps_raw[t, :, :2]
        annotated = draw_pose(f.copy(), kps_xy)
        # Labels for takeoff/landing
        if takeoff is not None and t == takeoff:
            cv2.putText(annotated, "TAKEOFF", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 0), 3)
        if landing is not None and t == landing:
            cv2.putText(annotated, "LANDING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 235), 3)

        # Optional live number: height-up (clip to 2 decimals)
        hu = float(height_up[t]) if np.isfinite(height_up[t]) else 0.0
        cv2.putText(annotated, f"Height +up: {hu:+.2f}", (20, annotated.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        out_frames.append(annotated)

    # ----- Save overlay video (optional) -----
    if save_overlay:
        os.makedirs("./viz/airtime_overlay", exist_ok=True)
        out_path = os.path.join(
            "./viz/airtime_overlay",
            os.path.splitext(os.path.basename(video_path))[0] + "_overlay.mp4"
        )
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in out_frames:
            writer.write(f)
        writer.release()
        print(f"[save] Overlay video -> {out_path}")

    # ----- Show plot (optional) -----
    if show_plot:
        t_axis = np.arange(T) / fps
        plt.figure(figsize=(9, 4.5))
        plt.plot(t_axis, height_up, label="Foot height (+up, normalized)")
        if takeoff is not None:
            plt.axvline(takeoff / fps, color="green", linestyle="--", label="Takeoff")
        if landing is not None:
            plt.axvline(landing / fps, color="red", linestyle="--", label="Landing")
        plt.title("Airtime Visualization")
        plt.xlabel("Time (s)")
        plt.ylabel("Height above baseline (arb. units)")
        plt.legend()
        plt.tight_layout()
        plt.show()

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to a single video")
    ap.add_argument("--save_overlay", type=str, default="no", help="yes/no")
    ap.add_argument("--show_plot", type=str, default="no", help="yes/no")
    args = ap.parse_args()

    visualize_trick(
        args.video,
        save_overlay=args.save_overlay.lower().startswith("y"),
        show_plot=args.show_plot.lower().startswith("y")
    )

if __name__ == "__main__":
    main()
