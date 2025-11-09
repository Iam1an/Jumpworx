#!/usr/bin/env python3
"""
CLI test for jwcore.phase_detect

Example:
  python -m scripts.test_phase_detect cache/TRICK26_BACKFLIP.posetrack.npz \
      --plot

Outputs:
  • Summary of detected phases
  • Per-frame labels written to <stem>_phases.txt
  • Optional ankle-height plot colored by phase
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure parent directory on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.phase_detect import detect_phases_from_pose


def plot_phases(ankle_y: np.ndarray, labels: list[str], fps: float):
    """Simple visualization of phase segmentation."""
    T = len(ankle_y)
    t = np.arange(T) / fps

    # Map phases to colors
    phase_colors = {
        "approach": "#777777",
        "set": "#F4B400",
        "early_air": "#DB4437",
        "midair": "#4285F4",
        "late_air": "#0F9D58",
        "landing": "#AA00FF",
        "unknown": "#999999",
    }
    colors = [phase_colors.get(lbl, "#000000") for lbl in labels]

    plt.figure(figsize=(10, 4))
    plt.title("Ankle Height vs. Time (colored by phase)")
    plt.xlabel("Time (s)")
    plt.ylabel("Ankle height (y, pixels)")
    plt.gca().invert_yaxis()  # smaller y = higher in image

    # Plot line and phase-colored background
    plt.plot(t, ankle_y, "k-", lw=1)
    for i in range(T - 1):
        plt.axvspan(t[i], t[i + 1], color=colors[i], alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Test phase detection from .posetrack.npz")
    ap.add_argument("npz", help="Path to pose-track NPZ file")
    ap.add_argument("--plot", action="store_true", help="Plot ankle height vs phase")
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        ap.error(f"File not found: {args.npz}")

    # Load pose data
    P, V, fps, meta = load_posetrack_npz(args.npz)
    print(f"[INFO] Loaded: {args.npz}")
    print(f"        Pose shape: {P.shape}, fps: {fps:.2f}")

    # Run detection
    result = detect_phases_from_pose(P, fps)

    print("\n[RESULT SUMMARY]")
    for k, v in result.quality.items():
        print(f"{k:16s}: {v}")
    print(f"Takeoff: {result.takeoff_idx}, Landing: {result.landing_idx}, Apex: {result.apex_idx}")

    unique, counts = np.unique(result.labels, return_counts=True)
    print("\nPhase counts:")
    for u, c in zip(unique, counts):
        print(f"  {u:10s} {c:4d}")

    # Save per-frame labels
    stem = os.path.splitext(args.npz)[0]
    out_txt = f"{stem}_phases.txt"
    with open(out_txt, "w") as f:
        for i, lbl in enumerate(result.labels):
            f.write(f"{i:04d} {lbl}\n")
    print(f"\n[INFO] Wrote per-frame labels to {out_txt}")

    # Optional plot
    if args.plot:
        # Reconstruct mean ankle y to visualize
        left, right = 27, 28  # MP landmarks
        ankle_y = np.nanmean(P[:, [left, right], 1], axis=1)
        plot_phases(ankle_y, result.labels, fps)


if __name__ == "__main__":
    main()
