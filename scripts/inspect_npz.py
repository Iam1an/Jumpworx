#!/usr/bin/env python3
"""
inspect_npz.py
Quick NPZ introspection for pose arrays.

- Supports files with keys: 'kps_xyz', 'visibility', 'fps' (your layout)
- Also falls back to common keys: 'pose', 'keypoints', 'landmarks', 'arr_0'
- Prints keys, shapes, fps, finite/NaN counts, and value ranges
- Optional scatter plot with landmark indices for a chosen frame
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def load_npz_any(path):
    """Return (arr, vis, fps, meta) where:
       arr: (T, 33, 2|3) ndarray
       vis: (T, 33) or (T, 33, 1) ndarray or None
       fps: float or None
       meta: dict with 'keys' (list[str]) and 'picked_key' (str)
    """
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())

    # Primary: your layout
    if "kps_xyz" in d:
        arr = d["kps_xyz"]
        vis = d["visibility"] if "visibility" in d else None
        fps = float(d["fps"]) if "fps" in d else None
        picked = "kps_xyz"
        return arr, vis, fps, {"keys": keys, "picked_key": picked}

    # Common fallbacks
    for k in ("pose", "keypoints", "landmarks", "arr_0"):
        if k in d:
            arr = d[k]
            picked = k
            return arr, None, None, {"keys": keys, "picked_key": picked}

    # Last resort: first ndarray with >=2 dims
    for k in keys:
        v = d[k]
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return v, None, None, {"keys": keys, "picked_key": k}

    raise KeyError(
        f"No array-like pose found in npz. Keys present: {keys}"
    )


def summarize_array(arr, vis=None):
    """Compute robust summary strings for printing."""
    T = arr.shape[0]
    J = arr.shape[1]
    D = arr.shape[2] if arr.ndim >= 3 else None

    finite_all = np.isfinite(arr).all()
    finite_ratio = np.isfinite(arr).sum() / arr.size

    # Only look at X,Y for min/max (avoid depth blowing ranges)
    xy = arr[..., :2]
    xy_min = np.nanmin(xy, axis=(0, 1))
    xy_max = np.nanmax(xy, axis=(0, 1))

    vis_summary = None
    if vis is not None:
        v = vis
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v[..., 0]
        try:
            vis_min = np.nanmin(v)
            vis_max = np.nanmax(v)
            vis_mean = np.nanmean(v)
            vis_summary = f"visibility: shape={vis.shape} min={vis_min:.3f} max={vis_max:.3f} mean={vis_mean:.3f}"
        except ValueError:
            vis_summary = f"visibility: shape={vis.shape} (empty/NaN only)"

    return {
        "shape": (T, J, D),
        "finite_all": finite_all,
        "finite_ratio": finite_ratio,
        "xy_min": xy_min,
        "xy_max": xy_max,
        "vis_summary": vis_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to *.npz file")
    ap.add_argument("--frame", type=int, default=0, help="frame index for scatter view")
    ap.add_argument("--scatter", action="store_true", help="show scatter of landmarks with indices")
    ap.add_argument("--invert-y", action="store_true", help="apply (1 - y) before plotting (useful if normalized/upsidedown)")
    ap.add_argument("--save", help="optional path to save the scatter as an image (e.g., viz/frame_100.png)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window when plotting (useful with --save)")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print(f"File not found: {args.npz}", file=sys.stderr)
        sys.exit(2)

    try:
        arr, vis, fps, meta = load_npz_any(args.npz)
    except Exception as e:
        print(f"Error reading {args.npz}: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Loaded: {args.npz}")
    print(f"keys: {meta['keys']}")
    print(f"picked key: {meta['picked_key']}")

    if arr.ndim != 3:
        print(f"WARNING: expected 3D array (T,33,2|3) but got shape {arr.shape}")

    if arr.shape[1] != 33:
        print(f"WARNING: expected 33 landmarks, got {arr.shape[1]}")

    fps_str = "unknown"
    try:
        # Float cast might fail if fps is None or not numeric
        fps_str = f"{float(fps):.3f}" if fps is not None else "None"
    except Exception:
        pass

    summary = summarize_array(arr, vis=vis)
    print(f"shape (T,J,D): {summary['shape']}")
    print(f"fps: {fps_str}")
    print(f"finite_all: {summary['finite_all']} (ratio={summary['finite_ratio']:.6f})")
    print(f"x range: [{summary['xy_min'][0]:.6f}, {summary['xy_max'][0]:.6f}]")
    print(f"y range: [{summary['xy_min'][1]:.6f}, {summary['xy_max'][1]:.6f}]")
    if summary["vis_summary"]:
        print(summary["vis_summary"])

    # Optional scatter
    if args.scatter:
        T = arr.shape[0]
        f = max(0, min(args.frame, T - 1))
        pts = arr[f, :, :2].copy()

        # Heuristic: if values look like normalized [0,1] and appear upside-down,
        # --invert-y can help. We leave the decision to the CLI flag.
        if args.invert_y:
            pts[:, 1] = 1.0 - pts[:, 1]

        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(pts[:, 0], pts[:, 1], s=20)
        for i, (x, y) in enumerate(pts):
            ax.text(x, y, str(i), fontsize=8)
        # Image-style orientation (y down) often feels more natural for pose
        ax.invert_yaxis()
        ax.set_title(f"{os.path.basename(args.npz)}  |  frame {f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            plt.savefig(args.save, bbox_inches="tight", dpi=150)
            print(f"Saved scatter to: {args.save}")

        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
