#!/usr/bin/env python3
"""
predict_new.py

Single-clip inference for Jumpworx.

Modes:
  1) --features <features.json>          # classify from existing features
  2) --video <path> --clip_id <ID>       # extract NPZ -> features -> classify

Notes:
- Uses TrickClassifier(model_base_path) to load the model.
- When using --video, we:
    a) call scripts/extract_keypoints.py, passing --out cache  (a DIR),
    b) write/expect cache/<clip_id>.posetrack.npz (a FILE),
    c) call: python -m jwcore.pose_extract <npz_path> <feat_path> (positional args)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# --- Core classifier
from jwcore.trick_classifier import TrickClassifier


def run(cmd: list[str]) -> None:
    """Run a subprocess and raise if it fails, echoing the command."""
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_features_from_video(
    video_path: str,
    clip_id: str,
    every: int = 1,
    model_complexity: int = 1,
    cache_dir: str = "cache",
    features_dir: str = "features",
) -> str:
    """
    From a raw video, ensure we have:
      cache/<clip_id>.posetrack.npz  and  features/<clip_id>.json
    Returns the features path.
    """
    video_path = str(video_path)
    cache_dir = str(cache_dir)
    features_dir = str(features_dir)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(features_dir).mkdir(parents=True, exist_ok=True)

    npz_path = os.path.join(cache_dir, f"{clip_id}.posetrack.npz")
    feat_path = os.path.join(features_dir, f"{clip_id}.json")

    # 1) Extract keypoints -> NPZ (pass directory as --out)
    if not os.path.isfile(npz_path):
        run([
            sys.executable, "scripts/extract_keypoints.py",
            "--video", video_path,
            "--every", str(every),
            "--model_complexity", str(model_complexity),
            "--out", cache_dir
        ])
        # guard against accidental ".npz" directory creation
        if os.path.isdir(npz_path):
            raise RuntimeError(
                f"Expected NPZ file, but found a directory: {npz_path}\n"
                "Edit scripts/extract_keypoints.py to ensure --out is treated as a directory, "
                "or pass --out cache (a directory)."
            )

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found after extraction: {npz_path}")

    # 2) NPZ -> features JSON (pose_extract uses POSITIONAL args)
    if not os.path.isfile(feat_path):
        run([sys.executable, "-m", "jwcore.pose_extract", npz_path, feat_path])

    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"Features JSON not found after pose_extract: {feat_path}")

    return feat_path


def load_features(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True, help="Path to .pkl model (e.g., models/trick_model_zonly.pkl)")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--features", help="Existing features JSON to classify")
    group.add_argument("--video", help="Raw video to process, then classify")

    ap.add_argument("--clip_id", help="Logical ID (stem) used for cache/features filenames (required if --video)")
    ap.add_argument("--every", type=int, default=1, help="Frame subsampling for extraction (default: 1)")
    ap.add_argument("--model_complexity", type=int, choices=[0,1,2], default=1, help="MediaPipe model complexity")
    ap.add_argument("--show_proba", action="store_true", help="Print class probabilities (if available)")
    ap.add_argument("--threshold", type=float, default=None, help="Optional decision threshold for printing")
    args = ap.parse_args()

    # Sanity: if using video, clip_id is required
    if args.video and not args.clip_id:
        ap.error("--clip_id is required when using --video")

    # Load model
    model_base = args.model_base[:-4] if args.model_base.endswith(".pkl") else args.model_base
    clf = TrickClassifier(model_base)

    # Resolve features
    if args.features:
        feat_path = args.features
    else:
        feat_path = ensure_features_from_video(
            video_path=args.video,
            clip_id=args.clip_id,
            every=args.every,
            model_complexity=args.model_complexity,
            cache_dir="cache",
            features_dir="features",
        )

    feats = load_features(feat_path)

    # Predict
    pred = clf.predict(feats)

    # Probability (optional) + robust label selection
    proba_str = ""
    top_label = None
    try:
        pmap = clf.predict_proba_dict(feats)  # {label: prob}
        if isinstance(pmap, dict) and pmap:
            top_label = max(pmap, key=pmap.get)
            # pretty print
            ordered = sorted(pmap.items(), key=lambda kv: kv[1], reverse=True)
            proba_str = " | " + ", ".join(f"{k}:{v:.3f}" for k, v in ordered[:6])
    except Exception:
        pmap = None

    # Coerce pred to a string label robustly
    if isinstance(pred, (list, tuple)):
        pred = pred[0] if pred else None
    if isinstance(pred, set):
        pred = next(iter(pred)) if pred else None
    if pmap and top_label is not None:
        pred = top_label  # trust the model’s highest-prob label

    # Threshold handling / printing
    if args.show_proba:
        if args.threshold is not None and pmap and top_label and pmap[top_label] < args.threshold:
            print(f"Prediction: {pred} (conf {pmap[top_label]:.3f} < threshold {args.threshold:.3f})")
        else:
            if proba_str:
                print(f"Prediction: {pred} {proba_str}")
            else:
                print(f"Prediction: {pred}")
    else:
        print(f"Prediction: {pred}")



if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"\nSubprocess failed (exit {e.returncode}): {' '.join(e.cmd)}\n")
        sys.exit(e.returncode)
    except Exception as e:
        sys.stderr.write(f"\nERROR: {e}\n")
        sys.exit(2)
