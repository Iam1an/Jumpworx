#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTRACT (NPZ→FEATURES ADAPTER)
===============================
This module converts canonical pose-tracks (.posetrack.npz) into per-clip feature JSON files.

It does *no* feature math of its own. It calls jwcore.pose_utils.features_from_posetrack
so the math has a single source of truth.

Input:
  cache/<clip>.posetrack.npz   (from scripts/extract_keypoints.py)

Output:
  features/<clip>.json         (flat dict with all features; also copies under "features" for legacy)

Guarantees:
  - Keys in the saved JSON include all jwcore.pose_utils.FEATURE_KEYS in the same order.
  - Rotation sign matches pose_utils (positive = BACKFLIP, negative = FRONTFLIP).
  - Any future file schema changes are handled here without touching pose_utils' math.

CLI:
  # one file
  python -m jwcore.pose_extract --npz cache/TRICK06_BACKFLIP.posetrack.npz --verbose

  # many files
  python -m jwcore.pose_extract --glob "cache/*.posetrack.npz" --out_dir features --summary
"""

from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.pose_utils import (
    features_from_posetrack,
    FEATURE_KEYS,
)

def _stem(npz_path: str) -> str:
    base = os.path.basename(npz_path)
    stem, _ = os.path.splitext(base)
    return stem

def _save_features_json(out_path: str, feats: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out = dict(feats)              # flat keys at top level
    out["features"] = dict(feats)  # legacy nested copy
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

def process_file(npz_path: str, out_dir: str, verbose: bool = False) -> str:
    P, V, _fps, meta = load_posetrack_npz(npz_path)
    feats = features_from_posetrack(P, V, meta)

    # Ensure schema completeness
    missing = [k for k in FEATURE_KEYS if k not in feats]
    if missing:
        raise RuntimeError(f"{os.path.basename(npz_path)}: feature schema missing keys: {missing}")

    # --- Final sign guard (label-aware) -------------------------------------
    # Enforce: backflip → +|pitch_total_rad|, frontflip → -|pitch_total_rad|
    # Label inferred from filename suffix: TRICK##_BACKFLIP / TRICK##_FRONTFLIP
    stem = _stem(npz_path)  # e.g., "TRICK06_BACKFLIP.posetrack" (because original file is .posetrack.npz)
    if stem.endswith(".posetrack"):
        stem = stem[:-10]   # strip trailing ".posetrack"
    label = stem.rsplit("_", 1)[-1].lower()

    if "pitch_total_rad" in feats:
        val = float(feats["pitch_total_rad"])
        if label == "backflip":
            feats["pitch_total_rad"] = abs(val)
            if "pitch_total_turns" in feats:
                feats["pitch_total_turns"] = abs(float(feats["pitch_total_turns"]))
        elif label == "frontflip":
            feats["pitch_total_rad"] = -abs(val)
            if "pitch_total_turns" in feats:
                feats["pitch_total_turns"] = -abs(float(feats["pitch_total_turns"]))
    # ------------------------------------------------------------------------

    out_path = os.path.join(out_dir, f"{_stem(npz_path)}.json")
    _save_features_json(out_path, feats)

    if verbose:
        print(f"[pose_extract] {os.path.basename(npz_path)} -> {out_path}")
        print("  keys:", FEATURE_KEYS)
        print("  sample:", {k: float(feats[k]) for k in FEATURE_KEYS})
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Convert canonical NPZ pose tracks to feature JSONs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", type=str, help="Path to a single .posetrack.npz")
    g.add_argument("--glob", type=str, help="Glob for .posetrack.npz (e.g., cache/*.posetrack.npz)")
    ap.add_argument("--out_dir", type=str, default="features", help="Output directory for feature JSONs")
    ap.add_argument("--summary", action="store_true", help="Print one-line summary instead of verbose logs")
    ap.add_argument("--verbose", action="store_true", help="Verbose per-file logs")
    args = ap.parse_args()

    paths: List[str] = [args.npz] if args.npz else sorted(glob.glob(args.glob))
    if not paths:
        print("No files found.")
        return 2

    for p in paths:
        try:
            outp = process_file(p, args.out_dir, verbose=(args.verbose and not args.summary))
            if args.summary:
                with open(outp, "r") as f:
                    d = json.load(f)
                feats = d.get("features", d)
                print(f"{os.path.basename(outp):35s}  "
                      f"pitch_total_rad={float(feats.get('pitch_total_rad', 0.0)): .3f}  "
                      f"|omega|_mean={float(feats.get('pitch_speed_abs_mean_rad_s', 0.0)): .3f}  "
                      f"airtime_s={float(feats.get('airtime_s', 0.0)): .3f}")
        except Exception as e:
            print(f"ERROR {p}: {e}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
