#!/usr/bin/env python3
"""
DEPRECATED: This script is out of sync with the current TrickClassifier API
and FEATURE_KEYS. Use scripts/train_zonly.py instead.

Known issues (not fixed):
  - USED_FEATURES references names that no longer exist in FEATURE_KEYS
  - Calls TrickClassifier(model_base_path=...) — actual param is model_path_or_base
  - Calls tc.train(), tc.set_params(), tc.predict_many() — none exist on TrickClassifier

Canonical training script: scripts/train_zonly.py
  python scripts/train_zonly.py --glob "cache/*.posetrack.npz"

--- Original docstring below ---

Train (or re-train) the TrickClassifier from labeled clips.

Now defaults to a Z-dominant feature set that cleanly separates frontflip/backflip
from a front-facing camera:
    USED_FEATURES = ["pitch_total", "knees_z_delta", "shoulderhip_z_slope", "pitch_rate"]

Two workflows still supported:
A) From existing feature JSONs (fast)
B) From raw videos (will call your extractors to build features first)

Inputs:
  - labels CSV (required): columns = clip_id,label[,video_path]
  - features_dir (default: features/)
  - videos_dir   (optional: used if features are missing or you want to rebuild)

Outputs:
  - models/<model_name>.pkl
  - models/<model_name>.json
  - models/<model_name>_report.json

Example:
  python scripts/train_trick_model.py \
      --labels_csv data/labels.csv \
      --videos_dir videos/training \
      --model_name trick_model_zonly
"""
from __future__ import annotations

# ---- path bootstrap: must be FIRST lines ----
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__))          # .../Jumpworx/scripts
ROOT = os.path.abspath(os.path.join(HERE, ".."))           # .../Jumpworx
if HERE not in sys.path:
    sys.path.insert(0, HERE)   # allow "import label_utils"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)   # allow "import jwcore"
# ----------------------------------------------------------

import argparse, csv, json, subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# JWCore
from jwcore.trick_classifier import TrickClassifier
from jwcore.pose_utils import FEATURE_KEYS

# local helper (label canonicalization)
try:
    from label_utils import canon_label, TARGET_DEFAULT
except Exception:
    # allow package-style import if scripts/ is a package
    from scripts.label_utils import canon_label, TARGET_DEFAULT


# ===================== Z-dominant feature set =====================
# These four Z cues best separate front/back in front-facing videos.
USED_FEATURES = ["pitch_total", "knees_z_delta", "shoulderhip_z_slope", "pitch_rate"]
# ================================================================


# ------------------------------- IO helpers --------------------------------

@dataclass
class LabeledRow:
    clip_id: str
    label: str
    video_path: Optional[str] = None

def load_labels_csv(path: str) -> List[LabeledRow]:
    rows: List[LabeledRow] = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        need = {"clip_id", "label"}
        if not need.issubset(set(rdr.fieldnames or [])):
            raise SystemExit("labels_csv must have at least columns: clip_id,label[,video_path]")
        for r in rdr:
            rows.append(LabeledRow(
                clip_id=r["clip_id"].strip(),
                label=canon_label(r["label"]),  # canonicalize
                video_path=(r.get("video_path") or "").strip() or None,
            ))
    if not rows:
        raise SystemExit("labels_csv is empty.")
    return rows

def feature_path(features_dir: str, clip_id: str) -> str:
    os.makedirs(features_dir, exist_ok=True)
    return os.path.join(features_dir, f"{clip_id}.json")

def npz_path(cache_dir: str, clip_id: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{clip_id}.posetrack.npz")

def ensure_features_for_clip(
    clip: LabeledRow,
    features_dir: str,
    cache_dir: str,
    videos_dir: Optional[str],
    every: int,
    model_complexity: int,
) -> str:
    """Ensure features/<clip_id>.json exists. If missing and we have a video, build it."""
    fpath = feature_path(features_dir, clip.clip_id)
    if os.path.exists(fpath):
        return fpath

    # Need to (a) have video_path in CSV or (b) be able to infer it from videos_dir
    video = clip.video_path
    if not video and videos_dir:
        # try common names under videos_dir with typical extensions
        base = os.path.join(videos_dir, clip.clip_id)
        for ext in (".mp4", ".MP4", ".mov", ".MOV"):
            cand = base + ext if not clip.clip_id.lower().endswith(ext.lower()) else base
            if os.path.exists(cand):
                video = cand
                break
    if not video or not os.path.exists(video):
        raise SystemExit(f"Missing features for {clip.clip_id} and no usable video_path found.")

    # 1) video -> npz  (extract with Z)
    npz = npz_path(cache_dir, clip.clip_id)
    if not os.path.exists(npz):
        cmd = [
            sys.executable, "scripts/extract_keypoints.py",
            "--video", video,
            "--every", str(every),
            "--model_complexity", str(model_complexity),
        ]
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)
        # Normalize produced name to our convention if needed
        stem = os.path.splitext(os.path.basename(video))[0]
        produced = os.path.join("cache", f"{stem}.posetrack.npz")
        if os.path.exists(produced) and produced != npz:
            os.replace(produced, npz)

    # 2) npz -> features (calls jwcore.pose_extract which prefers kps_xyz)
    if not os.path.exists(fpath):
        cmd = [
            sys.executable, "-m", "jwcore.pose_extract",
            "--posetrack", npz,
            "--out", fpath,
        ]
        print(">>", " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", ROOT)
        subprocess.check_call(cmd, env=env)

    return fpath

def load_feature_row(path: str) -> Tuple[str, Dict[str, float]]:
    obj = json.load(open(path))
    return obj["clip_id"], obj["features"]


# ------------------------------- training ----------------------------------

def collect_dataset(rows: List[LabeledRow], features_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, used = [], [], []
    missing_keys = set()
    for row in rows:
        fpath = feature_path(features_dir, row.clip_id)
        if not os.path.exists(fpath):
            print(f"!! Skipping {row.clip_id}: missing {fpath}")
            continue
        clip_id, feats = load_feature_row(fpath)

        # Verify required keys exist; accumulate missing for debug
        for k in USED_FEATURES:
            if k not in feats:
                missing_keys.add(k)

        # Build vector only from Z-dominant subset
        X.append([float(feats.get(k, 0.0)) for k in USED_FEATURES])
        y.append(canon_label(row.label))
        used.append(clip_id)

    if missing_keys:
        print(f"!! Warning: some feature keys missing in inputs: {sorted(missing_keys)} "
              f"(defaulted to 0.0 where absent)")

    if not X:
        raise SystemExit("No feature rows collected. Did you build features? Check --features_dir and CSV clip_id names.")
    return np.array(X, np.float32), np.array(y), used


def save_report(model_base: str, y_true: np.ndarray, y_pred: np.ndarray, labels_used: List[str]) -> str:
    rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred).tolist()
    # explicitly include sorted labels for downstream plotting scripts
    labels_sorted = sorted(list(set(labels_used)))
    out = {
        "labels_used": labels_used,
        "labels": labels_sorted,
        "classification_report": rep,
        "confusion_matrix": cm,
        "features_used": USED_FEATURES,
    }
    os.makedirs(os.path.dirname(model_base) or ".", exist_ok=True)
    path = model_base + "_report.json"
    json.dump(out, open(path, "w"), indent=2)
    return path


def main():
    import warnings
    warnings.warn(
        "train_trick_model.py is DEPRECATED and will not produce a working model. "
        "Use scripts/train_zonly.py instead. See docs/ml/PIPELINE.md for details.",
        DeprecationWarning,
        stacklevel=2,
    )
    ap = argparse.ArgumentParser(description="Train (re-train) TrickClassifier with Z-dominant features.")
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: clip_id,label[,video_path]")
    ap.add_argument("--features_dir", default="features", help="Where feature JSONs live (or will be written)")
    ap.add_argument("--videos_dir", help="If provided, will build missing features from these videos")
    ap.add_argument("--cache_dir", default="cache", help="Where to store pose-track .npz")
    ap.add_argument("--model_name", default="trick_model_zonly", help="Base name under models/")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=1, help="Stride for keypoint extraction when building from videos")
    ap.add_argument("--model_complexity", type=int, choices=[0,1,2], default=1, help="MediaPipe model complexity")
    # Parameters that TrickClassifier may accept (pipeline uses LogisticRegression by default)
    ap.add_argument("--class_weight", default=None, help="e.g., 'balanced' (string) or leave None")
    ap.add_argument("--n_estimators", type=int, default=None, help="Only used if underlying estimator supports it")
    args = ap.parse_args()

    rows = load_labels_csv(args.labels_csv)

    # Ensure features exist (build if needed)
    for r in rows:
        fpath = feature_path(args.features_dir, r.clip_id)
        if not os.path.exists(fpath):
            if not args.videos_dir and not r.video_path:
                raise SystemExit(f"Missing {fpath} and no video path provided for {r.clip_id}. "
                                 "Provide --videos_dir or add video_path to CSV.")
            ensure_features_for_clip(
                r, args.features_dir, args.cache_dir, args.videos_dir,
                every=args.every, model_complexity=args.model_complexity
            )

    # Collect dataset (Z features only)
    X, y, used_clips = collect_dataset(rows, args.features_dir)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Configure classifier with our subset order
    model_base = os.path.join("models", args.model_name)
    tc = TrickClassifier(model_base_path=model_base, feature_names=list(USED_FEATURES))

    # Optional: forward params if supported (e.g., class_weight for LogisticRegression)
    try:
        if args.class_weight is not None:
            tc.set_params(class_weight=args.class_weight)
        if args.n_estimators is not None:
            tc.set_params(n_estimators=int(args.n_estimators))
    except Exception:
        pass

    # Train
    Xtr_dicts = [dict(zip(USED_FEATURES, row)) for row in Xtr]
    info = tc.train(Xtr_dicts, list(ytr))
    print("Saved model:", info["saved_pkl"])
    print("Saved meta :", info["saved_json"])

    # Evaluate
    Xte_dicts = [dict(zip(USED_FEATURES, row)) for row in Xte]
    try:
        yhat = np.asarray(tc.predict_many(Xte_dicts))
    except AttributeError:
        yhat = np.asarray([tc.predict(d) for d in Xte_dicts])

    report_path = save_report(model_base, yte, yhat, used_clips)
    print("Saved report:", report_path)

    # Dataset summary + quick preview
    classes, counts = np.unique(y, return_counts=True)
    print(f"\nDataset: {X.shape} classes: " + "{" + ", ".join(f"'{c}': {n}" for c, n in zip(classes, counts)) + "}")
    print("\nPreview (first 8):")
    for i in range(min(8, len(yte))):
        print(f"  true={yte[i]:10s} pred={yhat[i]:10s}")


if __name__ == "__main__":
    main()
