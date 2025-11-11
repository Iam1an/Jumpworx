#!/usr/bin/env python3
"""
Evaluate Jumpworx trick classifier against data/labels.csv using precomputed features.

- Reads data/labels.csv with columns: clip_id,label,video_path
- Loads features from features/<clip_id>.json (expects ZONLY 12-key schema at top level)
- Uses TrickClassifier(model_base) where model_base may be either:
    "models/trick_model_zonly"       (preferred)  OR
    "models/trick_model_zonly.pkl"   (will be trimmed to base)
- Prints per-row results and summary metrics (if scikit-learn is available)
- Writes CSV to data/predictions_eval_fast.csv

Usage:
  python scripts/eval_from_labels.py
"""

from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

from jwcore.trick_classifier import TrickClassifier

# ---------------------------------------------------------------------------
# Config
MODEL_PATH = "models/trick_model_zonly"     # base path preferred (no extension)
LABELS_CSV = "data/labels.csv"
FEATURES_DIR = "features"
OUT_CSV = "data/predictions_eval_fast.csv"
REQUIRED_KEYS = [
    "airtime_s","height_max","height_mean","angle_range","angle_speed",
    "rotation_sign","n_frames","fps","pitch_total","pitch_rate",
    "knees_z_delta","shoulderhip_z_slope",
]
# ---------------------------------------------------------------------------


def _model_base(path: str) -> str:
    """Accept base or .pkl and normalize to base."""
    return path[:-4] if path.endswith(".pkl") else path


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _coerce_label(pred):
    """Coerce various return types from TrickClassifier.predict into a single label string."""
    # common: pred already a string
    if isinstance(pred, str):
        return pred
    # sometimes list/tuple/ndarray with one item
    try:
        from numpy import ndarray
        if isinstance(pred, (list, tuple, ndarray)):
            return pred[0] if len(pred) else None
    except Exception:
        if isinstance(pred, (list, tuple)):
            return pred[0] if len(pred) else None
    # some buggy paths may return a set
    if isinstance(pred, set):
        return next(iter(pred)) if pred else None
    # fallback to str
    return str(pred) if pred is not None else None


def main():
    # ensure paths exist
    if not os.path.exists(LABELS_CSV):
        print(f"ERROR: labels file not found: {LABELS_CSV}", file=sys.stderr)
        sys.exit(2)
    Path(Path(OUT_CSV).parent).mkdir(parents=True, exist_ok=True)

    # load classifier (normalize to model base)
    model_base = _model_base(MODEL_PATH)
    clf = TrickClassifier(model_base)
    classes = getattr(clf, "classes_", None)
    if classes is None:
        print("WARNING: classes_ is None (model may not have loaded correctly).", file=sys.stderr)
    else:
        print(f"Loaded model with classes: {classes}")

    rows = []
    y_true, y_pred = [], []

    with open(LABELS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            clip_id = r.get("clip_id")
            actual = r.get("label")
            feat_path = os.path.join(FEATURES_DIR, f"{clip_id}.json")

            if not os.path.exists(feat_path):
                print(f"⚠️  Missing features for {clip_id}: {feat_path} — skipping")
                continue

            d = _load_json(feat_path)

            # Prefer flat top-level keys (your current writer also includes a nested copy)
            feats = d
            # If you ever wanted to force nested:
            # feats = d.get("features", d)

            # Quick schema check (warn only)
            missing = [k for k in REQUIRED_KEYS if k not in feats]
            if missing:
                print(f"⚠️  {clip_id}: features missing {len(missing)} keys; e.g., {missing[:6]}")

            # Predict (robust)
            pred = _coerce_label(clf.predict(feats))

            # Prefer proba map for top label + confidence
            conf = None
            try:
                pmap = clf.predict_proba_dict(feats)  # {label: prob}
                if isinstance(pmap, dict) and pmap:
                    top = max(pmap, key=pmap.get)
                    pred = top
                    conf = float(pmap[top])
            except Exception:
                # generic proba fallback (optional)
                try:
                    pv = clf.predict_proba(feats)
                    # handle 1-D or 2-D returns
                    if hasattr(pv, "__len__"):
                        if hasattr(pv, "shape") and len(getattr(pv, "shape", [])) == 2 and pv.shape[0] == 1:
                            conf = float(max(pv[0]))
                        else:
                            conf = float(max(pv))
                except Exception:
                    pass

            correct = (pred == actual)
            print(f"{clip_id}: actual={actual}  predicted={pred}"
                  f"{'' if conf is None else f'  proba={conf:.3f}'}"
                  f"{'  ✅' if correct else '  ❌'}")

            rows.append({
                "clip_id": clip_id,
                "actual": actual,
                "predicted": pred,
                "proba": conf,
                "correct": correct,
            })
            y_true.append(actual)
            y_pred.append(pred)

    # Write CSV
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved {OUT_CSV} with {len(rows)} rows")

    # Summary metrics (if sklearn is available)
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        if rows:
            print("\n--- Classification report ---")
            print(classification_report(y_true, y_pred, zero_division=0))
            labs = sorted(set(y_true + y_pred))
            print("labels:", labs)
            print(confusion_matrix(y_true, y_pred, labels=labs))
    except Exception as e:
        print(f"(sklearn metrics unavailable: {e})")


if __name__ == "__main__":
    main()
