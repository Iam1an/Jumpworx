#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a tiny z-only flip classifier directly from .posetrack.npz files.

Pipeline:
  .posetrack.npz (P,V,meta_json)
    -> jwcore.pose_utils.features_from_posetrack(...)
    -> vector X using FEATURE_KEYS
    -> StandardScaler + LogisticRegression
    -> models/trick_model_zonly.pkl (+ report JSON)

Labels:
  - Inferred from the NPZ stem if not provided via CSV:
      * contains "backflip" => backflip
      * contains "frontflip" => frontflip

Usage:
  python scripts/train_zonly.py --npz cache/TRICK06_BACKFLIP.posetrack.npz \
                                --npz cache/TRICK16_FRONTFLIP.posetrack.npz

  # or:
  python scripts/train_zonly.py --glob "cache/*.posetrack.npz"
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.pose_utils import (
    features_from_posetrack,
    FEATURE_KEYS,
)

def infer_label_from_stem(stem: str) -> Optional[str]:
    s = stem.lower()
    if "backflip" in s or s.startswith("back"):
        return "backflip"
    if "frontflip" in s or s.startswith("front"):
        return "frontflip"
    return None

def collect_examples(paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    X: List[List[float]] = []
    y: List[str] = []
    kept: List[str] = []

    for pth in paths:
        try:
            P, V, _fps, meta = load_posetrack_npz(pth)
            feats = features_from_posetrack(P, V, meta)
        except Exception as e:
            print(f"[skip] {pth}: {e}", file=sys.stderr)
            continue

        stem = os.path.splitext(os.path.basename(pth))[0]
        label = infer_label_from_stem(stem)
        if label not in ("frontflip", "backflip"):
            print(f"[skip] {pth}: could not infer label from filename", file=sys.stderr)
            continue

        X.append([float(feats[k]) for k in FEATURE_KEYS])
        y.append(label)
        kept.append(stem)

    if not X:
        raise RuntimeError("No training examples found after filtering.")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=object)
    classes = ["frontflip", "backflip"]
    return X, y, FEATURE_KEYS, classes

def train_model(X: np.ndarray, y: np.ndarray, classes: List[str]):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=3.0, class_weight="balanced", max_iter=1000)
    )

    if len(y) >= 6 and len(set(y.tolist())) == 2:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        pipe.fit(Xtr, ytr)
        ypred = pipe.predict(Xte)
        report = classification_report(yte, ypred, labels=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(yte, ypred, labels=classes).tolist()
    else:
        # tiny set: fit and evaluate on train (for sanity only)
        pipe.fit(X, y)
        ypred = pipe.predict(X)
        report = classification_report(y, ypred, labels=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, ypred, labels=classes).tolist()

    return pipe, report, cm

def save_artifacts(pipe, features: List[str], classes: List[str], report, cm,
                   model_out: str, report_out: str):
    os.makedirs(os.path.dirname(model_out) or "models", exist_ok=True)
    bundle = {
        "model": pipe,
        "feature_names": features,
        # store the actual learned ordering for predict_proba:
        "class_names": list(pipe.classes_),   # <-- change was here
    }
    joblib.dump(bundle, model_out)
    with open(report_out, "w") as f:
        json.dump({
            "classification_report": report,
            "confusion_matrix": cm,
            "feature_names": features,
            "class_names": list(pipe.classes_),  # <-- and here
        }, f, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Train z-only flip classifier from .posetrack.npz files")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob", type=str, help='Glob of npz files, e.g. "cache/*.posetrack.npz"')
    g.add_argument("--npz", action="append", help="Repeatable: one or more specific npz paths")
    ap.add_argument("--model_out", type=str, default="models/trick_model_zonly.pkl")
    ap.add_argument("--report_out", type=str, default="models/trick_model_zonly_report.json")
    args = ap.parse_args()

    paths: List[str] = sorted(glob.glob(args.glob)) if args.glob else (args.npz or [])
    if not paths:
        print("No input .npz files.", file=sys.stderr)
        return 2

    X, y, feat_names, class_names = collect_examples(paths)
    uniq, cnts = np.unique(y, return_counts=True)
    print("class counts:", dict(zip(uniq.tolist(), cnts.tolist())))

    pipe, report, cm = train_model(X, y, class_names)
    save_artifacts(pipe, feat_names, class_names, report, cm, args.model_out, args.report_out)

    print(f"✅ saved model to {args.model_out}")
    print(f"📄 report -> {args.report_out}")
    print(json.dumps(report, indent=2))
    print("confusion_matrix (rows=true, cols=pred):", cm)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
