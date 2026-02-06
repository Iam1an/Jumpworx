#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTRACT (INFERENCE ADAPTER)
============================
Loads a trained model bundle (joblib) and classifies 1+ .posetrack.npz files.

Supported model bundle formats:

1) Legacy dict bundle (from scripts/train_zonly.py):
      {
        "model": sklearn Pipeline,
        "feature_names": FEATURE_KEYS,
        "class_names": list(pipe.classes_)  # ordering for predict_proba
      }

2) Plain sklearn Pipeline / estimator:
      joblib.dump(pipeline, "models/jumpworx_model.joblib")

In both cases we:

  - Use jwcore.pose_utils.features_from_posetrack as the single source of truth.
  - Expose an importable TrickClassifier for programmatic use.
  - Provide a CLI:

      python -m jwcore.trick_classifier --model models/jumpworx_model.joblib \
        --npz cache/TRICK16_FRONTFLIP.posetrack.npz --summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np

from jwcore.posetrack_io import load_posetrack_npz
from jwcore.pose_utils import (
    features_from_posetrack,
    FEATURE_KEYS,
)

# =========================
# Defaults
# =========================

# Repo root = .../Jumpworx (one level above jwcore/)
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Base path used when no explicit model path is given.
# TrickClassifier() with no args will try to resolve this.
DEFAULT_MODEL_BASE = _REPO_ROOT / "models" / "jumpworx_model"


# =========================
# Helpers
# =========================

def _stem(p: str) -> str:
    b = os.path.basename(p)
    return os.path.splitext(b)[0]


# =========================
# Core classifier wrapper
# =========================

class TrickClassifier:
    """
    Importable classifier adapter for use in run_master, demo_cli, and this CLI.

    It hides the differences between:
      - legacy dict bundles: {"model"/"pipeline", "feature_names"/"feature_keys", "class_names", ...}
      - plain sklearn Pipeline / estimator objects (e.g., jumpworx_model.joblib)

    Attributes:
      - pipe         : underlying sklearn model/pipeline
      - feature_keys : ordered list of feature names
      - class_names  : ordered list of class labels (if available)

    API:
      clf = TrickClassifier()  # uses default models/jumpworx_model.joblib
      clf = TrickClassifier("models/jumpworx_model")  # or any base/path
      label = clf.predict(features_dict)
      label, proba = clf.predict_with_proba(features_dict)
    """

    def __init__(self, model_path_or_base: Optional[str] = None):
        # If no model path is given, fall back to the default bundle under models/
        if model_path_or_base is None:
            base = str(DEFAULT_MODEL_BASE)
        else:
            base = model_path_or_base

        self.model_path = self._resolve_model_path(base)
        bundle = joblib.load(self.model_path)

        # Accept both bare pipeline and dict bundle
        if isinstance(bundle, dict):
            self.pipe = bundle.get("model") or bundle.get("pipeline")
            self.feature_keys = (
                bundle.get("feature_names")
                or bundle.get("feature_keys")
                or list(FEATURE_KEYS)
            )
            self.class_names = bundle.get("class_names")
        else:
            self.pipe = bundle
            self.feature_keys = list(FEATURE_KEYS)
            self.class_names = None

        if self.pipe is None:
            raise RuntimeError(
                f"TrickClassifier: no model pipeline found in bundle '{self.model_path}'"
            )

        # Fallback: if class_names missing, try underlying estimator
        if (not self.class_names) and hasattr(self.pipe, "classes_"):
            try:
                self.class_names = list(self.pipe.classes_)
            except Exception:
                self.class_names = None

    # ---------- path resolution ----------

    def _resolve_model_path(self, base: str) -> str:
        """
        Support:
          - explicit file path
          - base + .pkl
          - base + .joblib
          - directory containing a .pkl/.joblib (pick first sorted)
        """
        # Allow Path-like bases
        base = os.fspath(base)

        candidates = [base, base + ".pkl", base + ".joblib"]

        for c in candidates:
            if os.path.isdir(c):
                hits = [
                    os.path.join(c, p)
                    for p in os.listdir(c)
                    if p.endswith(".pkl") or p.endswith(".joblib")
                ]
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    return sorted(hits)[0]
            elif os.path.exists(c):
                return c

        raise FileNotFoundError(
            f"TrickClassifier: could not resolve model path from base '{base}'. "
            f"Tried: {', '.join(candidates)}"
        )

    # ---------- feature vector helper ----------

    def _to_row(self, features: Dict) -> np.ndarray:
        """
        Map a feature dict -> 1 x D vector in the expected order.
        Missing or bad values become 0.0 (consistent with training).
        """
        vals: List[float] = []
        for k in self.feature_keys:
            v = features.get(k, 0.0)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(0.0)
        return np.array([vals], dtype=np.float32)

    # ---------- public API ----------

    def predict(self, features: Dict) -> str:
        """
        Predict label for a single feature dict.
        """
        X = self._to_row(features)
        y = self.pipe.predict(X)[0]
        return str(y)

    def predict_with_proba(self, features: Dict) -> Tuple[str, Optional[float]]:
        """
        Predict label and scalar probability for that label.

        Behavior:
          - If underlying model has predict_proba:
              * Use its output and map using:
                    pipe.classes_      (preferred)
                    self.class_names   (if provided)
              * If mapping fails, fall back to max(proba_vec)
          - If no predict_proba:
              * Return (label, None)

        Returns:
          (label, proba or None)
        """
        label = self.predict(features)

        # No probability support
        if not hasattr(self.pipe, "predict_proba"):
            return label, None

        X = self._to_row(features)
        proba_vec = self.pipe.predict_proba(X)[0]

        # Try mapping using classes_ from the underlying estimator
        if hasattr(self.pipe, "classes_"):
            try:
                classes = list(self.pipe.classes_)
                if len(classes) == len(proba_vec) and label in classes:
                    idx = classes.index(label)
                    return label, float(proba_vec[idx])
            except Exception:
                pass

        # Try mapping using stored class_names from bundle
        if self.class_names is not None:
            try:
                classes = list(self.class_names)
                if len(classes) == len(proba_vec) and label in classes:
                    idx = classes.index(label)
                    return label, float(proba_vec[idx])
            except Exception:
                pass

        # Fallback: use max probability as a generic confidence
        try:
            return label, float(np.max(proba_vec))
        except Exception:
            return label, None

    # ---------- (optional) per-class probabilities ----------

    def predict_proba_vector(self, features: Dict) -> Optional[Dict[str, float]]:
        """
        Return a dict of {class_name: prob} if available, else None.

        Used by the CLI for pretty summaries. Not required by run_master.
        """
        if not hasattr(self.pipe, "predict_proba"):
            return None

        if not self.class_names:
            # If we don't have names, try classes_
            if hasattr(self.pipe, "classes_"):
                class_names = list(self.pipe.classes_)
            else:
                return None
        else:
            class_names = list(self.class_names)

        X = self._to_row(features)
        proba_vec = self.pipe.predict_proba(X)[0]

        if len(proba_vec) != len(class_names):
            return None

        return {str(c): float(p) for c, p in zip(class_names, proba_vec)}


# =========================
# CLI helper
# =========================

def predict_one(clf, npz_path: str) -> Dict:
    """
    Load NPZ, compute features, run classifier via TrickClassifier.
    Returns a dict with:
      - clip
      - npz_path
      - pred_label
      - probs: full per-class dict if available, else {label: proba} or None
      - features: flat feature dict used
    """
    P, V, _fps, meta = load_posetrack_npz(npz_path)
    feats = features_from_posetrack(P, V, meta)

    # Core prediction
    label, scalar_proba = clf.predict_with_proba(feats)

    # Best-effort full probability dict
    probs = clf.predict_proba_vector(feats)
    if probs is None and scalar_proba is not None:
        # At least expose scalar for predicted class
        probs = {str(label): float(scalar_proba)}

    return {
        "clip": _stem(npz_path),
        "npz_path": npz_path,
        "pred_label": label,
        "probs": probs,
        "features": {k: float(feats.get(k, 0.0)) for k in clf.feature_keys},
    }


# =========================
# CLI entrypoint
# =========================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Classify .posetrack.npz with a trained trick model (dict bundle or bare pipeline)."
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Path or base path to joblib model bundle "
             "(e.g. models/trick_model_zonly.pkl or models/jumpworx_model.joblib)",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", action="append", help="Repeatable: specific .npz paths")
    g.add_argument("--glob", type=str, help='Glob for .npz files, e.g. "cache/*.posetrack.npz"')
    ap.add_argument("--jsonl_out", type=str, help="Optional: write per-clip JSON lines")
    ap.add_argument("--summary", action="store_true", help="Print one-line summary per clip")

    args = ap.parse_args(argv)

    try:
        clf = TrickClassifier(args.model)
    except Exception as e:
        print(f"ERROR: failed to load model from '{args.model}': {e}", file=sys.stderr)
        return 2

    # Collect input NPZ paths
    paths: List[str] = sorted(glob.glob(args.glob)) if args.glob else (args.npz or [])
    if not paths:
        print("No input .npz files.", file=sys.stderr)
        return 2

    out_f = open(args.jsonl_out, "w") if args.jsonl_out else None

    try:
        for p in paths:
            try:
                res = predict_one(clf, p)
            except Exception as e:
                print(f"ERROR {p}: {e}", file=sys.stderr)
                continue

            if args.summary:
                probs = res.get("probs")
                if probs:
                    # If we have class_names on clf, respect that ordering for stability
                    if getattr(clf, "class_names", None):
                        ordered = [f"{c}={probs.get(c, 0.0):.3f}" for c in clf.class_names]
                    else:
                        ordered = [f"{c}={probs[c]:.3f}" for c in sorted(probs.keys())]
                    pr = ", ".join(ordered)
                else:
                    pr = "(no probabilities)"
                print(f"{_stem(p):35s} -> {str(res['pred_label']):10s} | {pr}")
            else:
                print(json.dumps(res, indent=2))

            if out_f:
                out_f.write(json.dumps(res) + "\n")

    finally:
        if out_f:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
