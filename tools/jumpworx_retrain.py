#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json

def infer_features(df, label_col, features):
    if features:
        return [f for f in features if f in df.columns]
    return [c for c in df.columns if c != label_col]

def main():
    p = argparse.ArgumentParser(description="Retrain flip classifier on all examples.")
    p.add_argument("--data", required=True, help="CSV with features and 'label' column")
    p.add_argument("--label_col", default="label", help="Name of label column (default: label)")
    p.add_argument("--features", nargs="*", help="Optional list of feature names to use; defaults to all non-label columns")
    p.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction for sanity-check (default 0.2)")
    p.add_argument("--model_out", default="jumpworx_model.joblib", help="Path to save model (default: jumpworx_model.joblib)")
    p.add_argument("--meta_out", default="jumpworx_model_meta.json", help="Path to save model metadata (default: jumpworx_model_meta.json)")
    p.add_argument("--C", type=float, default=2.0, help="Inverse regularization for LogisticRegression (default 2.0)")
    p.add_argument("--max_iter", type=int, default=2000, help="Max iterations (default 2000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    assert args.label_col in df.columns, f"Label column '{args.label_col}' not in CSV."

    feat_cols = infer_features(df, args.label_col, args.features)
    if not feat_cols:
        raise ValueError("No feature columns found. Pass --features or include features in CSV.")

    X = df[feat_cols].astype(float).values
    y = df[args.label_col].values

    # Handle class imbalance via class_weight='balanced'
    classes = np.unique(y)
    if y.dtype.kind in ("U","S","O"):
        # coerce to string for consistent metadata
        classes_sorted = np.array(sorted(classes.tolist()))
    else:
        classes_sorted = classes

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=args.C, max_iter=args.max_iter, class_weight="balanced", multi_class="auto",
            solver="lbfgs"))
    ])

    # quick sanity split (training will use full set; split is only to report a quick check)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    pipe.fit(Xtr, ytr)
    acc_te = pipe.score(Xte, yte)

    # Refit on ALL data for the final model
    pipe.fit(X, y)

    joblib.dump(pipe, args.model_out)

    meta = {
        "features": feat_cols,
        "label_col": args.label_col,
        "classes": classes_sorted.tolist(),
        "sanity_holdout_acc": float(acc_te),
        "n_samples": int(len(df)),
        "n_features": int(len(feat_cols))
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"[OK] Saved model to {args.model_out}")
    print(f"[OK] Saved metadata to {args.meta_out}")
    print(f"[INFO] Sanity holdout accuracy: {acc_te:.4f}")
    print(f"[INFO] Used features: {feat_cols}")

if __name__ == "__main__":
    main()
