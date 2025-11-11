#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

def print_confusion(cm, classes):
    # Pretty print confusion matrix
    widths = [max(7, len(str(c)) + 2) for c in classes]
    header = " " * 10 + " ".join(str(c).center(w) for c, w in zip(classes, widths))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(header)
    for i, row in enumerate(cm):
        label = str(classes[i]).ljust(8)
        cells = " ".join(str(v).center(widths[j]) for j, v in enumerate(row))
        print(f"{label}  {cells}")
    print("")

def main():
    p = argparse.ArgumentParser(description="Evaluate saved model with k-fold CV + holdout, save metrics.")
    p.add_argument("--data", required=True, help="CSV with features and 'label' column")
    p.add_argument("--model", default="jumpworx_model.joblib", help="Path to saved model")
    p.add_argument("--meta", default="jumpworx_model_meta.json", help="Path to model meta JSON")
    p.add_argument("--label_col", default="label", help="Name of label column")
    p.add_argument("--features", nargs="*", help="Optional list of feature names; defaults to meta['features'] if present")
    p.add_argument("--kfold", type=int, default=5, help="Number of folds for StratifiedKFold CV")
    p.add_argument("--out_json", default="jumpworx_eval.json", help="Path to write a JSON with summary metrics")
    args = p.parse_args()

    model = joblib.load(args.model)
    meta = {}
    if Path(args.meta).exists():
        meta = json.loads(Path(args.meta).read_text())

    df = pd.read_csv(args.data)
    assert args.label_col in df.columns, f"Label column '{args.label_col}' not found."

    features = args.features or meta.get("features")
    if not features:
        features = [c for c in df.columns if c != args.label_col]

    X = df[features].astype(float).values
    y = df[args.label_col].values
    classes = np.unique(y)

    # Holdout metrics (using the persisted model as-is)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, labels=classes, average="weighted", zero_division=0)
    report = classification_report(y, y_pred, digits=4, zero_division=0)

    cm = confusion_matrix(y, y_pred, labels=classes)

    print("[HOLDOUT on full data vs saved model]")
    print(f"Accuracy: {acc:.4f} | Precision (weighted): {prec:.4f} | Recall (weighted): {rec:.4f} | F1 (weighted): {f1:.4f}")
    print(report)
    print_confusion(cm, classes)

    # K-fold cross-validation from scratch using same feature set and estimator
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    accs, precs, recs, f1s = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # Clone-like behavior via reloading; since our saved model is a Pipeline with fitted params,
        # we just refit a fresh copy from the same type. For simplicity, re-use the loaded model's classes & hyperparams
        import copy
        est = copy.deepcopy(model)
        est.fit(Xtr, ytr)
        yhat = est.predict(Xte)

        a = accuracy_score(yte, yhat)
        p, r, f, _ = precision_recall_fscore_support(yte, yhat, average="weighted", zero_division=0)
        accs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"[CV fold {fold}] acc={a:.4f} prec={p:.4f} rec={r:.4f} f1={f:.4f}")

    summary = {
        "features": features,
        "classes": [str(c) for c in classes],
        "holdout": {"accuracy": acc, "precision_weighted": prec, "recall_weighted": rec, "f1_weighted": f1},
        "cv": {
            "k": args.kfold,
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "precision_mean": float(np.mean(precs)), "recall_mean": float(np.mean(recs)), "f1_mean": float(np.mean(f1s))
        }
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"[OK] Wrote: {args.out_json}")

if __name__ == "__main__":
    main()
