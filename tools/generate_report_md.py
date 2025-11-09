#!/usr/bin/env python3
import json, sys

SPECIAL_KEYS = {"accuracy", "macro avg", "weighted avg"}

def infer_labels(r):
    if "labels" in r and r["labels"]:
        return r["labels"]
    cr = r.get("classification_report", {})
    keys = [k for k in cr.keys() if k not in SPECIAL_KEYS]
    return list(keys) if keys else []

TEMPLATE = """# Trick Model Report

**Timestamp:** {ts}

## Labels
{labels}

## Summary
- **Accuracy (micro-F1):** {acc:.3f}
- **Classes:** {ncls}
- **AUC (label-based):** {auc}

## Per-Class Metrics
{perclass}

## Confusion Matrix
Raw: `viz/confusion_matrix.png`  
Normalized: `viz/confusion_matrix_normalized.png`
"""

def main():
    report = sys.argv[1] if len(sys.argv) > 1 else "models/trick_model_v2_report.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "models/trick_model_v2_report.md"

    with open(report) as f:
        r = json.load(f)

    labels = infer_labels(r)
    rep = r.get("classification_report", {})
    acc = float(rep.get("accuracy", 0.0))
    auc = r.get("auc_label_based")
    auc_str = "n/a" if auc is None else f"{auc:.3f}"

    # Prefer precomputed per_class; otherwise derive from classification_report
    per_class = r.get("per_class", {})
    if not per_class and labels:
        per_class = {}
        for lbl in labels:
            m = rep.get(lbl, {})
            per_class[lbl] = {
                "precision": float(m.get("precision", 0.0)),
                "recall": float(m.get("recall", 0.0)),
                "f1": float(m.get("f1-score", m.get("f1", 0.0))),
                "support": int(m.get("support", 0)),
            }

    if not labels and "confusion_matrix" in r:
        # last resort: count classes from CM
        n = len(r["confusion_matrix"])
        labels = [str(i) for i in range(n)]

    pcs_lines = []
    for lbl in labels:
        m = per_class.get(lbl, {"precision":0.0,"recall":0.0,"f1":0.0,"support":0})
        pcs_lines.append(f"- **{lbl}** — P: {m['precision']:.3f}, R: {m['recall']:.3f}, F1: {m['f1']:.3f}, N={m['support']}")

    content = TEMPLATE.format(
        ts=r.get("ts",""),
        labels=", ".join(labels) if labels else "(labels unavailable)",
        acc=acc,
        ncls=len(labels) if labels else 0,
        auc=auc_str,
        perclass="\n".join(pcs_lines) if pcs_lines else "(no per-class metrics found)"
    )

    with open(out, "w") as f:
        f.write(content)
    print("Wrote", out)

if __name__ == "__main__":
    main()
