#!/usr/bin/env python3
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt  # no seaborn

SPECIAL_KEYS = {"accuracy", "macro avg", "weighted avg"}

def infer_labels(rep_dict):
    # 1) Preferred: explicit labels
    if "labels" in rep_dict and rep_dict["labels"]:
        return rep_dict["labels"]
    # 2) From classification_report keys
    cr = rep_dict.get("classification_report", {})
    keys = [k for k in cr.keys() if k not in SPECIAL_KEYS]
    if keys:
        return list(keys)
    # 3) Fallback: numeric indices from CM
    cm = np.array(rep_dict.get("confusion_matrix", []))
    if cm.ndim == 2 and cm.shape[0] == cm.shape[1] and cm.shape[0] > 0:
        return [str(i) for i in range(cm.shape[0])]
    # Last resort
    return []

def main():
    report_path = sys.argv[1] if len(sys.argv) > 1 else "models/trick_model_v2_report.json"
    outdir = "viz"
    os.makedirs(outdir, exist_ok=True)

    with open(report_path) as f:
        rep = json.load(f)

    labels = infer_labels(rep)
    cm = np.array(rep["confusion_matrix"])
    cmn = np.array(rep.get("confusion_matrix_normalized")) if rep.get("confusion_matrix_normalized") is not None \
          else (cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None))

    if not labels or len(labels) != cm.shape[0]:
        # Keep going with safe fallbacks
        labels = labels if labels and len(labels) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]

    # Raw CM
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (raw)")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    p1 = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(p1, dpi=180)

    # Normalized CM
    plt.figure()
    plt.imshow(cmn, interpolation="nearest")
    plt.title("Confusion Matrix (normalized)")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    p2 = os.path.join(outdir, "confusion_matrix_normalized.png")
    plt.savefig(p2, dpi=180)

    print("Wrote:", p1, "and", p2)

if __name__ == "__main__":
    main()
