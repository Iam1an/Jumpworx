#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

NUM_KINDS = (int, float)

def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, NUM_KINDS) for x in v) and len(v) <= 8:
                for i, x in enumerate(v):
                    out[f"{key}[{i}]"] = x
        elif isinstance(v, NUM_KINDS):
            out[key] = v
    return out

def load_label_map(labels_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(labels_csv)
    cols = [c.lower() for c in df.columns]
    if "id" in cols and "label" in cols:
        id_col = df.columns[cols.index("id")]
        label_col = df.columns[cols.index("label")]
    elif "trick" in cols and "label" in cols:
        id_col = df.columns[cols.index("trick")]
        label_col = df.columns[cols.index("label")]
    else:
        id_col, label_col = df.columns[:2]
    return {str(row[id_col]): str(row[label_col]) for _, row in df.iterrows()}

def infer_id_from_filename(p: Path) -> str:
    return re.sub(r"\.(json|posetrack\.json)$", "", p.name, flags=re.I)

def collect_features(features_dir: Path, select_features: List[str] = None):
    rows, ids = [], []
    featset = set(select_features) if select_features else None
    for p in sorted(features_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        flat = flatten_dict(data)
        if featset is not None:
            flat = {k: flat.get(k, np.nan) for k in select_features}
        rows.append(flat)
        ids.append(infer_id_from_filename(p))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No feature JSONs found or all failed to parse.")
    df.insert(0, "id", ids)
    # Drop all-NaN/constant columns; keep only numeric
    nunique = df.nunique(dropna=True)
    keep = [c for c in df.columns if c == "id" or (nunique.get(c, 0) > 1)]
    df = df[keep]
    numeric_cols = ["id"] + [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols]
    features = [c for c in df.columns if c not in ("id","label")]
    return df, features

def main():
    ap = argparse.ArgumentParser(description="Build training CSV from ./features + ./data labels.")
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--select_features", nargs="*")
    ap.add_argument("--out", default="data/training_dataset.csv")
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    assert fdir.exists(), f"Missing features dir: {fdir}"

    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    if labels_csv is None:
        for cand in [Path("data/labels.csv"), Path("data/tricks.csv")]:
            if cand.exists():
                labels_csv = cand; break
    assert labels_csv and labels_csv.exists(), "Missing labels CSV (try --labels_csv or data/labels.csv)."

    X, feats = collect_features(fdir, args.select_features)
    id2lab = load_label_map(labels_csv)

    labels = []
    for vid in X["id"].tolist():
        if vid in id2lab:
            labels.append(id2lab[vid])
        else:
            m = re.search(r"_([A-Za-z0-9]+)$", vid)
            labels.append(m.group(1) if m else "UNKNOWN")
    X.insert(1, "label", labels)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.out, index=False)
    Path(str(Path(args.out).with_suffix(".features.txt"))).write_text("\n".join([c for c in X.columns if c not in ("id","label")]))
    print(f"[OK] Wrote {args.out} shape={X.shape}")
    print(f"[OK] Wrote {Path(args.out).with_suffix('.features.txt')}")

if __name__ == "__main__":
    main()
