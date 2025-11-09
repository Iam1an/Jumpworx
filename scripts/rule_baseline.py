#!/usr/bin/env python3
"""
Rule baseline on feature JSONs using Z cues.

Usage:
  python scripts/rule_baseline.py --labels_csv data/labels.csv --features_dir features/ \
      [--per_clip_csv viz/rule_scores.csv] [--threshold 0.0]
"""

from __future__ import annotations
import argparse, csv, json, os
from typing import Dict, List, Tuple

USED = ["pitch_total", "knees_z_delta", "shoulderhip_z_slope"]

def norm_id(s: str) -> str:
    """Normalize clip IDs: strip video/ext suffixes and keep basename root."""
    s = s.replace(".posetrack", "")
    for ext in (".mov", ".mp4", ".avi", ".mkv"):
        s = s.replace(ext, "")
    base = os.path.basename(s)
    root, _ = os.path.splitext(base)
    return root

def load_features(features_dir: str, clip_id: str) -> Dict[str, float] | None:
    """
    Load features for a normalized clip_id.
    Supports both flat JSON (preferred) and legacy nested {"features": {...}}.
    """
    cid = norm_id(clip_id)
    path = os.path.join(features_dir, f"{cid}.json")
    if not os.path.exists(path):
        return None
    data = json.load(open(path, "r"))
    feats = data.get("features", data)  # legacy nested OR flat
    # ensure numeric
    out = {}
    for k in USED + ["airtime_s"]:
        v = feats.get(k)
        if v is None:
            continue
        try:
            out[k] = float(v)
        except Exception:
            pass
    # require at least the USED keys
    if not all(k in out for k in USED):
        return None
    return out

def score_rule(f: Dict[str, float], w: Tuple[float, float, float] = (1.2, 0.8, 0.5)) -> float:
    pt = float(f.get("pitch_total", 0.0))
    kd = float(f.get("knees_z_delta", 0.0))
    sh = float(f.get("shoulderhip_z_slope", 0.0))
    return w[0]*pt + w[1]*kd + w[2]*sh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--threshold", type=float, default=0.0, help="Decision threshold on the rule score")
    ap.add_argument("--per_clip_csv", help="Optional path to write per-clip diagnostics CSV")
    args = ap.parse_args()

    # Load labels (expects DictReader with columns at least: clip_id,label)
    with open(args.labels_csv, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    total = 0
    correct = 0
    skipped_no_feats = 0
    skipped_bad_label = 0
    out_rows: List[Dict[str, str]] = []

    for r in rows:
        raw_cid = (r.get("clip_id") or "").strip()
        true = (r.get("label") or "").strip().lower()
        if not raw_cid or true not in ("frontflip", "backflip"):
            skipped_bad_label += 1
            continue

        feats = load_features(args.features_dir, raw_cid)
        if feats is None:
            skipped_no_feats += 1
            continue

        s = score_rule(feats)
        pred = "frontflip" if s > args.threshold else "backflip"
        total += 1
        correct += int(pred == true)

        out_rows.append({
            "clip_id": norm_id(raw_cid),
            "true": true,
            "pred": pred,
            "score": f"{s:.6f}",
            "pitch_total": f"{feats.get('pitch_total', 0.0):.6f}",
            "knees_z_delta": f"{feats.get('knees_z_delta', 0.0):.6f}",
            "shoulderhip_z_slope": f"{feats.get('shoulderhip_z_slope', 0.0):.6f}",
            "airtime_s": f"{feats.get('airtime_s', 0.0):.6f}",
        })

    acc = correct / max(total, 1)
    print(f"Rule baseline (JSON features): {correct}/{total} = {acc:.3f}  (thr={args.threshold:+.3f})")
    print(f"Skipped: no_features={skipped_no_feats}, bad_label_or_missing={skipped_bad_label}")

    if args.per_clip_csv and out_rows:
        os.makedirs(os.path.dirname(args.per_clip_csv) or ".", exist_ok=True)
        with open(args.per_clip_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote per-clip diagnostics → {args.per_clip_csv}")

if __name__ == "__main__":
    main()
