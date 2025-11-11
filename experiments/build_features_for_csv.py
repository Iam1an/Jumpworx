#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, sys, subprocess
from dataclasses import dataclass

@dataclass
class Row:
    clip_id: str
    video_path: str

def run(cmd):
    print(">>"," ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--model_complexity", type=int, choices=[0,1,2], default=1)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    with open(args.labels_csv, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            cid = r["clip_id"].strip()
            vp = (r.get("video_path") or "").strip()
            if not vp or not os.path.exists(vp):
                print(f"!! skip {cid}: missing/invalid video_path"); continue
            npz = os.path.join(args.cache_dir, f"{cid}.posetrack.npz")
            feat = os.path.join(args.features_dir, f"{cid}.json")

            if not os.path.exists(npz):
                run([sys.executable, "scripts/extract_keypoints.py",
                     "--video", vp,
                     "--every", str(args.every),
                     "--model_complexity", str(args.model_complexity),
                     "--out", npz])

            if not os.path.exists(feat):
                run([sys.executable, "-m", "jwcore.pose_extract",
                     "--posetrack", npz,
                     "--out", feat])

    print("Done.")

if __name__ == "__main__":
    main()
