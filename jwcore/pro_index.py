from __future__ import annotations
import json, os, random
from typing import Dict, List, Optional, Tuple
import subprocess, sys, json as _json

from jwcore.pose_utils import FEATURE_KEYS  # used for comparing trick feature vectors

# ----------------------------
# PURPOSE
# ----------------------------
# This module manages your “pro examples” — the ideal trick reference videos.
# Each trick label (e.g., 'backflip') can have one or more associated “pro” clips.
# When a student’s video is classified, this file helps you pick a matching pro clip
# for comparison or feedback.

# The main function you’ll call from the pipeline is:
#   pick_pro_for_label(label="backflip", student_features_path="features/student.json")
# which will return the filepath to the pro video best suited for that trick.

# ----------------------------
# LOAD THE PRO INDEX
# ----------------------------

def load_pro_index(path: str = "data/pro_index.json") -> Dict[str, List[str]]:
    """
    Reads your pro video catalog (JSON file) and ensures all listed files exist.

    Example JSON structure (data/pro_index.json):
    {
      "backflip": ["videos/pro/backflip_pro_01.mp4", "videos/pro/backflip_pro_02.mp4"],
      "frontflip": ["videos/pro/frontflip_pro_01.mp4"]
    }

    Returns:
        A dict mapping each trick label -> list of verified video file paths.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pro index: {path}")
    obj = json.load(open(path))

    out: Dict[str, List[str]] = {}
    for label, vals in obj.items():
        # Normalize: ensure each entry is a list and filter out any missing files.
        files = vals if isinstance(vals, list) else [vals]
        files = [p for p in files if os.path.exists(p)]
        if files:
            out[label] = files
    return out


# ----------------------------
# HELPER: BUILD OR FIND FEATURES FOR A VIDEO
# ----------------------------

def _feature_path_for_video(video_path: str, features_dir: str = "features") -> str:
    """
    Given a video path, compute where its feature JSON should live.
    e.g. videos/pro/backflip_pro_01.mp4 → features/backflip_pro_01.json
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(features_dir, f"{stem}.json")


def _ensure_features_for_video(video_path: str) -> str:
    """
    Ensure that we have pose-tracking features for a given video.

    If missing, this function will:
      1. Run extract_keypoints.py  (video → cache/*.npz)
      2. Run jwcore.pose_extract    (npz → features/*.json)

    Returns:
        Path to the feature JSON file.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    npz = os.path.join("cache", f"{stem}.posetrack.npz")
    feat = os.path.join("features", f"{stem}.json")

    if not os.path.exists(npz):
        cmd = [sys.executable, "scripts/extract_keypoints.py", "--video", video_path]
        subprocess.check_call(cmd)

    if not os.path.exists(feat):
        cmd = [sys.executable, "-m", "jwcore.pose_extract", "--posetrack", npz, "--out", feat]
        subprocess.check_call(cmd)

    return feat


# ----------------------------
# HELPER: LOAD SELECTED FEATURES FOR COMPARISON
# ----------------------------
def _load_feature_subset(feat_path: str,
                         keys=("airtime_s", "height_clearance_px_p95", "pitch_total_rad")) -> Tuple[float, float, float]:
    """
    Load a small subset of numeric features from a feature JSON.

    We only use simple metrics for matching — airtime, max height, and angle range —
    to find the closest 'pro' clip to a student's trick.
    """
    obj = _json.load(open(feat_path))
    f = obj["features"]
    return tuple(float(f.get(k, 0.0)) for k in keys)


# ----------------------------
# MAIN FUNCTION
# ----------------------------

def pick_pro_for_label(label: str,
                       student_features_path: Optional[str] = None,
                       index_path: str = "data/pro_index.json",
                       strategy: str = "first",
                       seed: Optional[int] = None) -> Optional[str]:
    """
    Selects the best pro reference for a given trick label.

    Args:
        label: predicted trick label from classifier (e.g., "backflip")
        student_features_path: path to student's feature JSON (used for 'closest' strategy)
        index_path: path to data/pro_index.json (the catalog)
        strategy: how to choose among multiple pro clips for this label:
            - 'first': always return the first available (default)
            - 'random': pick randomly (for variety)
            - 'closest': compute similarity in a few key metrics and pick best match
        seed: optional random seed for reproducibility

    Returns:
        Path to chosen pro video, or None if none found.
    """
    idx = load_pro_index(index_path)
    candidates = idx.get(label)
    if not candidates:
        # No known pro video for this trick label
        return None

    if strategy == "first":
        # Always pick the first one — deterministic and simple.
        return candidates[0]

    if strategy == "random":
        rnd = random.Random(seed)
        return rnd.choice(candidates)

    if strategy == "closest":
        # Compare basic numeric features to pick most similar pro.
        if not student_features_path or not os.path.exists(student_features_path):
            raise ValueError("closest strategy requires student_features_path (features/*.json).")

        s_air, s_h, s_ang = _load_feature_subset(student_features_path)
        best = None
        best_d = float("inf")

        for vid in candidates:
            # For each pro video, load its features (build if missing)
            pro_feat = _feature_path_for_video(vid)
            if not os.path.exists(pro_feat):
                pro_feat = _ensure_features_for_video(vid)

            # Compute simple distance metric between student and pro
            p_air, p_h, p_ang = _load_feature_subset(pro_feat)
            d = (s_air - p_air) ** 2 + (s_h - p_h) ** 2 + 0.002 * (s_ang - p_ang) ** 2

            if d < best_d:
                best_d, best = d, vid

        return best

    raise ValueError(f"Unknown strategy: {strategy}")
