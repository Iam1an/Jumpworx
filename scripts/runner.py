#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import subprocess
import shlex
import glob
import numpy as np

# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jwcore.posetrack_io import load_posetrack
from jwcore.compare_metrics import compare_metrics_from_xyz
from jwcore.pro_utils import get_or_build_pro_features
from jwcore.coach import coach
from jwcore.pose_extract import process_file as process_npz
from jwcore.pose_utils import FEATURE_KEYS
from jwcore.feature_deltas import stance_hand_deltas
from jwcore.coaching_thresholds import TAU as COACH_TAU
from jwcore.pro_index import pick_pro_for_label
from scripts.generate_coaching import _make_llm_callable


# -----------------------------------------------------------------------------
# Event emitter
# -----------------------------------------------------------------------------
def emit(event_type: str, **kwargs):
    rec = {"type": event_type, "ts": time.time()}
    rec.update(kwargs)
    print("JW_EVENT:" + json.dumps(rec))
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str):
    """
    Ensure directory exists.
    If given a file path, creates its parent dir.
    If given a directory path, creates it directly.
    """
    if not path:
        return
    # Heuristic: if has an extension, treat as file path
    d = path
    if os.path.splitext(path)[1]:
        d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _npz_for_video(video_path: str) -> str:
    return os.path.join("cache", f"{stem(video_path)}.posetrack.npz")


def _ensure_npz(video_path: str) -> str:
    npz = _npz_for_video(video_path)
    if not os.path.exists(npz):
        print(f"[extract_keypoints] -> {npz}")
        ensure_dir(npz)
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "extract_keypoints.py"),
                "--video",
                video_path,
                "--out",
                npz,
            ]
        )
    return npz


def _normalize_tips(items):
    out = []
    for t in (items or []):
        if not t:
            continue
        s = str(t).strip()
        if not s:
            continue
        if s.startswith("- "):
            s = s[2:]
        out.append(s)
    return out


# -----------------------------------------------------------------------------
# Pro selection via compare_metrics_from_xyz + pro_features
# -----------------------------------------------------------------------------
def _select_best_pro(
    amateur_npz: str,
    pro_npz_list: list[str],
    predicted_label: str | None,
):
    """
    Metric-based pro selection:
      - Optional coarse filter by predicted_label (frontflip/backflip substring).
      - For each candidate:
          * compare_metrics_from_xyz(am, pro)
          * get_or_build_pro_features(pro_npz)
          * style distance over stable metrics
      - Return (best_pro_npz, best_metrics_dict, best_pro_features_dict)
    """
    if not pro_npz_list:
        return None, None, None

    P_am, _, _, _ = load_posetrack(amateur_npz)
    label_lower = (predicted_label or "").lower()

    best_npz = None
    best_metrics = None
    best_pro_feats = None
    best_dist = float("inf")
    debug_rows: list[dict] = []

    for pro_npz in pro_npz_list:
        name = os.path.basename(pro_npz).lower()

        if label_lower in ("frontflip", "backflip") and label_lower not in name:
            continue

        try:
            P_pr, _, _, _ = load_posetrack(pro_npz)
        except Exception:
            continue

        try:
            metrics = compare_metrics_from_xyz(P_am, P_pr)
        except Exception:
            continue

        try:
            pro_feats = get_or_build_pro_features(pro_npz)
        except Exception:
            continue

        dist = 0.0
        count = 0
        for k in (
            "midair_hand_span_pct_of_torso",
            "midair_hand_asym_pct_of_torso",
            "landing_stance_width_pct_of_hip",
        ):
            am_v = metrics["scalars"].get(k)
            pr_v = pro_feats.get(k)
            if am_v is None or pr_v is None:
                continue
            try:
                dv = float(am_v) - float(pr_v)
            except Exception:
                continue
            if math.isfinite(dv):
                dist += dv * dv
                count += 1

        if count == 0:
            continue

        debug_rows.append({"pro_npz": pro_npz, "dist": dist, "count": count})

        if dist < best_dist:
            best_dist = dist
            best_npz = pro_npz
            best_metrics = metrics
            best_pro_feats = pro_feats

    if debug_rows and best_npz is not None:
        try:
            rows_sorted = sorted(debug_rows, key=lambda r: r["dist"])
            emit(
                "pro_select_debug",
                label=predicted_label,
                candidates=[
                    {
                        "pro_npz": os.path.basename(r["pro_npz"]),
                        "dist": round(float(r["dist"]), 4),
                        "count": int(r["count"]),
                    }
                    for r in rows_sorted
                ],
                chosen=os.path.basename(best_npz),
            )
        except Exception:
            pass

    return best_npz, best_metrics, best_pro_feats


# -----------------------------------------------------------------------------
# Feature extraction from NPZ
# -----------------------------------------------------------------------------
def _extract_features_from_npz(npz_path: str, verbose: bool = False) -> dict:
    out_dir = os.path.join("cache", "_pose_extract")
    os.makedirs(out_dir, exist_ok=True)

    maybe_path = process_npz(npz_path, out_dir, verbose=verbose)

    if (
        isinstance(maybe_path, str)
        and maybe_path.endswith(".json")
        and os.path.exists(maybe_path)
    ):
        jpath = maybe_path
    else:
        jpath = os.path.join(out_dir, f"{stem(npz_path)}.json")

    with open(jpath, "r") as f:
        obj = json.load(f)

    return obj["features"] if isinstance(obj, dict) and "features" in obj else obj


def _parse_meta_json_like(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return {}


def _load_P_and_fps(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as d:
        if "P" in d.files:
            P = d["P"]
        elif "kps_xyz" in d.files:
            P = d["kps_xyz"]
        else:
            raise KeyError(f"{npz_path}: no P/kps_xyz array")
        meta = {}
        if "meta_json" in d.files:
            meta = _parse_meta_json_like(d["meta_json"])
        try:
            fps = float(meta.get("fps", 30.0))
        except Exception:
            fps = 30.0
    return P, fps


# -----------------------------------------------------------------------------
# Legacy geometry helpers (kept for compatibility)
# -----------------------------------------------------------------------------
L_WR = 9
R_WR = 10
L_SH = 5
R_SH = 6
L_HIP = 11
R_HIP = 12
L_ANK = 15
R_ANK = 16
L_FOOTIDX = 19
R_FOOTIDX = 20


def _midline_x(P):
    return 0.5 * (P[:, L_SH, 0] + P[:, R_SH, 0])


def _torso_len(P):
    v = P[:, L_SH, :] - P[:, L_HIP, :]
    return np.sqrt(np.sum(v**2, axis=1))


def _hip_width(P):
    v = P[:, R_HIP, :] - P[:, L_HIP, :]
    return np.sqrt(np.sum(v**2, axis=1))


def _hand_span(P):
    return np.sqrt(np.sum((P[:, R_WR, :] - P[:, L_WR, :]) ** 2, axis=1))


def _hand_asym_about_midline(P):
    mid = _midline_x(P)
    return (P[:, R_WR, 0] - mid) - (mid - P[:, L_WR, 0])


def _stance_width(P):
    return np.sqrt(np.sum((P[:, R_FOOTIDX, :] - P[:, L_FOOTIDX, :]) ** 2, axis=1))


def _mean_in_window_adaptive(x, i, halfw):
    if i is None or not np.isfinite(i):
        return float("nan")
    i = int(i)
    lo, hi = max(0, i - halfw), min(len(x), i + halfw + 1)
    seg = x[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmean(seg)) if seg.size else float("nan")


def _safe_delta(a, b):
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


# -----------------------------------------------------------------------------
# Legacy series-based pro picker (very last-resort)
# -----------------------------------------------------------------------------
def _series_from_npz(npz_path: str, feature: str):
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            if "P" in d.files:
                P = d["P"]
            else:
                P = d["kps_xyz"]
    except Exception:
        return None

    if P.ndim != 3 or P.shape[1] <= max(R_ANK, R_HIP, R_SH):
        return None

    if feature == "hip_y":
        s = 0.5 * (P[:, L_HIP, 1] + P[:, R_HIP, 1])
    elif feature == "pitch":
        torso = 0.5 * (P[:, L_SH, :] + P[:, R_SH, :]) - 0.5 * (
            P[:, L_HIP, :] + P[:, R_HIP, :]
        )
        horiz = np.sqrt(torso[:, 0] ** 2 + torso[:, 2] ** 2)
        s = np.arctan2(horiz, np.abs(torso[:, 1]) + 1e-6)
    else:  # ankle_y
        s = 0.5 * (P[:, L_ANK, 1] + P[:, R_ANK, 1])

    s = s.astype(np.float32)
    m = np.isfinite(s)
    if m.sum() < 5:
        return None
    s = s[m]
    s -= np.nanmin(s)
    if s.size > 300:
        k = max(1, s.size // 300)
        s = s[::k]
    return s


def _series_dist(a, b):
    if a is None or b is None:
        return float("inf")
    n = min(len(a), len(b))
    if n <= 0:
        return float("inf")
    a = a[:n].astype(np.float32)
    b = b[:n].astype(np.float32)
    return float(np.nanmean((a - b) ** 2))


def pick_pro_video(amateur_video: str, pro_dir: str, feature: str = "ankle_y") -> str | None:
    amat_npz = _ensure_npz(amateur_video)
    amat_s = _series_from_npz(amat_npz, feature)
    best = (float("inf"), None)
    for fn in sorted(os.listdir(pro_dir)):
        if not fn.lower().endswith((".mp4", ".mov", ".m4v", ".avi")):
            continue
        v = os.path.join(pro_dir, fn)
        try:
            npz = _ensure_npz(v)
        except Exception:
            continue
        pro_s = _series_from_npz(npz, feature)
        d = _series_dist(amat_s, pro_s)
        if d < best[0]:
            best = (d, v)
    return best[1]


# -----------------------------------------------------------------------------
# Classification helper
# -----------------------------------------------------------------------------
def _predict_label_and_proba(clf, feats):
    """
    Try multiple classifier APIs; return (label or None, proba or None).
    Not used directly now, but kept for completeness.
    """
    if hasattr(clf, "predict_with_proba"):
        try:
            y, p = clf.predict_with_proba(feats)
            label = str(y)
            proba = None
            if isinstance(p, (float, int)):
                proba = float(p)
            elif isinstance(p, dict) and label in p:
                proba = float(p[label])
            return label, proba
        except Exception:
            pass

    if hasattr(clf, "predict"):
        try:
            y = clf.predict([feats])
            if isinstance(y, (list, tuple, np.ndarray)):
                y = y[0]
            return str(y), None
        except Exception:
            return None, None

    return None, None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run pipeline on one amateur video: classify → pick pro → coach → visualize."
    )
    ap.add_argument("--amateur_video", required=True)
    ap.add_argument("--pro_videos_dir", default="videos/pro")
    ap.add_argument(
        "--model_base",
        default="models/jumpworx_model",
        help="Classifier model base path (no extension: tries .joblib/.pkl or dir).",
    )
    ap.add_argument(
        "--strategy",
        choices=["closest", "first", "random"],
        default="closest",
        help="How to pick pro reference if index lookup not used/available.",
    )
    ap.add_argument("--align", choices=["dtw", "apex", "takeoff", "landing"], default="dtw")
    ap.add_argument(
        "--align_feature",
        choices=["ankle_y", "hip_y", "pitch"],
        default="ankle_y",
    )
    ap.add_argument(
        "--hud_corner",
        choices=["tl", "tr", "bl", "br"],
        default="tr",
        help="Corner for HUD in viz script.",
    )
    ap.add_argument("--seconds_before", type=float, default=1.0)
    ap.add_argument("--seconds_after", type=float, default=1.0)
    ap.add_argument("--output_fps", type=float, default=15.0)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no_phase", action="store_true", help="Disable phase-aware metrics path.")
    ap.add_argument("--no_llm", action="store_true", help="Disable LLM; use rule-based coaching only.")
    ap.add_argument("--llm_provider", default="openai")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--llm_max_tokens", type=int, default=256)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--dump_frame_table",
        action="store_true",
        help="If set, run scripts.build_frame_table for the selected pair.",
    )

    args = ap.parse_args()

    emit("stage", name="start", msg="Starting runner")

    # 1) Amateur NPZ
    amat_npz = _ensure_npz(args.amateur_video)
    emit("ok", msg="Amateur NPZ ready", npz=amat_npz)

    # 2) Amateur features
    amateur_feats = _extract_features_from_npz(amat_npz, verbose=args.verbose)
    emit("ok", msg="Features extracted", keys=sorted(amateur_feats.keys()))

    # 3) Classification
    emit("stage", name="clf", msg="Loading classifier and predicting label")
    label = None
    proba = None
    predicted_label = None
    clf = None

    # 3a) Native TrickClassifier
    try:
        from jwcore.trick_classifier import TrickClassifier as JWTrickClassifier  # type: ignore

        default_model_path = "models/jumpworx_model.joblib"
        model_base = getattr(args, "model_base", None) or default_model_path
        clf = JWTrickClassifier(model_base)

        if hasattr(clf, "predict_with_proba"):
            try:
                res = clf.predict_with_proba(amateur_feats)
                if isinstance(res, tuple) and len(res) == 2:
                    y, p = res
                    label = str(y)
                    if isinstance(p, (float, int)):
                        proba = float(p)
                    elif isinstance(p, dict) and label in p:
                        try:
                            proba = float(p[label])
                        except Exception:
                            proba = None
                elif isinstance(res, dict):
                    y = clf.predict(amateur_feats)
                    label = str(y)
                    if label in res:
                        try:
                            proba = float(res[label])
                        except Exception:
                            proba = None
                if label is None:
                    y = clf.predict(amateur_feats)
                    label = str(y)
            except Exception:
                y = clf.predict(amateur_feats)
                label = str(y)
        else:
            y = clf.predict(amateur_feats)
            label = str(y)

    except Exception as e:
        emit(
            "warn",
            msg="Native TrickClassifier failed; falling back to generic loader",
            error=str(e),
        )
        clf = None

    # 3b) Generic fallback classifier
    if clf is None and label is None:
        import joblib

        class FallbackTrickClassifier:
            def __init__(self, model_base_path: str):
                self.model_base = model_base_path
                self.pipe = None
                self.class_names = None
                self.feature_keys = list(FEATURE_KEYS)
                cands = [
                    model_base_path,
                    model_base_path + ".pkl",
                    model_base_path + ".joblib",
                ]
                bundle = None
                self._path = None
                for c in cands:
                    if os.path.isdir(c):
                        hits = [
                            os.path.join(c, p)
                            for p in os.listdir(c)
                            if p.endswith(".pkl") or p.endswith(".joblib")
                        ]
                        if hits:
                            self._path = hits[0]
                            bundle = joblib.load(self._path)
                            break
                    elif os.path.exists(c):
                        self._path = c
                        bundle = joblib.load(c)
                        break
                if bundle is None:
                    raise FileNotFoundError(
                        f"Model not found for base '{model_base_path}' "
                        f"(tried {', '.join(cands)})"
                    )
                if isinstance(bundle, dict):
                    self.pipe = bundle.get("model") or bundle.get("pipeline")
                    self.feature_keys = bundle.get("feature_keys") or self.feature_keys
                    self.class_names = bundle.get("class_names")
                else:
                    self.pipe = bundle
                if self.pipe is None:
                    raise RuntimeError(
                        "FallbackTrickClassifier: no model pipeline in bundle"
                    )

            def _to_row(self, features: dict):
                vals = []
                for k in self.feature_keys:
                    try:
                        vals.append(float(features.get(k, 0.0)))
                    except Exception:
                        vals.append(0.0)
                return np.array([vals], dtype=np.float32)

            def predict(self, features: dict) -> str:
                x = self._to_row(features)
                y = self.pipe.predict(x)[0]
                return str(y)

            def predict_with_proba(self, features: dict):
                if not hasattr(self.pipe, "predict_proba"):
                    return self.predict(features), None
                x = self._to_row(features)
                proba_vec = self.pipe.predict_proba(x)[0]
                y = self.pipe.predict(x)[0]
                if (
                    self.class_names is not None
                    and len(proba_vec) == len(self.class_names)
                ):
                    try:
                        idx = list(self.class_names).index(y)
                        return str(y), float(proba_vec[idx])
                    except Exception:
                        return str(y), None
                return str(y), None

        try:
            fb = FallbackTrickClassifier(args.model_base)
            y, p = fb.predict_with_proba(amateur_feats)
            label = str(y)
            proba = float(p) if p is not None else None
        except Exception as e:
            emit(
                "warn",
                msg="Generic classifier load failed; proceeding without classification",
                error=str(e),
            )
            label = None
            proba = None

    if label is not None:
        emit("ok", msg="Classification complete", label=label, proba=proba)
        predicted_label = label
    else:
        emit("warn", msg="No label predicted; proceeding without class label")

    # 4) Pro selection
    emit("stage", name="pick", msg="Selecting reference/pro")
    pro_video = None
    pro_npz = None
    pro_feats = None
    pro_features_for_coach = None
    metrics_for_coach = None
    pro_dir = args.pro_videos_dir

    try:
        hits = [
            os.path.join(pro_dir, p)
            for p in sorted(os.listdir(pro_dir))
            if p.lower().endswith((".mp4", ".mov", ".m4v", ".avi"))
        ]

        # Precompute NPZs for all candidate pros
        video_to_npz: dict[str, str] = {}
        pro_npz_list: list[str] = []
        for v in hits:
            try:
                npz = _ensure_npz(v)
                video_to_npz[v] = npz
                pro_npz_list.append(npz)
            except Exception:
                continue

        # 4a) pro_index-based selection (if label + features path)
        amat_feat_path = os.path.join(
            "cache", "_pose_extract", f"{stem(amat_npz)}.json"
        )
        if predicted_label and os.path.exists(amat_feat_path):
            try:
                pro_video = pick_pro_for_label(
                    label=str(predicted_label),
                    student_features_path=amat_feat_path,
                    index_path="data/pro_index.json",
                    strategy=args.strategy,
                )
            except Exception as e:
                emit(
                    "warn",
                    msg="pro_index selection failed; falling back",
                    error=str(e),
                )
                pro_video = None

        # 4b) Metric-based "closest" selection
        if pro_video is None and args.strategy == "closest" and pro_npz_list:
            best_npz, best_metrics, best_pro_feats = _select_best_pro(
                amateur_npz=amat_npz,
                pro_npz_list=pro_npz_list,
                predicted_label=predicted_label,
            )
            if best_npz is not None:
                pro_npz = best_npz
                metrics_for_coach = best_metrics
                pro_features_for_coach = best_pro_feats
                for v, n in video_to_npz.items():
                    if n == best_npz:
                        pro_video = v
                        break

        # 4c) Legacy fallbacks
        if pro_video is None:
            if args.strategy == "first" and hits:
                pro_video = hits[0]
            elif args.strategy == "random" and hits:
                rng = np.random.default_rng(123)
                pro_video = str(rng.choice(hits))
            elif args.strategy == "closest" and hits and pro_npz is None:
                pro_video = pick_pro_video(
                    args.amateur_video, pro_dir, feature=args.align_feature
                )

        # 4d) Ensure NPZ + features for chosen pro
        if pro_video and pro_npz is None:
            pro_npz = video_to_npz.get(pro_video) or _ensure_npz(pro_video)

        if pro_npz:
            pro_feats = _extract_features_from_npz(pro_npz, verbose=args.verbose)
            if pro_features_for_coach is None:
                pro_features_for_coach = get_or_build_pro_features(pro_npz)

    except Exception as e:
        emit(
            "warn",
            msg="Pro selection failed",
            error=f"{type(e).__name__}: {e}",
        )
        pro_video = None
        pro_npz = None
        pro_feats = None
        pro_features_for_coach = None
        metrics_for_coach = None

    emit("ok", msg="Pro selected", pro_video=pro_video, pro_npz=pro_npz)
    print(f"[runner] pro_video = {pro_video}")
    print(f"[runner] pro_npz   = {pro_npz}")

    # 5) Coaching / metrics assembly
    try:
        extras = {}
        if pro_feats is not None:
            extras = stance_hand_deltas(amateur_feats, pro_feats)

        midair_hand_span_pct = float("nan")
        midair_hand_asym_pct = float("nan")
        landing_stance_pct = float("nan")

        use_compare_metrics = (
            metrics_for_coach is not None and isinstance(metrics_for_coach, dict)
        )

        if use_compare_metrics and not args.no_phase and pro_npz is not None:
            scalars = metrics_for_coach.get("scalars", {}) or {}

            midair_hand_span_pct = float(
                scalars.get(
                    "midair_hand_span_pct_of_torso",
                    float("nan"),
                )
            )
            midair_hand_asym_pct = float(
                scalars.get(
                    "midair_hand_asym_pct_of_torso",
                    float("nan"),
                )
            )
            landing_stance_pct = float(
                scalars.get(
                    "landing_stance_width_pct_of_hip",
                    float("nan"),
                )
            )

            merged_feats = dict(
                amateur_feats,
                **extras,
                **scalars,
                series=metrics_for_coach.get("series", {}),
                joint_diffs=metrics_for_coach.get("joint_diffs"),
                torso_lengths=metrics_for_coach.get("torso_lengths"),
                phase_scores=metrics_for_coach.get("phase_scores", {}),
            )

        elif args.no_phase or pro_npz is None or pro_feats is None:
            merged_feats = dict(amateur_feats, **extras)

        else:
            # Legacy phase-aware path (kept for back-compat)
            amat_P, amat_fps = _load_P_and_fps(amat_npz)
            pro_P, pro_fps = _load_P_and_fps(pro_npz)

            air_t0 = float(amateur_feats.get("air_t0", float("nan")))
            air_t1 = float(amateur_feats.get("air_t1", float("nan")))
            mid_t = (
                air_t0 + 0.5 * (air_t1 - air_t0)
                if (np.isfinite(air_t0) and np.isfinite(air_t1))
                else float("nan")
            )

            takeoff_i = int(round(air_t0 * amat_fps)) if np.isfinite(air_t0) else None
            midair_i = int(round(mid_t * amat_fps)) if np.isfinite(mid_t) else None
            landing_i = (
                int(round(air_t1 * amat_fps)) if np.isfinite(air_t1) else None
            )

            if midair_i is None:
                if amat_P.shape[1] > max(L_FOOTIDX, R_FOOTIDX):
                    fy = 0.5 * (
                        amat_P[:, L_FOOTIDX, 1] + amat_P[:, R_FOOTIDX, 1]
                    )
                else:
                    fy = 0.5 * (amat_P[:, L_ANK, 1] + amat_P[:, R_ANK, 1])
                if np.isfinite(fy).sum() > 10:
                    midair_i = int(np.nanargmin(fy))

            if landing_i is None and midair_i is not None:
                landing_i = min(
                    len(amat_P) - 1, midair_i + int(0.5 * amat_fps)
                )

            tl_amat = _torso_len(amat_P)
            tl_pro = _torso_len(pro_P)
            hw_amat = _hip_width(amat_P)
            hw_pro = _hip_width(pro_P)
            hspan_amat = _hand_span(amat_P)
            hasym_amat = _hand_asym_about_midline(amat_P)
            stance_amat = _stance_width(amat_P)

            hspan_norm = hspan_amat / np.maximum(1e-6, tl_amat)
            hasym_norm = hasym_amat / np.maximum(1e-6, tl_amat)
            stance_norm = stance_amat / np.maximum(1e-6, hw_amat)
            hspan_norm_p = _hand_span(pro_P) / np.maximum(1e-6, tl_pro)
            hasym_norm_p = (
                _hand_asym_about_midline(pro_P) / np.maximum(1e-6, tl_pro)
            )
            stance_norm_p = _stance_width(pro_P) / np.maximum(1e-6, hw_pro)

            midair_hand_span_pct = 100.0 * _mean_in_window_adaptive(
                hspan_norm, midair_i, 6
            )
            midair_hand_asym_pct = 100.0 * _mean_in_window_adaptive(
                hasym_norm, midair_i, 6
            )
            landing_stance_pct = 100.0 * _mean_in_window_adaptive(
                stance_norm, landing_i, 4
            )

            pro_midair_hspan_pct = 100.0 * _mean_in_window_adaptive(
                hspan_norm_p, midair_i, 6
            )
            pro_midair_hasym_pct = 100.0 * _mean_in_window_adaptive(
                hasym_norm_p, midair_i, 6
            )
            pro_landing_stance_pct = 100.0 * _mean_in_window_adaptive(
                stance_norm_p, landing_i, 4
            )

            midair_hand_span_pct_delta = _safe_delta(
                midair_hand_span_pct, pro_midair_hspan_pct
            )
            midair_hand_asym_pct_delta = _safe_delta(
                midair_hand_asym_pct, pro_midair_hasym_pct
            )
            landing_stance_pct_delta = _safe_delta(
                landing_stance_pct, pro_landing_stance_pct
            )

            T = min(amat_P.shape[0], pro_P.shape[0])
            joint_diffs = (amat_P[:T] - pro_P[:T]).astype(np.float32)

            merged_feats = dict(
                amateur_feats,
                **extras,
                midair_hand_span_pct_of_torso=float(midair_hand_span_pct),
                midair_hand_asym_pct_of_torso=float(midair_hand_asym_pct),
                landing_stance_width_pct_of_hip=float(landing_stance_pct),
                delta_midair_hand_span_pct_of_torso=float(
                    midair_hand_span_pct_delta
                ),
                delta_midair_hand_asym_pct_of_torso=float(
                    midair_hand_asym_pct_delta
                ),
                delta_landing_stance_width_pct_of_hip=float(
                    landing_stance_pct_delta
                ),
                joint_diffs=joint_diffs,
            )

        emit(
            "info",
            msg="Phase metric health",
            midair_hspan_ok=bool(np.isfinite(midair_hand_span_pct)),
            midair_hasym_ok=bool(np.isfinite(midair_hand_asym_pct)),
            landing_stance_ok=bool(np.isfinite(landing_stance_pct)),
        )

        emit("stage", name="llm", msg="Calling coaching generator")

        # LLM setup
        if args.no_llm:
            llm_callable = None
            emit(
                "info",
                msg="LLM disabled via --no_llm; using rule-based coaching only",
            )
        else:
            try:
                llm_callable = _make_llm_callable(
                    provider=args.llm_provider,
                    model=args.llm_model,
                    max_tokens=args.llm_max_tokens,
                )
                emit(
                    "info",
                    msg="LLM initialized for coaching",
                    provider=args.llm_provider,
                    model=args.llm_model,
                    max_tokens=args.llm_max_tokens,
                )
            except RuntimeError as e:
                emit(
                    "warn",
                    msg="LLM init failed; using rule-based coaching only",
                    error=str(e),
                )
                llm_callable = None

        # Pro features for coach
        pro_features_arg = None
        if pro_features_for_coach is not None:
            pro_features_arg = pro_features_for_coach
        elif pro_feats is not None:
            pro_features_arg = pro_feats

        coach_out = coach(
            features=merged_feats,
            predicted_label=predicted_label,
            pro_name=stem(pro_video) if pro_video else None,
            pro_features=pro_features_arg,
            llm=llm_callable,
        )

    except Exception as e:
        emit(
            "warn",
            msg="Coaching phase failed; falling back minimal coaching",
            error=f"{type(e).__name__}: {e}",
        )
        merged_feats = dict(
            amateur_feats,
            **(extras if "extras" in locals() else {}),
        )
        coach_out = coach(
            features=merged_feats,
            predicted_label=predicted_label,
            pro_name=stem(pro_video) if pro_video else None,
            pro_features=pro_features_for_coach
            if "pro_features_for_coach" in locals()
            else None,
            llm=None,
        )

    # Top metrics + tips
    try:
        top_metrics = coach_out.get("top_metrics") or coach_out.get(
            "meta", {}
        ).get("top_metrics", [])
    except Exception:
        top_metrics = []
    if top_metrics:
        emit("top_metrics", items=top_metrics, tau=COACH_TAU)

    tips_list = _normalize_tips(coach_out.get("tips", []))
    emit("tips", items=tips_list)

    # 6) Visualization
    emit("stage", name="viz", msg="Rendering side-by-side visualization")
    ensure_dir("viz")

    if pro_npz and pro_video:
        # HUD side expected by viz script: left/right from tl/tr/bl/br
        hud_side = "left" if args.hud_corner in ("tl", "bl") else "right"

        fps_tag = int(args.output_fps) if args.output_fps else 0
        base = f"compare_{stem(args.amateur_video)}_vs_{stem(pro_video)}"
        if fps_tag > 0:
            base += f"_fps{fps_tag}"

        out_mp4 = os.path.join("viz", base + ".mp4")
        out_json2 = os.path.join("viz", base + ".json")
        out_csv = os.path.join("viz", base + ".csv")

        viz_script = os.path.join(SCRIPT_DIR, "viz_compare_side_by_side.py")

        cmd = [
            sys.executable,
            viz_script,
            "--video_a",
            args.amateur_video,
            "--video_b",
            pro_video,
            "--npz_a",
            amat_npz,
            "--npz_b",
            pro_npz,
            "--label_a",
            "Amateur",
            "--label_b",
            "Pro (auto)",
            "--align",
            args.align,
            "--align_feature",
            args.align_feature,
            "--seconds_before",
            str(args.seconds_before),
            "--seconds_after",
            str(args.seconds_after),
            "--height",
            str(args.height),
            "--hud_corner",
            hud_side,
            "--out",
            out_mp4,
            "--metrics_out",
            out_json2,
            "--metrics_csv",
            out_csv,
            "--metrics_panel",  
        ]

        if args.output_fps:
            cmd.extend(["--output_fps", str(args.output_fps)])

        emit(
            "info",
            msg="viz command",
            cmd=" ".join(shlex.quote(c) for c in cmd),
        )

        try:
            subprocess.check_call(cmd)
            emit(
                "done",
                msg="Visualization complete",
                out_mp4=out_mp4,
                out_json=out_json2,
                out_csv=out_csv,
            )
        except Exception as e:
            emit(
                "warn",
                msg="Visualization failed",
                error=f"{type(e).__name__}: {e}",
            )
    else:
        emit(
            "warn",
            msg="Skipping visualization: missing pro reference",
            have_pro_npz=bool(pro_npz),
            have_pro_video=bool(pro_video),
        )

    # 7) Optional frame table
    if args.dump_frame_table and pro_npz and pro_video:
        ft_stem = f"{stem(args.amateur_video)}_vs_{stem(pro_video)}"
        ft_csv = os.path.join("viz", f"frame_table_{ft_stem}.csv")

        cmd = [
            sys.executable,
            "-m",
            "scripts.build_frame_table",
            amat_npz,
            pro_npz,
            "--align_feature",
            args.align_feature,
            "--out_csv",
            ft_csv,
        ]

        emit(
            "info",
            msg="Running build_frame_table",
            cmd=" ".join(shlex.quote(c) for c in cmd),
        )
        try:
            subprocess.run(cmd, check=True)
            emit("done", msg="Frame table written", out_csv=ft_csv)
        except subprocess.CalledProcessError as e:
            emit(
                "warn",
                msg="build_frame_table failed",
                error=str(e),
            )


if __name__ == "__main__":
    main()
