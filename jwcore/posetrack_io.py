# jwcore/posetrack_io.py
# Unified loader for Jumpworx pose-track NPZ files.
#
# Goals:
# - Hide historical layout differences:
#     * New:  P (T,J,3), V (T,J), meta_json (JSON string/dict-like)
#     * Old:  kps_xyz (T,J,3), visibility (T,J), fps[...] or fps scalar
# - Always return:
#     P:    (T, J, 3) float32, NaN-safe
#     V:    (T, J)   float32 in [0,1] where possible
#     fps:  float (default 30.0 if missing)
#     meta: dict with at least:
#              - "fps" (float)
#              - "landmark_names" (if available)
#              - "image_size" (if available)
#
# Everything downstream (metrics, runner, compare tools) should use this
# instead of poking NPZ internals directly.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np


def _parse_meta_json_like(raw: Any) -> Dict[str, Any]:
    """
    Best-effort parse of meta_json-like content:
      - if dict-like, return as dict
      - if bytes/str, try json.loads, else {}
      - else {}
    """
    if raw is None:
        return {}

    # Already a dict-like object
    if isinstance(raw, dict):
        return dict(raw)

    # Numpy scalar / 0-d array often holds JSON string
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()

    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    return {}


def _coerce_fps(meta: Dict[str, Any], npz: np.lib.npyio.NpzFile) -> float:
    """
    Extract fps from:
      1) meta["fps"] if valid
      2) npz["fps"] as scalar or first element
      3) fallback 30.0
    """
    # meta_json wins
    try:
        if "fps" in meta:
            v = float(meta["fps"])
            if np.isfinite(v) and v > 1e-3:
                return v
    except Exception:
        pass

    # Legacy 'fps' array / scalar
    try:
        if "fps" in npz.files:
            arr = npz["fps"]
            if np.isscalar(arr):
                v = float(arr)
            else:
                v = float(arr.ravel()[0])
            if np.isfinite(v) and v > 1e-3:
                return v
    except Exception:
        pass

    # Default
    return 30.0


def load_posetrack(path: str) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Load a pose-track NPZ and return (P, V, fps, meta).

    P:   (T, J, 3) float32
    V:   (T, J) float32 in [0,1] where possible
    fps: float
    meta: dict; includes parsed meta_json if present, plus ensured:
          - meta["fps"]
          - meta["landmark_names"] if present
          - meta["image_size"] if present
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PoseTrack NPZ not found: {path}")

    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)

        meta: Dict[str, Any] = {}
        if "meta_json" in files:
            meta = _parse_meta_json_like(d["meta_json"])

        # -------- Layout 1: current Jumpworx: P / V / meta_json --------
        if "P" in files:
            P = d["P"].astype(np.float32)
            if "V" in files:
                V = d["V"].astype(np.float32)
            else:
                V = np.ones(P.shape[:2], dtype=np.float32)

        # -------- Layout 2: legacy: kps_xyz / visibility / fps --------
        elif "kps_xyz" in files:
            P = d["kps_xyz"].astype(np.float32)
            if "visibility" in files:
                V = d["visibility"].astype(np.float32)
            else:
                V = np.ones(P.shape[:2], dtype=np.float32)

        else:
            raise KeyError(
                f"Unrecognized pose layout in '{path}'. "
                f"Expected one of: P / (kps_xyz), found {sorted(files)}"
            )

        # Sanity: ensure shapes consistent
        if P.ndim != 3 or P.shape[2] < 2:
            raise ValueError(f"Invalid P shape in '{path}': {P.shape}")
        T, J, _ = P.shape

        if V.shape[0] != T or V.shape[1] != J:
            # Fallback; logically this should not happen
            V = np.ones((T, J), dtype=np.float32)

        # Clean obviously bad vis
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        V = np.clip(V, 0.0, 1.0)

        fps = _coerce_fps(meta, d)

        # Enrich meta a bit, but non-invasively
        meta = dict(meta)  # ensure plain dict
        meta.setdefault("fps", float(fps))

        # Pass through landmark_names / image_size if present at top-level
        if "landmark_names" in files and "landmark_names" not in meta:
            try:
                ln = d["landmark_names"]
                if isinstance(ln, np.ndarray):
                    meta["landmark_names"] = [str(x) for x in ln.tolist()]
            except Exception:
                pass

        if "image_size" in files and "image_size" not in meta:
            try:
                sz = d["image_size"]
                # Expect (H, W) or (H, W, C); store as list
                meta["image_size"] = [int(x) for x in np.array(sz).ravel().tolist()]
            except Exception:
                pass

        # Helpful for debugging
        meta.setdefault("source_path", os.path.abspath(path))

    return P, V, float(fps), meta


def load_posetrack_npz(path: str) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Alias for load_posetrack for compatibility with existing imports.
    """
    return load_posetrack(path)


def resolve_posetrack_path(arg: str, cache_dir: str = "cache") -> str:
    """
    Resolve a user-supplied identifier into an actual .posetrack.npz path.

    Rules:
      - If arg is an existing file path → return as-is.
      - If arg has no extension → look for {cache_dir}/{arg}.posetrack.npz
      - If arg already endswith ".posetrack.npz" → try under cache_dir as fallback.
    """
    # Direct file path
    if os.path.isfile(arg):
        return arg

    # As stem under cache_dir
    base = os.path.join(cache_dir, f"{arg}.posetrack.npz")
    if os.path.isfile(base):
        return base

    # If given like "TRICK12_BACKFLIP.posetrack.npz" without dir
    if arg.endswith(".posetrack.npz"):
        cand = os.path.join(cache_dir, arg)
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        f"Could not resolve pose-track NPZ for '{arg}'. "
        f"Tried as file and as stem in '{cache_dir}'."
    )


__all__ = [
    "load_posetrack",
    "load_posetrack_npz",
    "resolve_posetrack_path",
]
