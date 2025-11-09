# jwcore/phase_detect.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from jwcore.joints import (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
)


@dataclass
class PhaseResult:
    labels: List[str]              # len T, per-frame phase label
    takeoff_idx: Optional[int]     # last contact frame before air
    landing_idx: Optional[int]     # first clear contact after air
    air_start: Optional[int]       # first in-air frame
    air_end: Optional[int]         # last in-air frame
    apex_idx: Optional[int]        # frame of max height (min ankle y)
    quality: Dict[str, object]     # status, airtime, reasons, etc.


# ===== helpers =====

def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if not np.isnan(y).any():
        return y
    idx = np.arange(len(y))
    m = np.isfinite(y)
    if not m.any():
        return np.zeros_like(y)
    y[~m] = np.interp(idx[~m], idx[m], y[m])
    return y


def _series_mean_ankle_y(P: np.ndarray) -> np.ndarray:
    """
    Mean of left/right ankle y over time.
    P: (T, J, 3)
    """
    T, J, _ = P.shape
    if max(LEFT_ANKLE, RIGHT_ANKLE) >= J:
        return np.full(T, np.nan, float)

    pair = P[:, [LEFT_ANKLE, RIGHT_ANKLE], 1]  # y coords
    with np.errstate(invalid="ignore"):
        finite_any = np.isfinite(pair).any(axis=1)
        out = np.full(T, np.nan, float)
        out[finite_any] = np.nanmean(pair[finite_any], axis=1)
    return _interp_nan_1d(out)


def _estimate_air_segment_from_ankle(
    ankle_y: np.ndarray,
    min_air_frames: int,
) -> Tuple[Optional[int], Optional[int], Dict[str, object]]:
    """
    Heuristic: find best contiguous 'air' segment using ankle height.
    Coordinate convention assumed: smaller y = higher in image.

    Returns (air_start, air_end, info).
    """
    info: Dict[str, object] = {}
    y = np.asarray(ankle_y, float)
    if y.size < 5 or not np.isfinite(y).any():
        info["reason"] = "no_valid_ankle_y"
        return None, None, info

    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    rng = y_max - y_min
    info["y_max"] = y_max
    info["y_min"] = y_min
    info["y_range"] = rng

    # Too flat -> no real jump
    if rng < 3.0:  # pixels; tune if needed
        info["reason"] = "too_flat_for_air"
        return None, None, info

    # Define air as being significantly above the bed
    thr = y_max - 0.20 * rng  # frames well above bottom
    air_mask = y < thr

    # Find contiguous air segments
    best_start = best_end = None
    best_len = 0
    cur_start = None

    for i, is_air in enumerate(air_mask):
        if is_air:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_end = i - 1
                length = cur_end - cur_start + 1
                if length > best_len:
                    best_start, best_end, best_len = cur_start, cur_end, length
                cur_start = None

    # tail segment
    if cur_start is not None:
        cur_end = len(air_mask) - 1
        length = cur_end - cur_start + 1
        if length > best_len:
            best_start, best_end, best_len = cur_start, cur_end, length

    if best_start is None or best_end is None:
        info["reason"] = "no_contiguous_air_segment"
        return None, None, info

    if best_len < min_air_frames:
        info["reason"] = "air_segment_too_short"
        info["air_len"] = best_len
        return None, None, info

    info["air_len"] = best_len
    return int(best_start), int(best_end), info


def _apex_from_ankle_y(ankle_y: np.ndarray) -> Optional[int]:
    if not np.isfinite(ankle_y).any():
        return None
    return int(np.nanargmin(ankle_y))


def _fill_label_gaps(labels: List[str]) -> List[str]:
    """
    Fill short 'unknown' gaps by propagating nearest known phase.
    Also forward-fills over isolated NaN/weak frames.

    Strategy:
      1) forward-fill
      2) backward-fill remaining
    """
    T = len(labels)
    out = list(labels)

    # forward-fill
    last = None
    for i in range(T):
        if out[i] not in (None, "", "unknown"):
            last = out[i]
        elif last is not None:
            out[i] = last

    # backward-fill
    next_label = None
    for i in range(T - 1, -1, -1):
        if out[i] not in (None, "", "unknown"):
            next_label = out[i]
        elif next_label is not None:
            out[i] = next_label

    # If still 'unknown' everywhere, keep as-is (caller interprets fail).
    return out


# ===== main API =====

def detect_phases_from_pose(
    P: np.ndarray,
    fps: float,
    *,
    min_air_time: float = 0.12,
) -> PhaseResult:
    """
    Phase segmentation for a single trampoline-style jump.

    Inputs:
      P: (T, J, 3) pose (MediaPipe layout)
      fps: frames per second
      min_air_time: minimum airtime (s) to consider as real jump

    Returns:
      PhaseResult with per-frame labels chosen from:
        - "approach"
        - "set"
        - "early_air"
        - "midair"
        - "late_air"
        - "landing"
        - "unknown" (fallback / no good detection)
    """
    T = P.shape[0]
    labels = ["unknown"] * T
    quality: Dict[str, object] = {}

    if T < 8 or not np.isfinite(P).any():
        quality["status"] = "fail"
        quality["reason"] = "too_short_or_all_nan"
        return PhaseResult(labels, None, None, None, None, None, quality)

    # --- ankle signal & apex ---
    ankle_y = _series_mean_ankle_y(P)
    apex_idx = _apex_from_ankle_y(ankle_y)
    quality["apex_idx"] = apex_idx

    # --- air segment detection ---
    min_air_frames = max(3, int(min_air_time * max(fps, 1.0)))
    air_start, air_end, air_info = _estimate_air_segment_from_ankle(
        ankle_y, min_air_frames=min_air_frames
    )
    quality.update(air_info)

    if air_start is None or air_end is None:
        # No confident airtime: everything unknown
        quality.setdefault("status", "fail")
        return PhaseResult(labels, None, None, None, apex_idx, apex_idx, quality)

    # airtime
    airtime = (air_end - air_start + 1) / float(max(fps, 1e-6))
    quality["airtime_seconds"] = float(airtime)

    # --- takeoff / landing indices ---
    # Contact is where ankles are "near" ground_y; we approximate:
    y = ankle_y
    y_max = float(np.nanmax(y))
    thr_contact = y_max - 0.05 * max(air_info.get("y_range", 1.0), 1.0)

    # Takeoff: last contact frame before air_start
    takeoff_idx = None
    for i in range(air_start - 1, -1, -1):
        if y[i] >= thr_contact:
            takeoff_idx = i
            break

    # Landing: first contact frame after air_end
    landing_idx = None
    for i in range(air_end + 1, T):
        if y[i] >= thr_contact:
            landing_idx = i
            break

    # Fallback if weird; but we already have air_start/end so we can still phase air.
    if takeoff_idx is None:
        takeoff_idx = max(0, air_start - 1)
    if landing_idx is None:
        landing_idx = min(T - 1, air_end + 1)

    quality["takeoff_idx"] = int(takeoff_idx)
    quality["landing_idx"] = int(landing_idx)

    # --- assign phases ---

    # 1) Air phases: early / mid / late air
    air_len = max(1, air_end - air_start + 1)
    e_end = air_start + int(0.30 * air_len)
    m_end = air_start + int(0.70 * air_len)

    e_end = min(e_end, air_end)
    m_end = min(max(e_end, m_end), air_end)

    for t in range(air_start, air_end + 1):
        if t <= e_end:
            labels[t] = "early_air"
        elif t <= m_end:
            labels[t] = "midair"
        else:
            labels[t] = "late_air"

    # 2) Landing window
    # Use a short but non-trivial window after landing_idx
    land_win = max(3, int(0.20 * air_len))
    land_end = min(T - 1, landing_idx + land_win)
    for t in range(landing_idx, land_end + 1):
        labels[t] = "landing"

    # 3) Set vs approach (pre-takeoff contact)
    # Heuristic: last ~40% of pre-takeoff frames is 'set', rest 'approach'
    if takeoff_idx > 0:
        pre_len = takeoff_idx + 1
        set_len = max(1, int(0.40 * pre_len))
        set_start = max(0, takeoff_idx + 1 - set_len)

        for t in range(0, set_start):
            labels[t] = "approach"
        for t in range(set_start, takeoff_idx + 1):
            labels[t] = "set"

    # 4) Clean up labels: fill gaps, handle NaN frames:
    labels = _fill_label_gaps(labels)

    quality.setdefault("status", "ok")
    return PhaseResult(
        labels=labels,
        takeoff_idx=int(takeoff_idx) if takeoff_idx is not None else None,
        landing_idx=int(landing_idx) if landing_idx is not None else None,
        air_start=int(air_start) if air_start is not None else None,
        air_end=int(air_end) if air_end is not None else None,
        apex_idx=int(apex_idx) if apex_idx is not None else None,
        quality=quality,
    )
