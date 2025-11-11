"""
jwcore.phase_segmentation
-------------------------

Unified phase and airtime detection module.

This supersedes `phase_detect.py` by combining its user-friendly
phase labeling with the robust airtime engine from
`segment_phases_with_airtime_v2`.

Usage (library):
    from jwcore.phase_segmentation import phase_result_from_pose
    res = phase_result_from_pose(kps, fps=30)
    print(res.labels)

Usage (CLI test):
    python jwcore/phase_segmentation.py path/to/pose.npz --fps 30
"""

from __future__ import annotations
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------

@dataclass
class Phases:
    """Core representation of jump phases."""
    approach: Optional[Tuple[int, int]]
    set: Optional[Tuple[int, int]]
    airtime: Optional[Tuple[int, int]]
    landing: Optional[Tuple[int, int]]
    takeoff_idx: Optional[int]
    landing_idx: Optional[int]
    airtime_seconds: Optional[float]
    quality: Dict[str, str]


@dataclass
class PhaseResult:
    """High-level user-facing result (legacy-compatible)."""
    labels: List[str]
    takeoff_idx: Optional[int]
    landing_idx: Optional[int]
    air_start: Optional[int]
    air_end: Optional[int]
    apex_idx: Optional[int]
    quality: Dict[str, str]


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _ffill_bfill(y: np.ndarray) -> np.ndarray:
    y = y.copy()
    isnan = np.isnan(y)
    if not np.any(isnan):
        return y
    idx = np.arange(len(y))
    not_nan = ~isnan
    y[isnan] = np.interp(idx[isnan], idx[not_nan], y[not_nan])
    return y


def _moving_mean(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[win:] - cumsum[:-win]) / float(win)


def _finite_percentile(y: np.ndarray, p: float) -> float:
    y = y[np.isfinite(y)]
    return np.percentile(y, p) if len(y) else np.nan


def _merge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Merge short 0-gaps in a binary mask."""
    m = mask.copy().astype(np.uint8)
    gaps = np.diff(np.pad(m, (1, 1), 'constant'))
    starts = np.where(gaps == -1)[0]
    ends = np.where(gaps == 1)[0]
    for s, e in zip(starts, ends):
        if e - s <= max_gap:
            m[s:e] = 1
    return m.astype(bool)


def _hysteresis_mask(y: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return mask of frames considered 'air' based on hysteresis thresholds."""
    m = np.zeros_like(y, dtype=bool)
    above = y < high
    below = y < low
    active = False
    for i in range(len(y)):
        if not active and above[i]:
            active = True
        elif active and not below[i]:
            active = False
        m[i] = active
    return m


# -------------------------------------------------------------------------
# Core function
# -------------------------------------------------------------------------

def segment_phases_with_airtime_v2(
    kps: np.ndarray,
    fps: float,
    smooth_win_ms: int = 150,
    min_air_ms: int = 150,
    max_gap_ms: int = 80,
    hysteresis_frac: float = 0.05,
    invert_y: bool = False,
    return_debug: bool = False,
) -> Tuple[Phases, Optional[Dict]]:
    """
    Robustly segment phases of a jump based on vertical landmarks.

    Parameters
    ----------
    kps : np.ndarray
        (T, J, 3) pose array.
    fps : float
        Frames per second.
    Returns
    -------
    (Phases, debug) tuple.
    """
    T = kps.shape[0]
    debug = {}
    # y extraction
    ankles = [27, 28]  # BlazePose L/R ankle
    toes = [31, 32]    # BlazePose L/R foot
    yvals = np.nanmean(kps[:, ankles + toes, 1], axis=1)
    if invert_y:
        yvals = -yvals
    yvals = _ffill_bfill(yvals)
    # smoothing
    win = max(1, int((smooth_win_ms / 1000.0) * fps))
    if win > 1 and win < len(yvals):
        yvals = np.convolve(yvals, np.ones(win) / win, mode="same")
    # thresholds
    ground_y = _finite_percentile(yvals, 70)
    air_y = _finite_percentile(yvals, 20)
    hysteresis = hysteresis_frac * (ground_y - air_y)
    air_mask = _hysteresis_mask(yvals, air_y - hysteresis, ground_y + hysteresis)
    # gap merge & micro-flight removal
    max_gap = int((max_gap_ms / 1000.0) * fps)
    air_mask = _merge_short_gaps(air_mask, max_gap)
    min_air = int((min_air_ms / 1000.0) * fps)
    # find longest air span
    padded = np.pad(air_mask.astype(int), (1, 1))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    spans = [(s, e) for s, e in zip(starts, ends) if e - s >= min_air]
    if not spans:
        return Phases(None, None, None, None, None, None, None,
                      {"status": "fail", "reason": "no_air"}), (debug if return_debug else None)
    air_start, air_end = max(spans, key=lambda x: x[1] - x[0])
    takeoff_idx, landing_idx = air_start, air_end
    airtime_s = (air_end - air_start) / fps
    # define approach/set windows (heuristic)
    set_win = int(0.1 * fps)
    approach_span = (max(0, air_start - 2 * set_win), max(0, air_start - set_win))
    set_span = (max(0, air_start - set_win), air_start)
    landing_span = (air_end, min(T, air_end + set_win))
    ph = Phases(
        approach=approach_span,
        set=set_span,
        airtime=(air_start, air_end),
        landing=landing_span,
        takeoff_idx=takeoff_idx,
        landing_idx=landing_idx,
        airtime_seconds=airtime_s,
        quality={"status": "ok"}
    )
    if return_debug:
        debug.update(dict(yvals=yvals, air_mask=air_mask))
        return ph, debug
    return ph, None


# -------------------------------------------------------------------------
# Label helpers
# -------------------------------------------------------------------------

def fine_labels_from_phases(ph: Phases, T: int) -> List[str]:
    """Return fine-grained labels for each frame."""
    labels = ["unknown"] * T
    if not ph.airtime:
        return labels
    a0, a1 = ph.airtime
    air_len = max(1, a1 - a0)
    e1 = a0 + int(0.3 * air_len)
    e2 = a0 + int(0.7 * air_len)
    if ph.approach:  # pre-jump
        for i in range(*ph.approach):
            labels[i] = "approach"
    if ph.set:
        for i in range(*ph.set):
            labels[i] = "set"
    for i in range(a0, e1):
        labels[i] = "early_air"
    for i in range(e1, e2):
        labels[i] = "midair"
    for i in range(e2, a1):
        labels[i] = "late_air"
    if ph.landing:
        for i in range(*ph.landing):
            labels[i] = "landing"
    return labels


def _apex_idx_from_y(yvals: np.ndarray, ph: Phases) -> Optional[int]:
    if not ph.airtime:
        return None
    a0, a1 = ph.airtime
    seg = yvals[a0:a1]
    if len(seg) == 0:
        return None
    return a0 + int(np.argmin(seg))


def phase_result_from_pose(kps: np.ndarray, fps: float) -> PhaseResult:
    """High-level API replacing detect_phases_from_pose."""
    ph, debug = segment_phases_with_airtime_v2(kps, fps, return_debug=True)
    yvals = debug["yvals"]
    labels = fine_labels_from_phases(ph, len(yvals))
    apex_idx = _apex_idx_from_y(yvals, ph)
    air_start, air_end = ph.airtime if ph.airtime else (None, None)
    return PhaseResult(
        labels=labels,
        takeoff_idx=ph.takeoff_idx,
        landing_idx=ph.landing_idx,
        air_start=air_start,
        air_end=air_end,
        apex_idx=apex_idx,
        quality=ph.quality
    )


# -------------------------------------------------------------------------
# CLI Test Harness
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick CLI test for phase segmentation.")
    parser.add_argument("pose_npz", help="Path to .posetrack.npz file")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    npz = np.load(args.pose_npz)
    kps = npz["P"]
    res = phase_result_from_pose(kps, fps=args.fps)
    print("Takeoff:", res.takeoff_idx, "Landing:", res.landing_idx)
    print("Airtime frames:", res.air_start, "→", res.air_end)
    print("Apex idx:", res.apex_idx)
    print("Quality:", res.quality)
    uniq_labels = sorted(set(res.labels))
    print("Unique labels:", uniq_labels)
