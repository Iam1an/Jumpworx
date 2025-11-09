from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional, Dict, Any

# MediaPipe landmark indices we reference
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
L_FTO, R_FTO = 31, 32

@dataclass
class Phases:
    approach: Tuple[int,int] | None
    set:      Tuple[int,int] | None
    airtime:  Tuple[int,int] | None
    landing:  Tuple[int,int] | None
    takeoff_idx:  int | None
    landing_idx:  int | None
    airtime_seconds: float | None
    quality: dict

# ---------- helpers ----------
def _ffill_bfill(a: np.ndarray) -> np.ndarray:
    """
    Fill NaNs along time axis in-place:
      - back-fill first frame from the first finite frame
      - forward-fill subsequent frames
    """
    out = a.copy()
    T = out.shape[0]
    # back-fill first frame
    first_finite = None
    for t in range(T):
        if np.isfinite(out[t]).any():
            first_finite = t
            break
    if first_finite is not None and first_finite > 0:
        out[:first_finite] = out[first_finite]
    # forward fill
    for t in range(1, T):
        bad = ~np.isfinite(out[t])
        if bad.any():
            out[t][bad] = out[t-1][bad]
    return out

def _moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.astype(np.float32)
    x = x.astype(np.float32)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    c = np.convolve(xp, np.ones(win, dtype=np.float32)/win, mode="valid")
    return c.astype(np.float32)

def _merge_short_gaps(contact: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill small 0-gaps within 1-runs up to max_gap length (binary array)."""
    x = contact.astype(np.uint8).copy()
    n = len(x)
    i = 0
    while i < n:
        if x[i] == 1:
            j = i
            while j < n and x[j] == 1:
                j += 1
            k = j
            while k < n and x[k] == 0:
                k += 1
            gap = k - j
            if 0 < gap <= max_gap and k < n and x[k] == 1:
                x[j:k] = 1
            i = k
        else:
            i += 1
    return x.astype(bool)

def _hysteresis_mask(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Return True for 'air' using Schmitt trigger hysteresis on y.
    Assumes *image-style* Y by default: smaller y = higher (up).
    Enter air when y < hi; exit air when y >= lo.
    """
    air = np.zeros_like(y, dtype=bool)
    prev_air = False
    for t, v in enumerate(y):
        if not np.isfinite(v):
            air[t] = prev_air
            continue
        if prev_air:
            prev_air = v < lo   # stay True until we cross back above/equal lo
        else:
            prev_air = v < hi   # enter when we go above jump threshold (smaller y)
        air[t] = prev_air
    return air

def _longest_span(mask: np.ndarray, min_len: int = 1) -> Tuple[int,int] | None:
    best = None; best_len = 0
    i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len and (j - i) > best_len:
                best = (i, j-1); best_len = j - i
            i = j
        else:
            i += 1
    return best

def _finite_percentile(x: np.ndarray, q: float, default: float) -> float:
    if np.isfinite(x).any():
        return float(np.nanpercentile(x, q))
    return float(default)

# ---------- public helpers (nice for viz) ----------
def labels_from_phases(ph: Phases | None, T: int) -> list[str]:
    """
    Build frame-wise labels: 'pre', 'takeoff', 'air', 'landing'.
    If phases is None or failed, returns all 'pre'.
    """
    labs = ["pre"] * T
    if ph is None or ph.quality.get("status") == "fail":
        return labs
    def mark(span, name):
        if span is None: return
        a,b = span
        a = max(0,int(a)); b = min(T-1,int(b))
        for i in range(a, b+1): labs[i] = name
    mark(ph.approach, "pre")
    mark(ph.set,      "pre")
    mark(ph.airtime,  "air")
    mark(ph.landing,  "landing")
    if ph.takeoff_idx is not None: labs[int(ph.takeoff_idx)] = "takeoff"
    if ph.landing_idx is not None: labs[int(ph.landing_idx)] = "landing"
    return labs

def events_from_phases(ph: Phases | None) -> Dict[str, Optional[int]]:
    return {
        "takeoff_idx": None if ph is None else ph.takeoff_idx,
        "landing_idx": None if ph is None else ph.landing_idx,
    }

# ---------- main API ----------
def segment_phases_with_airtime_v2(
    kps: np.ndarray,
    fps: float,
    smooth_win_ms: int = 50,
    hysteresis_frac: float = 0.20,
    max_gap_ms: int = 50,
    require_precontact: bool = True,
    min_precontact_ms: int = 250,
    min_air_ms: int = 120,
    invert_y: bool = False,
    return_debug: bool = False,
) -> Phases | tuple[Phases, Dict[str,Any]]:
    """
    Robust airtime/phase segmentation (works best if kps are pelvis-translated & torso-scaled,
    but it also works on pixel coords). Uses UNROTATED Y so 'up' ≈ decreasing y by default.

    kps: (T,33,3) or (T,33,2); NaN tolerated.

    Added:
      - min_air_ms: ignore micro flights shorter than this.
      - invert_y:   set True if your upstream is y-up (smaller y = lower).
      - return_debug: also return a dict with y-series, thresholds and masks.
    """
    assert kps.ndim == 3 and kps.shape[1] >= 29, "kps must be (T,33,>=2)"
    T = kps.shape[0]
    if kps.shape[2] == 2:
        kps3 = np.dstack([kps, np.zeros((T, kps.shape[1]), dtype=np.float32)])
    else:
        kps3 = kps.astype(np.float32)

    kps3 = _ffill_bfill(kps3)

    # Height signal: prefer LOWEST of toes/ankles per frame (robust to occlusion)
    toes = kps3[:, [L_FTO, R_FTO], 1]
    anks = kps3[:, [L_ANK, R_ANK], 1]

    toes_y = np.nanmean(toes, axis=1) if np.isfinite(toes).any() else np.full(T, np.nan, np.float32)
    anks_y = np.nanmean(anks, axis=1) if np.isfinite(anks).any() else np.full(T, np.nan, np.float32)

    both = np.stack([toes_y, anks_y], axis=1)  # (T,2)
    y_raw = np.nanmin(both, axis=1)  # lower y = higher in image (by default)
    if invert_y:
        y_raw = -y_raw

    # Smooth
    win = max(1, int(round((smooth_win_ms/1000.0) * fps)))
    y_s = _moving_mean(y_raw, win)

    # Percentile-based thresholds (robust to outliers)
    # Defaults: ground-ish 70th, air-ish 20th
    g = _finite_percentile(y_s, 70, default=np.nanmean(y_s) if np.isfinite(y_s).any() else 0.0)
    a = _finite_percentile(y_s, 20, default=g)

    band = abs(g - a) * float(hysteresis_frac)
    mid = (g + a) / 2.0
    hi_thr = mid - band/2.0  # entry (go up → smaller y)
    lo_thr = mid + band/2.0  # exit  (come down → larger y)

    air0 = _hysteresis_mask(y_s, lo_thr, hi_thr)

    # Optional: ensure some contact before first takeoff (soft check)
    if require_precontact and air0.any():
        pre = max(1, int(round((min_precontact_ms/1000.0) * fps)))
        first_true = int(np.argmax(air0))
        if first_true > 0:
            start_check = max(0, first_true - pre)
            # We only compute this to potentially log/warn; not used otherwise.
            _ = (~air0[start_check:first_true]).all()

    # Merge tiny contact gaps (bed push flicker)
    max_gap = max(1, int(round((max_gap_ms/1000.0) * fps)))
    contact = ~air0
    contact = _merge_short_gaps(contact, max_gap)
    air = ~contact

    # Ignore micro flights (optional)
    min_air_len = max(1, int(round((min_air_ms/1000.0) * fps)))
    span = _longest_span(air, min_len=min_air_len)
    if span is None:
        ph = Phases(
            approach=(0, T-1), set=None, airtime=None, landing=None,
            takeoff_idx=None, landing_idx=None, airtime_seconds=None,
            quality={"status":"fail","reason":"No airtime detected"}
        )
        if return_debug:
            dbg = {
                "y_raw": y_raw.tolist(),
                "y_smooth": y_s.tolist(),
                "hi_thr": hi_thr, "lo_thr": lo_thr,
                "air_mask": air.tolist(),
                "percentiles": {"ground": g, "air": a},
            }
            return ph, dbg
        return ph

    takeoff, land = span
    airtime_seconds = (land - takeoff + 1) / max(1e-6, fps)

    # Phases
    set_len = int(round(0.15 * fps))  # ~150 ms
    set_start = max(0, takeoff - set_len)
    approach = (0, max(0, set_start-1)) if set_start > 0 else (0, max(0, takeoff-1))
    set_phase = (set_start, takeoff)
    landing_len = int(round(0.2 * fps))
    landing_phase = (land, min(T-1, land + landing_len))

    # Quality flags
    quality = {"status":"ok","reason":""}
    if airtime_seconds < 0.15:
        quality = {"status":"warn","reason":"Very short airtime (<150ms)"}
    elif airtime_seconds > 2.0:
        quality = {"status":"warn","reason":"Very long airtime (>2.0s), verify detection"}

    ph = Phases(
        approach=approach, set=set_phase, airtime=span, landing=landing_phase,
        takeoff_idx=int(takeoff), landing_idx=int(land),
        airtime_seconds=float(airtime_seconds), quality=quality
    )

    if return_debug:
        dbg = {
            "y_raw": y_raw.tolist(),
            "y_smooth": y_s.tolist(),
            "hi_thr": hi_thr, "lo_thr": lo_thr,
            "air_mask": air.tolist(),
            "percentiles": {"ground": g, "air": a},
        }
        return ph, dbg
    return ph
